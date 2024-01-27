import gc
from functools import partial
from typing import Callable, List, Union

import einops
import torch
import tqdm
from jaxtyping import Float, Int
from torch import Tensor
from tqdm import tqdm

from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint

from eap.eap_graph import EAPGraph

def EAP_corrupted_forward_hook(
    activations: Union[Float[Tensor, "batch_size seq_len n_heads d_model"], Float[Tensor, "batch_size seq_len d_model"]],
    hook: HookPoint,
    upstream_activations_difference: Float[Tensor, "batch_size seq_len n_upstream_nodes d_model"], 
    graph: EAPGraph
):
    hook_slice = graph.get_hook_slice(hook.name)
    if activations.ndim == 3:
        # We are in the case of a residual layer or MLP
        # Activations have shape [batch_size, seq_len, d_model]
        # We need to add an extra dimension to make it [batch_size, seq_len, 1, d_model]
        # The hook slice is a slice of length 1
        upstream_activations_difference[:, :, hook_slice, :] = -activations.unsqueeze(-2)
    elif activations.ndim == 4:
        # We are in the case of an attention layer
        # Activations have shape [batch_size, seq_len, n_heads, d_model]
        upstream_activations_difference[:, :, hook_slice, :] = -activations

def EAP_clean_forward_hook(
    activations: Union[Float[Tensor, "batch_size seq_len n_heads d_model"], Float[Tensor, "batch_size seq_len d_model"]],
    hook: HookPoint,
    upstream_activations_difference: Float[Tensor, "batch_size seq_len n_upstream_nodes d_model"], 
    graph: EAPGraph
):
    hook_slice = graph.get_hook_slice(hook.name)
    if activations.ndim == 3:
        upstream_activations_difference[:, :, hook_slice, :] += activations.unsqueeze(-2)
    elif activations.ndim == 4:
        upstream_activations_difference[:, :, hook_slice, :] += activations

def EAP_clean_backward_hook(
    grad: Union[Float[Tensor, "batch_size seq_len n_heads d_model"], Float[Tensor, "batch_size seq_len d_model"]],
    hook: HookPoint,
    upstream_activations_difference: Float[Tensor, "batch_size seq_len n_upstream_nodes d_model"],
    graph: EAPGraph
):
    hook_slice = graph.get_hook_slice(hook.name)

    # we get the slice of all upstream nodes that come before this downstream node
    earlier_upstream_nodes_slice = graph.get_slice_previous_upstream_nodes(hook)

    # grad has shape [batch_size, seq_len, n_heads, d_model] or [batch_size, seq_len, d_model]
    # we want to multiply it by the upstream activations difference
    if grad.ndim == 3:
        grad_expanded = grad.unsqueeze(-2)  # Shape: [batch_size, seq_len, 1, d_model]
    else:
        grad_expanded = grad  # Shape: [batch_size, seq_len, n_heads, d_model]

    # we compute the mean over the batch_size and seq_len dimensions
    result = torch.matmul(
        upstream_activations_difference[:, :, earlier_upstream_nodes_slice],
        grad_expanded.transpose(-1, -2)
    ).sum(dim=0).sum(dim=0) # we sum over the batch_size and seq_len dimensions

    graph.eap_scores[earlier_upstream_nodes_slice, hook_slice] += result 

def EAP_downstream_patching_hook(
    activations: Union[Float[Tensor, "batch_size seq_len n_heads d_model"], Float[Tensor, "batch_size seq_len d_model"]],
    hook: HookPoint,
    upstream_activations_difference: Float[Tensor, "batch_size seq_len n_upstream_nodes d_model"],
    graph: EAPGraph,
) -> Union[Float[Tensor, "batch_size seq_len n_heads d_model"], Float[Tensor, "batch_size seq_len d_model"]]:
    hook_slice = graph.downstream_hook_slice[hook.name]

    earlier_upstream_nodes_slice = graph.get_slice_previous_upstream_nodes(hook)

    # The tensor 'patch_difference' represents the sum of all upstream activation differences that are connected to this downstream node
    patch_difference = einops.einsum(
        graph.adj_matrix[earlier_upstream_nodes_slice, hook_slice],
        upstream_activations_difference[:, :, earlier_upstream_nodes_slice, :],
        "n_upstream n_downstream_at_hook, batch_size seq_len n_upstream d_model -> batch_size seq_len n_downstream_at_hook d_model"
    )

    # alternatively, it might be faster to
    # 1) gather the activation differences for every non-zero element in adj_matrix[:, hook_slice] and sum them
    # 2) use torch.sparse.mm to multiply the adj_matrix with the activation differences

    if activations.ndim == 3:
        assert patch_difference.shape[-2] == 1, "Number of downstream nodes should be 1 for this type of hook" 
        # patched_residual_stream = clean_residual_stream - (clean_upstream_activations - corrupted_upstream_activations)
        activations -= patch_difference.squeeze(-2)
    elif activations.ndim == 4:
        activations -= patch_difference
    
    return activations


def EAP(
    model: HookedTransformer,
    clean_tokens: Int[Tensor, "batch_size seq_len"],
    corrupted_tokens: Int[Tensor, "batch_size seq_len"],
    metric: Callable,
    upstream_nodes: List[str]=None,
    downstream_nodes: List[str]=None,
    batch_size: int=1,
):

    graph = EAPGraph(model.cfg, upstream_nodes, downstream_nodes)

    assert clean_tokens.shape == corrupted_tokens.shape, "Shape mismatch between clean and corrupted tokens"
    num_prompts, seq_len = clean_tokens.shape[0], clean_tokens.shape[1]

    assert num_prompts % batch_size == 0, "Number of prompts must be divisible by batch size"

    upstream_activations_difference = torch.zeros(
        (batch_size, seq_len, graph.n_upstream_nodes, model.cfg.d_model),
        device=model.cfg.device,
        dtype=model.cfg.dtype,
        requires_grad=False
    )

    # set the EAP scores to zero
    graph.reset_scores()

    upstream_hook_filter = lambda name: name.endswith(tuple(graph.upstream_hooks))
    downstream_hook_filter = lambda name: name.endswith(tuple(graph.downstream_hooks))

    corruped_upstream_hook_fn = partial(
        EAP_corrupted_forward_hook,
        upstream_activations_difference=upstream_activations_difference,
        graph=graph
    )

    clean_upstream_hook_fn = partial(
        EAP_clean_forward_hook,
        upstream_activations_difference=upstream_activations_difference,
        graph=graph
    )

    clean_downstream_hook_fn = partial(
        EAP_clean_backward_hook,
        upstream_activations_difference=upstream_activations_difference,
        graph=graph
    )

    for idx in tqdm(range(0, num_prompts, batch_size)):
        # we first perform a forward pass on the corrupted input 
        model.add_hook(upstream_hook_filter, corruped_upstream_hook_fn, "fwd")

        # we don't need gradients for this forward pass
        # we'll take the gradients when we perform the forward pass on the clean input
        with torch.no_grad(): 
            corrupted_tokens = corrupted_tokens.to(model.cfg.device)
            model(corrupted_tokens[idx:idx+batch_size], return_type=None)        

        # now we perform a forward and backward pass on the clean input
        model.reset_hooks()
        model.add_hook(upstream_hook_filter, clean_upstream_hook_fn, "fwd")
        model.add_hook(downstream_hook_filter, clean_downstream_hook_fn, "bwd")

        clean_tokens = clean_tokens.to(model.cfg.device)
        value = metric(model(clean_tokens[idx:idx+batch_size], return_type="logits"))
        value.backward()
        
        # We delete the activation differences tensor to free up memory
        model.zero_grad()
        upstream_activations_difference *= 0

    del upstream_activations_difference
    gc.collect()
    torch.cuda.empty_cache()
    model.reset_hooks()

    graph.eap_scores /= num_prompts
    graph.eap_scores = graph.eap_scores.cpu()

    return graph
