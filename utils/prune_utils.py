import sys
sys.path.append('Automatic-Circuit-Discovery/')

from acdc.TLACDCExperiment import TLACDCExperiment
from acdc.acdc_utils import TorchIndex, EdgeType
import torch as t
from torch import Tensor
import einops
import itertools

from transformer_lens import HookedTransformer, ActivationCache

import tqdm.notebook as tqdm

from jaxtyping import Bool
from typing import Callable, Tuple 

from utils.graphics_utils import get_node_name

def remove_redundant_node(exp, node, safe=True, allow_fails=True):
        if safe:
            for parent_name in exp.corr.edges[node.name][node.index]:
                for parent_index in exp.corr.edges[node.name][node.index][parent_name]:
                    if exp.corr.edges[node.name][node.index][parent_name][parent_index].present:
                        raise Exception(f"You should not be removing a node that is still used by another node {node} {(parent_name, parent_index)}")

        bfs = [node]
        bfs_idx = 0

        while bfs_idx < len(bfs):
            cur_node = bfs[bfs_idx]
            bfs_idx += 1

            children = exp.corr.graph[cur_node.name][cur_node.index].children

            for child_node in children:
                if not cur_node.index in exp.corr.edges[child_node.name][child_node.index][cur_node.name]:
                    #print(f'\t CANT remove edge {cur_node.name}, {cur_node.index} <-> {child_node.name}, {child_node.index}')
                    continue
                    
                try:
                    #print(f'\t Removing edge {cur_node.name}, {cur_node.index} <-> {child_node.name}, {child_node.index}')
                    exp.corr.remove_edge(
                        child_node.name, child_node.index, cur_node.name, cur_node.index
                    )
                except KeyError as e:
                    print("Got an error", e)
                    if allow_fails:
                        continue
                    else:
                        raise e

                remove_this = True
                for parent_of_child_name in exp.corr.edges[child_node.name][child_node.index]:
                    for parent_of_child_index in exp.corr.edges[child_node.name][child_node.index][parent_of_child_name]:
                        if exp.corr.edges[child_node.name][child_node.index][parent_of_child_name][parent_of_child_index].present:
                            remove_this = False
                            break
                    if not remove_this:
                        break

                if remove_this and child_node not in bfs:
                    bfs.append(child_node)

def remove_node(exp, node):
    '''
        Method that removes node from model. Assumes children point towards
        the end of the residual stream and parents point towards the beginning.

        exp: A TLACDCExperiment object with a reverse top sorted graph
        node: A TLACDCInterpNode describing the node to remove
        root: Initally the first node in the graph
    '''
    #Removing all edges pointing to the node
    remove_edges = []
    for p_name in exp.corr.edges[node.name][node.index]:
        for p_idx in exp.corr.edges[node.name][node.index][p_name]:
            edge = exp.corr.edges[node.name][node.index][p_name][p_idx]
            remove_edges.append((node.name, node.index, p_name, p_idx))
            edge.present = False
    for n_name, n_idx, p_name, p_idx in remove_edges:
        #print(f'\t Removing edge {p_name}, {p_idx} <-> {n_name}, {n_idx}')
        exp.corr.remove_edge(
            n_name, n_idx, p_name, p_idx
        )
    # Removing all outgoing edges from the node using BFS
    remove_redundant_node(exp, node, safe=False)

def find_attn_node(exp, layer, head):
    return exp.corr.graph[f'blocks.{layer}.attn.hook_result'][TorchIndex([None, None, head])]

def find_attn_node_qkv(exp, layer, head):
    nodes = []
    for qkv in ['q', 'k', 'v']:
        nodes.append(exp.corr.graph[f'blocks.{layer}.attn.hook_{qkv}'][TorchIndex([None, None, head])])
        nodes.append(exp.corr.graph[f'blocks.{layer}.hook_{qkv}_input'][TorchIndex([None, None, head])])
    return nodes
    
def split_layers_and_heads(act: Tensor, model: HookedTransformer) -> Tensor:
    return einops.rearrange(act, '(layer head) batch seq d_model -> layer head batch seq d_model',
                            layer=model.cfg.n_layers,
                            head=model.cfg.n_heads)

hook_filter = lambda name: name.endswith("ln1.hook_normalized") or name.endswith("attn.hook_result")
def get_3_caches(model, clean_input, corrupted_input, metric):
    # cache the activations and gradients of the clean inputs
    model.reset_hooks()
    clean_cache = {}

    def forward_cache_hook(act, hook):
        clean_cache[hook.name] = act.detach()

    model.add_hook(hook_filter, forward_cache_hook, "fwd")

    clean_grad_cache = {}

    def backward_cache_hook(act, hook):
        clean_grad_cache[hook.name] = act.detach()

    model.add_hook(hook_filter, backward_cache_hook, "bwd")

    value = metric(model(clean_input))
    value.backward()

    # cache the activations of the corrupted inputs
    model.reset_hooks()
    corrupted_cache = {}

    def forward_cache_hook(act, hook):
        corrupted_cache[hook.name] = act.detach()

    model.add_hook(hook_filter, forward_cache_hook, "fwd")
    model(corrupted_input)
    model.reset_hooks()

    clean_cache = ActivationCache(clean_cache, model)
    corrupted_cache = ActivationCache(corrupted_cache, model)
    clean_grad_cache = ActivationCache(clean_grad_cache, model)
    return clean_cache, corrupted_cache, clean_grad_cache

def get_nodes(correspondence):
    nodes = set()
    for child_hook_name in correspondence.edges:
        for child_index in correspondence.edges[child_hook_name]:
            for parent_hook_name in correspondence.edges[child_hook_name][child_index]:
                for parent_index in correspondence.edges[child_hook_name][child_index][parent_hook_name]:
                    edge = correspondence.edges[child_hook_name][child_index][parent_hook_name][parent_index]

                    parent = correspondence.graph[parent_hook_name][parent_index]
                    child = correspondence.graph[child_hook_name][child_index]

                    parent_name = get_node_name(parent, show_full_index=False)
                    child_name = get_node_name(child, show_full_index=False)
                    
                    if any(qkv in child_name or qkv in parent_name for qkv in ['_q_', '_k_', '_v_']):
                        continue
                    parent_name = parent_name.replace("_q>", ">").replace("_k>", ">").replace("_v>", ">")
                    child_name = child_name.replace("_q>", ">").replace("_k>", ">").replace("_v>", ">")

                    if parent_name == child_name:
                        # Important this go after the qkv removal
                        continue
                    
                    if edge.present and edge.edge_type != EdgeType.PLACEHOLDER:
                        #print(f'Edge from {parent_name=} to {child_name=}')
                        for node_name in [parent_name, child_name]:
                            nodes.add(node_name)
    return nodes

def acdc_nodes(model: HookedTransformer,
              clean_input: Tensor,
              corrupted_input: Tensor,
              metric: Callable[[Tensor], Tensor],
              threshold: float,
              exp: TLACDCExperiment,
              verbose: bool = False,
              attr_absolute_val: bool = False) -> Tuple[
                  HookedTransformer, Bool[Tensor, 'n_layer n_heads']]:
    '''
    Runs attribution-patching-based ACDC on the model, using the given metric and data.
    Returns the pruned model, and which heads were pruned.

    Arguments:
        model: the model to prune
        clean_input: the input to the model that contains should elicit the behavior we're looking for
        corrupted_input: the input to the model that should elicit random behavior
        metric: the metric to use to compare the model's performance on the clean and corrupted inputs
        threshold: the threshold below which to prune
        create_model: a function that returns a new model of the same type as the input model
        attr_absolute_val: whether to take the absolute value of the attribution before thresholding
    '''
    # get the 2 fwd and 1 bwd caches; cache "normalized" and "result" of attn layers
    clean_cache, corrupted_cache, clean_grad_cache = get_3_caches(model, clean_input, corrupted_input, metric)

    # compute first-order Taylor approximation for each node to get the attribution
    clean_head_act = clean_cache.stack_head_results()
    corr_head_act = corrupted_cache.stack_head_results()
    clean_grad_act = clean_grad_cache.stack_head_results()

    # compute attributions of each node
    node_attr = (clean_head_act - corr_head_act) * clean_grad_act
    # separate layers and heads, sum over d_model (to complete the dot product), batch, and seq
    node_attr = split_layers_and_heads(node_attr, model).sum((2, 3, 4))

    if attr_absolute_val:
        node_attr = node_attr.abs()
    del clean_cache
    del clean_head_act
    del corrupted_cache
    del corr_head_act
    del clean_grad_cache
    del clean_grad_act
    t.cuda.empty_cache()
    # prune all nodes whose attribution is below the threshold
    should_prune = node_attr < threshold
    pruned_nodes_attr = {}
    for layer, head in itertools.product(range(model.cfg.n_layers), range(model.cfg.n_heads)):
        if should_prune[layer, head]:
            # REMOVING NODE
            if verbose:
                print(f'PRUNING L{layer}H{head} with attribution {node_attr[layer, head]}')
            # Find the corresponding node in computation graph
            node = find_attn_node(exp, layer, head)
            if verbose:
                print(f'\tFound node {node.name}')
            # Prune node
            remove_node(exp, node)
            if verbose:
                print(f'\tRemoved node {node.name}')
            pruned_nodes_attr[(layer, head)] = node_attr[layer, head].item()
            
            # REMOVING QKV
            qkv_nodes = find_attn_node_qkv(exp, layer, head)
            for node in qkv_nodes:
                remove_node(exp, node)
    return pruned_nodes_attr