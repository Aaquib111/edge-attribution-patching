#%%

"""
Boilerplate taken from acdcpp_docstring.ipynb
"""

import os
import sys
from tqdm import tqdm
sys.path.append('..')
sys.path.append('../Automatic-Circuit-Discovery/')
sys.path.append('../tracr/')
import IPython
from IPython import get_ipython
ipython = get_ipython()
if ipython is not None:
    ipython.magic('load_ext autoreload')
    ipython.magic('autoreload 2')
import torch as t
import torch
from transformer_lens import HookedTransformer
import gc
from torch import Tensor
import matplotlib.pyplot as plt
from jaxtyping import Float
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
from ACDCPPExperiment import ACDCPPExperiment
from acdc.docstring.utils import get_all_docstring_things
from acdc.TLACDCExperiment import TLACDCExperiment

device = t.device("cuda" if t.cuda.is_available() else "CPU")
print(device)

#%%

model = HookedTransformer.from_pretrained(
    'gpt2-small',
    center_writing_weights=False,
    center_unembed=False,
    fold_ln=False,
    device=device,
)
model.set_use_hook_mlp_in(True)
model.set_use_split_qkv_input(True)
model.set_use_attn_result(True)

# %%

from ioi_task.ioi_dataset import IOIDataset, format_prompt, make_table
N = 25
clean_dataset = IOIDataset(
    prompt_type='mixed',
    N=N,
    tokenizer=model.tokenizer,
    prepend_bos=False,
    seed=1,
    device=device
)
corr_dataset = clean_dataset.gen_flipped_prompts('ABC->XYZ, BAB->XYZ')

make_table(
  colnames = ["IOI prompt", "IOI subj", "IOI indirect obj", "ABC prompt"],
  cols = [
    map(format_prompt, clean_dataset.sentences),
    model.to_string(clean_dataset.s_tokenIDs).split(),
    model.to_string(clean_dataset.io_tokenIDs).split(),
    map(format_prompt, clean_dataset.sentences),
  ],
  title = "Sentences from IOI vs ABC distribution",
)

def ave_logit_diff(
    logits: Float[Tensor, 'batch seq d_vocab'],
    ioi_dataset: IOIDataset,
    per_prompt: bool = False
):
    '''
        Return average logit difference between correct and incorrect answers
    '''
    # Get logits for indirect objects
    io_logits = logits[range(logits.size(0)), ioi_dataset.word_idx['end'], ioi_dataset.io_tokenIDs]
    s_logits = logits[range(logits.size(0)), ioi_dataset.word_idx['end'], ioi_dataset.s_tokenIDs]
    # Get logits for subject
    logit_diff = io_logits - s_logits
    return logit_diff if per_prompt else logit_diff.mean()

with t.no_grad():
    clean_logits = model(clean_dataset.toks)
    corrupt_logits = model(corr_dataset.toks)
    clean_logit_diff = ave_logit_diff(clean_logits, clean_dataset).item()
    corrupt_logit_diff = ave_logit_diff(corrupt_logits, corr_dataset).item()

def ioi_metric(
    logits: Float[Tensor, "batch seq_len d_vocab"],
    corrupted_logit_diff: float = corrupt_logit_diff,
    clean_logit_diff: float = clean_logit_diff,
    ioi_dataset: IOIDataset = clean_dataset
 ):
    patched_logit_diff = ave_logit_diff(logits, ioi_dataset)
    return (patched_logit_diff - corrupted_logit_diff) / (clean_logit_diff - corrupted_logit_diff)

def abs_ioi_metric(logits: Float[Tensor, "batch seq_len d_vocab"]):
    return abs(ioi_metric(logits))

def negative_ioi_metric(logits: Float[Tensor, "batch seq_len d_vocab"]):
    return -ioi_metric(logits)

def negative_abs_ioi_metric(logits: Float[Tensor, "batch seq_len d_vocab"]):
    return -abs_ioi_metric(logits)

# Get clean and corrupt logit differences
with t.no_grad():
    clean_metric = ioi_metric(clean_logits, corrupt_logit_diff, clean_logit_diff, clean_dataset)
    corrupt_metric = ioi_metric(corrupt_logits, corrupt_logit_diff, clean_logit_diff, corr_dataset)

print(f'Clean direction: {clean_logit_diff}, Corrupt direction: {corrupt_logit_diff}')
print(f'Clean metric: {clean_metric}, Corrupt metric: {corrupt_metric}')

#%%

threshold_dummy = -100.0 # Does not make a difference when only running edge based attribution patching, as all attributions are saved in the result dict anyways
RUN_NAME = 'greaterthan_edge_absval'
model.reset_hooks()
acdcpp_exp = ACDCPPExperiment(
    model,
    clean_dataset.toks,
    corr_dataset.toks,
    negative_ioi_metric,
    negative_ioi_metric,
    [threshold_dummy],
    run_name=RUN_NAME,
    verbose=False,
    attr_absolute_val=False,
    save_graphs_after=0,
    pruning_mode="edge",
    no_pruned_nodes_attr=1
)
acdc_exp = acdcpp_exp.setup_exp(threshold=threshold_dummy)
nodes, attr_results = acdcpp_exp.run_acdcpp(exp=acdc_exp, threshold=threshold_dummy)

#%%

sorted_ap_attr = sorted(attr_results.items(), key=lambda x: abs(x[1]), reverse=True)

#%%

for idx in range(2, 5):
    (sender_component, receiver_component), ap_val = sorted_ap_attr[idx]
    print(
        f"Sender component: {sender_component}, receiver component: {receiver_component}, ap_val: {ap_val}"
    )

    sender_node_name, sender_node_index = sender_component.hook_point_name, sender_component.index
    receiver_node_name, receiver_node_index = receiver_component.hook_point_name, receiver_component.index

    original_corrupted_cache_value_cpu = acdc_exp.global_cache.corrupted_cache[sender_node_name][sender_node_index.as_index].cpu()
    original_online_cache_value_cpu = acdc_exp.global_cache.online_cache[sender_node_name][sender_node_index.as_index].cpu()

    acdc_exp.update_cur_metric()
    original_metric = acdc_exp.cur_metric
    original_edges = acdc_exp.count_no_edges()

    edge=acdc_exp.corr.edges[receiver_node_name][receiver_node_index][sender_node_name][sender_node_index]
    edge.mask = 1.0
    
    if "addition" in str(edge.edge_type).lower():
        ret = acdc_exp.add_receiver_hook(acdc_exp.corr.graph[receiver_node_name][receiver_node_index], override=False, prepend=True)
    elif "direct" in str(edge.edge_type).lower():
        ret = acdc_exp.add_receiver_hook(acdc_exp.corr.graph[receiver_node_name][receiver_node_index], override=True, prepend=True)
        added_sender_hook = acdc_exp.add_sender_hook(acdc_exp.corr.graph[receiver_node_name][receiver_node_index], override=True)
    else:
        raise ValueError(f"Unknown edge type: {edge.edge_type}")

    if not ret: 
        print("Warning, hook already exists" + str(edge.edge_type))

    acdc_exp.update_cur_metric()
    new_metric = acdc_exp.cur_metric

    print(
        f"Original metric: {original_metric:.10f}, new metric: {new_metric:.10f}"
    )

    interpolated_metrics = []
    for interpolation in torch.linspace(0, 1, 101):
        edge.mask = interpolation
        acdc_exp.update_cur_metric()
        intermediate_metric = acdc_exp.cur_metric
        interpolated_metrics.append(intermediate_metric)
    edge.mask = 0.0

    # Plot the interpolated metrics
    # Clear fig for next iteration
    plt.clf()
    plt.figure(figsize=(10, 6))  # Adjust the dimensions as necessary
    plt.plot(torch.linspace(0, 1, len(interpolated_metrics)), interpolated_metrics)
    plt.xlabel('Interpolation')
    plt.ylabel('Metric')
    plt.title('Interpolated metric')

    # Label the x=0 point as "Clean edge"

    plt.annotate(
        'Clean edge',
        xy=(0, interpolated_metrics[0]),
        xytext=(0.2, interpolated_metrics[0]+0.05),
        # Skinnier arrow and tip
        arrowprops=dict(facecolor='black', shrink=0.05, width=0.5, headwidth=7),
    )

    plt.annotate(
        'Corrupted edge',
        xy=(1.0, interpolated_metrics[-1]),
        xytext=(0.5, interpolated_metrics[-1]+0.05),
        # Skinnier arrow and tip
        arrowprops=dict(facecolor='black', shrink=0.05, width=0.5, headwidth=7),    
    )

    # Approximate the tangent line at x=0 and extrapolate to x=1
    slope_at_zero = (interpolated_metrics[1] - interpolated_metrics[0]) * (len(interpolated_metrics)-1)  # Assuming a step size of 0.01
    tangent_line = slope_at_zero * torch.linspace(0, 1, len(interpolated_metrics)) + interpolated_metrics[0]
    plt.plot(torch.linspace(0, 1, len(interpolated_metrics)), tangent_line, linestyle='--')
    plt.plot(1.0, interpolated_metrics[0] - ap_val, marker='o', color='red') # Kind of annoying property that it seems be negative?
    plt.xlabel('Interpolation towards corruption')
    plt.ylabel('Metric')
    plt.title(f'Corrupting edge {sender_node_name}{sender_node_index} -> {receiver_node_name}{receiver_node_index} (AP value: {-ap_val:.10f})')
    fname = os.path.expanduser(f"~/acdcpp/ioi_task/edge_{idx}.png")

    # Save the figure with a widened layout
    plt.tight_layout()
    plt.savefig(fname)
    plt.show()

# %%
