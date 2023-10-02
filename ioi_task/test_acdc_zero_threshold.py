#%%

"""Copy and pasted from acdc repo launch_sixteen_heads.py, but with the following changes:"""

import math
from IPython import get_ipython

if get_ipython() is not None:
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')

import argparse
import gc
from copy import deepcopy
import os
import numpy as np
import plotly.graph_objects as go
import json
import sys
from tqdm import tqdm
sys.path.append('..')
sys.path.append('../Automatic-Circuit-Discovery/')
sys.path.append('../tracr/')
import IPython
import plotly.express as px

import torch
import wandb
from transformer_lens import HookedTransformer 
import torch as t
import torch
from transformer_lens import HookedTransformer
import gc
from torch import Tensor
import matplotlib.pyplot as plt
from jaxtyping import Float
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
from ACDCPPExperiment import ACDCPPExperiment
from acdc.docstring.utils import get_all_docstring_things, get_docstring_subgraph_true_edges
from acdc.TLACDCExperiment import TLACDCExperiment
from acdc.TLACDCCorrespondence import TLACDCCorrespondence
from acdc.TLACDCInterpNode import TLACDCInterpNode
from acdc.acdc_utils import (
    cleanup,
    ct,
    get_roc_figure,
    kl_divergence,
    make_nd_dict,
    shuffle_tensor,
    get_points,
)

from acdc.TLACDCEdge import (
    Edge,
    EdgeType,
    TorchIndex,
)

from acdc.acdc_utils import reset_network
from acdc.docstring.utils import get_all_docstring_things
from acdc.greaterthan.utils import get_all_greaterthan_things
from acdc.induction.utils import (
    get_all_induction_things,
    get_good_induction_candidates,
    get_mask_repeat_candidates,
    get_validation_data,
)
from acdc.ioi.utils import get_all_ioi_things
from acdc.TLACDCExperiment import TLACDCExperiment
from acdc.TLACDCInterpNode import TLACDCInterpNode, heads_to_nodes_to_mask
from acdc.tracr_task.utils import get_all_tracr_things
from subnetwork_probing.train import iterative_correspondence_from_mask
from notebooks.emacs_plotly_render import set_plotly_renderer

from subnetwork_probing.transformer_lens.transformer_lens.HookedTransformer import HookedTransformer as SPHookedTransformer
from subnetwork_probing.transformer_lens.transformer_lens.HookedTransformerConfig import HookedTransformerConfig as SPHookedTransformerConfig
from subnetwork_probing.train import do_random_resample_caching, do_zero_caching
from subnetwork_probing.transformer_lens.transformer_lens.hook_points import MaskedHookPoint
from acdc.induction.utils import (
    get_all_induction_things,
)

#%%

TASK: Literal["docstring", "ioi", "induction"] = "induction"

#%%

device = t.device("cuda" if t.cuda.is_available() else "CPU")

if TASK == "induction":
    num_examples = 50
    seq_len = 300
    things = get_all_induction_things(
        num_examples=num_examples, seq_len=seq_len, device=device, metric="nll", # nll seems best for AP comparison
    )
    model=things.tl_model

elif TASK == "ioi":
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

    from ioi_task.ioi_dataset import IOIDataset, format_prompt, make_table
    N = 25
    clean_dataset = IOIDataset(
        prompt_type='mixed',
        N=N,
        tokenizer=model.tokenizer,
        prepend_bos=False,
        seed=1,
        device=device,
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

elif TASK == "docstring":
    things = get_all_docstring_things(num_examples=40, seq_len=5, device=device, metric_name='docstring_metric', correct_incorrect_wandb=False)

    model = things.tl_model
    validation_metric = things.validation_metric
    validation_data = things.validation_data
    validation_labels = things.validation_labels
    validation_patch_data = things.validation_patch_data
    test_metrics = things.test_metrics
    test_data = things.test_data
    test_labels = things.test_labels
    test_patch_data = things.test_patch_data

    def abs_docstring_metric(logits):
        return -abs(test_metrics['docstring_metric'](logits))

else:
    raise ValueError(f"Unknown task: {TASK}")

#%%

threshold_dummy = -100.0 # Does not make a difference when only running edge based attribution patching, as all attributions are saved in the result dict anyways
RUN_NAME = 'docstring_and_ioi_noddling'

model.reset_hooks()

acdcpp_exp = ACDCPPExperiment(
    model,
    clean_dataset.toks if TASK == "ioi" else things.validation_data,
    corr_dataset.toks if TASK == "ioi" else things.validation_patch_data,
    acdc_metric=negative_ioi_metric if TASK == "ioi" else things.validation_metric,
    acdcpp_metric=negative_ioi_metric if TASK == "ioi" else things.validation_metric, # TODO do abs_docstring thing???
    thresholds=[threshold_dummy],
    run_name=RUN_NAME,
    verbose=False,
    zero_ablation=False,
    attr_absolute_val=True,
    save_graphs_after=0,
    pruning_mode="edge",
    no_pruned_nodes_attr=1
)
acdc_exp = acdcpp_exp.setup_exp(threshold=threshold_dummy)
do_mean_ablation = False
if do_mean_ablation:
    corrupted_cache_keys = list(acdc_exp.global_cache.corrupted_cache.keys())
    for key in corrupted_cache_keys:
        acdc_exp.global_cache.corrupted_cache[key] = acdc_exp.global_cache.corrupted_cache[key].mean(dim=0, keepdim=True)
nodes, attr_results = acdcpp_exp.run_acdcpp(exp=acdc_exp, threshold=threshold_dummy)

#%%

ap_attr_to_remove = sorted(attr_results.items(), key=lambda x: abs(x[1]), reverse=False)

#%%

if TASK == "docstring":
    true_edges = get_docstring_subgraph_true_edges()
    included_edges = [e for e, present in true_edges.items() if present]
    cnt=0
    for edge_tuple, e in list(acdc_exp.corr.all_edges().items()):
        e.present=False
        hashable_edge_tuple = (edge_tuple[0], edge_tuple[1].hashable_tuple, edge_tuple[2], edge_tuple[3].hashable_tuple)
        if hashable_edge_tuple in included_edges:
            e.present=True
            cnt+=1
    assert cnt==len(included_edges), f"Expected {len(included_edges)} edges to be included, but got {cnt}"
    ground_truth = deepcopy(acdc_exp.corr)
    for e in list(acdc_exp.corr.all_edges().values()):
        e.present=True

#%%

# Resetup experiment with test stuff

test_acdcpp_exp = ACDCPPExperiment(
    model,
    clean_dataset.toks if TASK == "ioi" else things.test_data,
    corr_dataset.toks if TASK == "ioi" else things.test_patch_data,
    acdc_metric=negative_ioi_metric if TASK == "ioi" else things.validation_metric,
    acdcpp_metric=negative_ioi_metric if TASK == "ioi" else things.validation_metric, # TODO do abs_docstring thing???
    thresholds=[threshold_dummy],
    run_name=RUN_NAME,
    verbose=False,
    zero_ablation=False,
    attr_absolute_val=True,
    save_graphs_after=0,
    pruning_mode="edge",
    no_pruned_nodes_attr=1
)
test_acdc_exp = test_acdcpp_exp.setup_exp(threshold=threshold_dummy)
acdc_exp.model.reset_hooks()
acdc_exp.setup_model_hooks(
    add_sender_hooks=True,
    add_receiver_hooks=True,
    doing_acdc_runs=False,
)

#%%

num_edges = []
test_metrics = []
corrs = []

max_subgraph_size = acdc_exp.count_no_edges()
for i in tqdm(range(len(ap_attr_to_remove))):
    # Remove the ith least important edge
    (sender_component, receiver_component), ap_val = ap_attr_to_remove[i]

    acdc_exp.corr.edges[receiver_component.hook_point_name][receiver_component.index][sender_component.hook_point_name][sender_component.index].present=False

    # Calculate Test KL
    cur_test_metrics = {}
    for name, fn in things.test_metrics.items():
        cur_test_metrics["test_"+name] = fn(acdc_exp.model(things.test_data)).item()
    test_metrics.append(cur_test_metrics)
    
    print(
        test_metrics[-1],
    )

    corrs.append(deepcopy(acdc_exp.corr))

    # If we now have something disconnected, add it to a BFS and remove everything
    # TODO?

#%%

points = get_points(
    [(c, {"score": 0.0}) for c in corrs], # Can just put empty dicts is we don't care about this?
    task=TASK,
    canonical_circuit_subgraph=ground_truth,
    canonical_circuit_subgraph_size=len(included_edges),
    max_subgraph_size=max_subgraph_size,
    decreasing=True,
)

#%%

def discard_non_pareto_optimal(points, auxiliary, cmp="gt"):
    ret = []
    for (x, y), aux in zip(points, auxiliary):
        for x1, y1 in points:
            if x1 < x and getattr(y1, f"__{cmp}__")(y) and (x1, y1) != (x, y):
                break
        else:
            ret.append(((x, y), aux))
    return list(sorted(ret))

xy_points = [(p["edge_fpr"], p["edge_tpr"]) for p in points]
filtered_points = discard_non_pareto_optimal(xy_points, [{} for _ in points], cmp="gt")
remmed_second = [x[0] for x in filtered_points]

#%%

fig = get_roc_figure([remmed_second], ["AP"])
# Save fig as PNG
fig.write_image(f"roc_remove_{RUN_NAME}.png")

#%%

json_metric_name = "kl_div" # TODO: Change this to whatever metric we want to plot
method_name = "16h"
json_fname = os.path.expanduser(f"~/Automatic-Circuit-Discovery/experiments/results/plots_data/{method_name}-induction-{json_metric_name}-False-0.json")
with open(json_fname, "r") as f:
    data = json.load(f)

filtered_data = data["trained"]["random_ablation"]["induction"][json_metric_name][method_name.upper()] # Baseline

# %%

metric_name = "nll"

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=filtered_data["n_edges"],
        y=filtered_data[f"test_{metric_name}"],
        mode="markers",
        name="ACDC on NLL", # Rename from NLL when we use different files
    )
)
fig.add_trace(
    go.Scatter(
        x = list(range(len(test_metrics)-1, -1, -1)),
        y = [x[f"test_{metric_name}"] for x in test_metrics],
        mode="markers",
        name="AP on NLL",
    )
)
fig.update_layout(
    title=f"Test {metric_name} vs. Number of Edges",
    xaxis_title="Number of Edges",
    yaxis_title=f"Test {metric_name}",
)
fig.show()

# %%
