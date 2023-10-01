#%%

"""
Boilerplate taken from acdcpp_docstring.ipynb
"""

import os
import sys
sys.path.append('..')
sys.path.append('../Automatic-Circuit-Discovery/')
sys.path.append('../tracr/')
import IPython
ipython = get_ipython()
if ipython is not None:
    ipython.magic('load_ext autoreload')
    ipython.magic('autoreload 2')
import torch as t
import torch
import gc
from torch import Tensor
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
from ACDCPPExperiment import ACDCPPExperiment
from acdc.docstring.utils import get_all_docstring_things
from acdc.TLACDCExperiment import TLACDCExperiment

device = t.device("cuda" if t.cuda.is_available() else "CPU")
print(device)

#%%

all_docstring_items = get_all_docstring_things(num_examples=40, seq_len=5, device=device, metric_name='docstring_metric', correct_incorrect_wandb=False)

tl_model = all_docstring_items.tl_model
validation_metric = all_docstring_items.validation_metric
validation_data = all_docstring_items.validation_data
validation_labels = all_docstring_items.validation_labels
validation_patch_data = all_docstring_items.validation_patch_data
test_metrics = all_docstring_items.test_metrics
test_data = all_docstring_items.test_data
test_labels = all_docstring_items.test_labels
test_patch_data = all_docstring_items.test_patch_data

# %%

def abs_docstring_metric(logits):
    return -abs(test_metrics['docstring_metric'](logits))

#%%

tl_model.reset_hooks()
RUN_NAME = 'abs_edges'
acdcpp_exp = ACDCPPExperiment(
    tl_model,
    test_data,
    test_patch_data,
    test_metrics['docstring_metric'],
    abs_docstring_metric,
    thresholds=[-100.0],
    run_name="arthur_acdcpp_approx",
    verbose=True,
    attr_absolute_val=True,
    save_graphs_after=0,
    pruning_mode='edge',
    no_pruned_nodes_attr=1,
    using_wandb=False,
)
acdc_exp = acdcpp_exp.setup_exp(-100.0)
# (We tested this gives what we want) 

#%%

ap_attr: Dict[Any, float]

# Do attribution patching
nodes, ap_attr = acdcpp_exp.run_acdcpp(acdc_exp, threshold = -100.0) # Ablate nothing 

# %%

sorted_ap_attr = sorted(ap_attr.items(), key=lambda x: x[1], reverse=True)

# %%

# From here likely we're going to wrap this in a function
# TODO to reset the online cache, probably we need resetup a TLACDCExperiment???

sender_component, receiver_component = sorted_ap_attr[1][0]

# %%

sender_node_name, sender_node_index = sender_component.hook_point_name, sender_component.index

# %%

receiver_node_name, receiver_node_index = receiver_component.hook_point_name, receiver_component.index

# %%

original_corrupted_cache_value_cpu = acdc_exp.global_cache.corrupted_cache[sender_node_name][sender_node_index.as_index].cpu()

# %%

original_metric = acdc_exp.cur_metric
original_edges = acdc_exp.count_no_edges()

# %%

acdc_exp.add_receiver_hook(acdc_exp.corr.graph[receiver_node_name][receiver_node_index], override=True, prepend=True)

#%%

acdc_exp.corr.edges[receiver_node_name][receiver_node_index][sender_node_name][sender_node_index].present = False

# %%

acdc_exp.update_cur_metric()
new_metric = acdc_exp.cur_metric

# %%

print(
    f"Original metric: {original_metric:.10f}, new metric: {new_metric:.10f}"
)

# %%

