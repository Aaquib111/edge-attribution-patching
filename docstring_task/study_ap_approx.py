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
import matplotlib.pyplot as plt
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
    attr_absolute_val=False,
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

sorted_ap_attr = sorted(ap_attr.items(), key=lambda x: abs(x[1]), reverse=True)

# %%

# From here likely we're going to wrap this in a function
# TODO to reset the online cache, probably we need resetup a TLACDCExperiment???

(sender_component, receiver_component), ap_val = sorted_ap_attr[2]
print(
    f"Sender component: {sender_component}, receiver component: {receiver_component}, ap_val: {ap_val}"
)
assert "addition" in str(acdc_exp.corr.graph[receiver_component.hook_point_name][receiver_component.index].incoming_edge_type).lower()

# %%

sender_node_name, sender_node_index = sender_component.hook_point_name, sender_component.index

# %%

receiver_node_name, receiver_node_index = receiver_component.hook_point_name, receiver_component.index

# %%

original_corrupted_cache_value_cpu = acdc_exp.global_cache.corrupted_cache[sender_node_name][sender_node_index.as_index].cpu()

#%%

original_online_cache_value_cpu = acdc_exp.global_cache.online_cache[sender_node_name][sender_node_index.as_index].cpu()

# %%

original_metric = acdc_exp.cur_metric
original_edges = acdc_exp.count_no_edges()

# %%

acdc_exp.add_receiver_hook(acdc_exp.corr.graph[receiver_node_name][receiver_node_index], override=True, prepend=True)

#%%

edge=acdc_exp.corr.edges[receiver_node_name][receiver_node_index][sender_node_name][sender_node_index]
edge.mask = 1.0

# %%

acdc_exp.update_cur_metric()
new_metric = acdc_exp.cur_metric

# %%

print(
    f"Original metric: {original_metric:.10f}, new metric: {new_metric:.10f}"
)

#%%

interpolated_metrics = []

#%%

for interpolation in torch.linspace(0, 1, 101):
    # acdc_exp.global_cache.corrupted_cache[sender_node_name][sender_node_index.as_index] = (interpolation * original_corrupted_cache_value_cpu + (-interpolation+1.0) * original_online_cache_value_cpu).to(device)

    edge.mask = interpolation
    acdc_exp.update_cur_metric()
    intermediate_metric = acdc_exp.cur_metric
    interpolated_metrics.append(intermediate_metric)

# %%

# Plot the interpolated metrics
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
plt.show()

# %%

