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
from torch import Tensor

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
exp = TLACDCExperiment(
    model=tl_model,
    threshold=-100.0, # So nothing gets pruned
    run_name="ap_approx_run",
    ds=test_data,
    ref_ds=test_patch_data,
    metric=test_metrics['docstring_metric'],
    zero_ablation=False,
    online_cache_cpu=False,
    corrupted_cache_cpu=False,
    verbose=True,
    using_wandb=False,
)

#%%

# Sanity check saving these norms to ensure ACDCPP isn't doing anything weird
cached_norms = {
    "corrupted": {}, 
    "online": {},
}
for cache_name, cache in zip(
    ["corrupted", "online"],
    [exp.global_cache.corrupted_cache, exp.global_cache.online_cache],
    strict=False,
):
    for node in cache:
        cached_norms[cache_name][node] = cache[node].norm().item()
torch.save(cached_norms, os.path.expanduser(f'~/acdcpp/TLACDCExperiment_norms.pt'))

#%%

# With respect to one edge, we want to see the Attribution patching effects