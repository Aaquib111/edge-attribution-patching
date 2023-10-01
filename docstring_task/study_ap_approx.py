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
    verbose=False,
    attr_absolute_val=True,
    save_graphs_after=0,
    pruning_mode='edge',
    no_pruned_nodes_attr=1,
    using_wandb=False,
)
acdc_exp = acdcpp_exp.setup_exp(- 100.0)

#%%

exp = acdc_exp

#%%

ground_truth = torch.load(os.path.expanduser('~/acdcpp/TLACDCExperiment_norms.pt'))

#%%

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

torch.save(cached_norms, os.path.expanduser(f'~/acdcpp/ACDCPP_norms.pt'))

#%%

# Assert all these norms are the same
for node in cached_norms['corrupted']:
    torch.testing.assert_allclose(cached_norms['corrupted'][node], ground_truth['corrupted'][node], atol=1e-5, rtol=1e-5)

for node in cached_norms['online']:
    torch.testing.assert_allclose(cached_norms['online'][node], ground_truth['online'][node], atol=1e-5, rtol=1e-5)

#%%

# With respect to one edge, we want to see the Attribution patching effects