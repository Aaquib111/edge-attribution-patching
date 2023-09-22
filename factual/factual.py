# In[1]:

"""
Notebook to get factual recall working.

I'm using this branch

https://github.com/ArthurConmy/Automatic-Circuit-Discovery/tree/try-to-acdcpp-speedup

of ACDC!!!
"""

from IPython import get_ipython
ipython = get_ipython()
if ipython is not None:
    ipython.run_line_magic('load_ext', 'autoreload')
    ipython.run_line_magic('autoreload', '2')
import os
import sys

if "arthur" in os.environ.get("CONDA_PREFIX", "None"):
    # Arthur is using this machine
    if (acdcpp_dir := os.path.expanduser("~/acdcpp")) not in sys.path: 
        sys.path.append(acdcpp_dir)

else:
    # TODO check these are correct on Aaquib's machine...
    sys.path.append('../Automatic-Circuit-Discovery/')
    sys.path.append('..')

from ACDCPPExperiment import ACDCPPExperiment
import json
import warnings
import re
from time import time
from functools import partial
import acdc
from acdc.acdc_graphics import show, get_node_name
from acdc.TLACDCInterpNode import TLACDCInterpNode
import pygraphviz as pgv
from pathlib import Path
import plotly.express as px
from acdc.TLACDCExperiment import TLACDCExperiment
import torch
from acdc.acdc_utils import TorchIndex, EdgeType
import numpy as np
import torch as t
from torch import Tensor
import einops
import itertools
from acdc.ioi_dataset import IOIDataset, format_prompt, make_table
import gc
from transformer_lens import HookedTransformer, ActivationCache
import tqdm.notebook as tqdm
import plotly
from rich import print as rprint
from rich.table import Table
from jaxtyping import Float, Bool, Int
import cProfile
from typing import Callable, Tuple, Union, Dict, Optional, Literal
from utils.prune_utils import (
    remove_redundant_node,
    remove_node,
    find_attn_node,
    find_attn_node_qkv,
    get_3_caches,
    acdc_nodes, 
    get_nodes
)

device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')
print(f'Device: {device}')
TESTING = True

# In[2]:

warnings.warn("Loading GPT-J takes some time. 5.75 minutes on a runpod A100. And why? On colab this takes 3 minutes, including the downloading part!")
def load_model():
    model = HookedTransformer.from_pretrained_no_processing( # Maybe this can speedup things more?
        "gpt2",
        # "gpt2-xl",
        # "gpt-j-6b", # Can smaller models be used so there is less waiting?
        center_writing_weights=False,
        center_unembed=False,
        fold_ln=False, # This is used as doing this processing is really slow
        # device=device, # CPU here makes things slower
    )
    return model

model = load_model()

model.set_use_hook_mlp_in(True)
model.set_use_split_qkv_input(True)
model.set_use_attn_result(True)

# In[4]:

# Munge the data to find some cities that we can track...

with open(os.path.expanduser("~/Automatic-Circuit-Discovery/acdc/factual_recall/city_data.json")) as f:
    raw_data = json.load(f) # Adjust fpath to yours...

prompt_templates = raw_data["prompt_templates"] + raw_data["prompt_templates_zs"]

if model.cfg.model_name == "gpt2-xl" or TESTING:
    warnings.warn("Just using one prompt")
    prompt_templates = ["In {} the largest city is"]

filtered_data = []

for sample in raw_data["samples"]:
    completion = model.to_tokens(" " + sample["object"], prepend_bos=False)
    subject = model.to_tokens(" " + sample["subject"], prepend_bos=False)
    if [completion.shape[-1], subject.shape[-1]] != [1, 1]:
        print(sample, "bad")
        continue
    else:
        print("Good")
        filtered_data.append(sample)

filtered_data = list(reversed(filtered_data))

losses = []
stds = []

for sample in filtered_data: # This helps as the model is confused by China->Shanghai!
    prompts = [
        template.format(sample["subject"]) for template in prompt_templates
    ]
    batched_tokens = model.to_tokens(prompts)
    completion = model.to_tokens(" " + sample["object"], prepend_bos=False).item()
    end_pos = [model.to_tokens(prompt).shape[-1]-1 for prompt in prompts]
    assert batched_tokens.shape[1] == max(end_pos) + 1
    logits = model(batched_tokens)[torch.arange(batched_tokens.shape[0]), end_pos]
    log_probs = t.log_softmax(logits, dim=-1)
    loss = - log_probs[torch.arange(batched_tokens.shape[0]), completion]
    losses.append(loss.mean().item())
    stds.append(loss.std().item())

    print(
        sample,
        "has loss",
        round(losses[-1], 4), 
        "+-",
        round(stds[-1], 4),
    )
    print(loss.tolist())

# Q: What are the losses here?
fig = px.bar(
    x=[sample["subject"] for sample in filtered_data],
    y=losses,
    # error_y=dict(
    #     type="data",
    #     array=stds,
    # ),
).update_layout(
    title="Average losses for factual recall",
)
if ipython is not None:
    fig.show()

# A: they are quite small
# Make the data

gc.collect()
t.cuda.empty_cache()

BATCH_SIZE = 5 # Make this small so no OOM...
CORR_MODE: Literal["here", "other_city"] = "here" # replace $city_name with $other_city_name or $here

assert len(filtered_data) >= BATCH_SIZE
all_subjects = [sample["subject"] for sample in filtered_data]

torch.manual_seed(0)
prompt_template_indices = torch.randint(len(prompt_templates), (BATCH_SIZE,))

clean_sentences = [prompt_templates[prompt_idx].format(all_subjects[subject_idx]) for subject_idx, prompt_idx in enumerate(prompt_template_indices)]
clean_toks = model.to_tokens([sentence for sentence in clean_sentences])
clean_end_positions = [model.to_tokens(sentence).shape[-1]-1 for sentence in clean_sentences]
clean_completions = [model.to_tokens(" " + filtered_data[i]["object"], prepend_bos=False).item() for i in range(BATCH_SIZE)]
clean_completions = t.tensor(clean_completions, device=device)

if CORR_MODE == "here":
    different_subjects = ["here" for _ in range(BATCH_SIZE)]

elif CORR_MODE == "other_city":
    different_subjects = list(set(all_subjects) - set(all_subjects[:BATCH_SIZE]))

    # This city data was too small... sigh
    different_subjects = different_subjects + different_subjects

assert len(different_subjects) >= BATCH_SIZE
corr_subjects = different_subjects[:BATCH_SIZE]

corr_sentences = [sentence.replace(all_subjects[i], corr_subjects[i]) for i, sentence in enumerate(clean_sentences)]

if model.cfg.model_name == "gpt2-xl" or TESTING:
    warnings.warn("Also corrupting corr_sentences more too")
    corr_sentences = [s.replace("city", "thing") for s in corr_sentences]

corr_toks = model.to_tokens(corr_sentences)

# Check that indeed losses are low
logits = model(clean_toks).cpu()[torch.arange(clean_toks.shape[0]), clean_end_positions]
logprobs = t.log_softmax(logits.cpu(), dim=-1)
loss = - logprobs.cpu()[torch.arange(clean_toks.shape[0]), clean_completions.cpu()]

#%%

print(loss, "are the losses") # Most look reasonable. But 4???
gc.collect()
t.cuda.empty_cache()

# In[5]:

def ave_loss(
    logits: Float[Tensor, 'batch seq d_vocab'],
    end_positions: Int[Tensor, 'batch'],
    correct_tokens: Int[Tensor, 'batch'],
):
    '''
    Return average neglogprobs of correct tokens
    '''

    end_logits = logits[range(logits.size(0)), end_positions]
    logprobs = t.log_softmax(end_logits, dim=-1)
    loss = - logprobs[range(logits.size(0)), correct_tokens]
    return loss.mean()

factual_recall_metric = partial(ave_loss, end_positions=clean_end_positions, correct_tokens=clean_completions)

with t.no_grad():
    clean_logits = model(clean_toks)
    corrupt_logits = model(corr_toks)

# Get clean and corrupt logit differences
with t.no_grad():
    clean_metric = factual_recall_metric(clean_logits)
    corrupt_metric = factual_recall_metric(corrupt_logits)

# In[10]:

run_name = "factual_recall_thresh_run"
pruned_nodes_per_thresh = {}
num_forward_passes_per_thresh = {}
heads_per_thresh = {}
os.makedirs(f'ims/{run_name}', exist_ok=True)
threshold = 0.2
start_thresh_time = time()

# Set up experiment
# For GPT-J this takes >3 minutes if caches are on CPU. 30 seconds if not.
# GPT-2 XL with positional splitting seems to take fairly long

acdcpp_exp = ACDCPPExperiment(
    model,
    clean_toks,
    corr_toks,
    factual_recall_metric,
    factual_recall_metric,
    [threshold], # THRESHOLDS[:1],
    run_name="my_factual_recall",
    verbose=False,
    attr_absolute_val=True,
    save_graphs_after=0.07,
    pruning_mode = "edge",
    no_pruned_nodes_attr=1, 
)
pruned_heads, num_passes, pruned_attrs = acdcpp_exp.run()

# %%
