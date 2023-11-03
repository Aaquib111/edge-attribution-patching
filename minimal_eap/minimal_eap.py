#%% Imports
import sys
sys.path.append('..')

import torch as t
import einops
import plotly.express as px

from transformer_lens import HookedTransformer
from acdc.greaterthan.utils import get_all_greaterthan_things
from utils.prune_utils import get_3_caches

device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')
print(f'Device: {device}')

#%% Get transformer model running

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

#%% Get clean and corrupting datasets and task specific metric
BATCH_SIZE = 50
things = get_all_greaterthan_things(
    num_examples=BATCH_SIZE, metric_name="greaterthan", device=device
)
greaterthan_metric = things.validation_metric
clean_ds = things.validation_data # clean data x_i
corr_ds = things.validation_patch_data # corrupted data x_i'

print("\nClean dataset samples")
for stage_cnt in range(5):
    print(model.tokenizer.decode(clean_ds[stage_cnt]))

print("\nReference dataset samples")
for stage_cnt in range(5):
    print(model.tokenizer.decode(corr_ds[stage_cnt]))

#%% Run the model on a dataset sample to verify the setup worked
next_token_logits = model(clean_ds[3])[-1, -1]
next_token_str = model.tokenizer.decode(next_token_logits.argmax())
print(f"prompt: {model.tokenizer.decode(clean_ds[3])}")
print(f"next token: {next_token_str}")

# %% Define Hook filters for upstream and downstream nodes
# Upstream nodes in {Embeddings ("blocks.0.hook_resid_pre"), Attn_heads ("result"), MLPs ("mlp_out")}
# Downstream nodes in {Attn_heads ("input") , MLPs ("mlp_in"), resid_final ("blocks.11.hook_resid_post")}
# Necessary Transformerlens flags: model.set_use_hook_mlp_in(True), model.set_use_split_qkv_input(True), model.set_use_attn_result(True)
upstream_hook_names = ("blocks.0.hook_resid_pre", "hook_result", "hook_mlp_out")
downstream_hook_names = ("hook_q_input","hook_k_input", "hook_v_input", "hook_mlp_in", "blocks.11.hook_resid_post")

# %% Get the required caches for calculating EAP scores
# (2 forward passes on clean and corr ds, backward pass on clean ds)

clean_cache, corrupted_cache, clean_grad_cache = get_3_caches(
    model,
    clean_ds,
    corr_ds,
    greaterthan_metric,
    mode="edge",
    upstream_hook_names=upstream_hook_names,
    downstream_hook_names=downstream_hook_names
)

# %% Compute matrix holding all attribution scores
# edge_attribution_score = (upstream_corr - upstream_clean) * downstream_grad_clean
N_UPSTREAM_STAGES = len(clean_cache)
N_DOWNSTREAM_STAGES = len(clean_grad_cache) - 2*model.cfg.n_layers # qkv
SEQUENCE_LENGTH = clean_ds.shape[1]

N_TOTAL_UPSTREAM_NODES = 1 + model.cfg.n_layers * (model.cfg.n_heads + 1)
N_TOTAL_DOWNSTREAM_NODES = 1 + model.cfg.n_layers * (3*model.cfg.n_heads + 1)

# Get (upstream_corr - upstream_clean) as matrix
N_UPSTREAM_NODES = model.cfg.n_heads
upstream_cache_clean = t.zeros((
    N_TOTAL_UPSTREAM_NODES,
    BATCH_SIZE,
    SEQUENCE_LENGTH,
    model.cfg.d_model
))
upstream_cache_corr = t.zeros_like(upstream_cache_clean)

upstream_names = []
upstream_levels = t.zeros(N_TOTAL_UPSTREAM_NODES)
idx = 0
for stage_cnt, name in enumerate(clean_cache.keys()): # stage_cnt relevant for keeping track which upstream-downstream mairs can be connected
    if name.endswith("result"): # layer of attn heads
        act_clean = einops.rearrange(clean_cache[name], "b s nh dm -> nh b s dm")
        act_corr = einops.rearrange(corrupted_cache[name], "b s nh dm -> nh b s dm")
        upstream_cache_clean[idx:idx+model.cfg.n_heads] = act_clean
        upstream_cache_corr[idx:idx+model.cfg.n_heads] = act_corr
        upstream_levels[idx:idx+model.cfg.n_heads] = stage_cnt
        idx += model.cfg.n_heads
        for i in range(model.cfg.n_heads):
            head_name = name + str(i)
            upstream_names.append(head_name)
    else:
        upstream_cache_clean[idx] = clean_cache[name]
        upstream_cache_corr[idx] = corrupted_cache[name]
        upstream_levels[idx] = stage_cnt
        idx += 1
        upstream_names.append(name)

upstream_diff = upstream_cache_corr - upstream_cache_clean

#%% Get downstream_grad as matrix
N_DOWNSTREAM_NODES = model.cfg.n_heads * 3 # q, k, v separate
downstream_grad_cache_clean = t.zeros((
    N_TOTAL_DOWNSTREAM_NODES,
    BATCH_SIZE,
    SEQUENCE_LENGTH,
    model.cfg.d_model
))

downstream_names = []
downstream_levels = t.zeros(N_TOTAL_DOWNSTREAM_NODES)
stage_cnt = 0
idx = 0
names = reversed(list(clean_grad_cache.keys()))
for name in names:
    if name.endswith("hook_q_input"): # do all q k v hooks of that layer simultaneously, as it is the same stage
        q_name = name
        k_name = name[:-7] + "k_input"
        v_name = name[:-7] + "v_input"
        q_act = einops.rearrange(clean_grad_cache[q_name], "b s nh dm -> nh b s dm")
        k_act = einops.rearrange(clean_grad_cache[k_name], "b s nh dm -> nh b s dm")
        v_act = einops.rearrange(clean_grad_cache[v_name], "b s nh dm -> nh b s dm")

        downstream_grad_cache_clean[idx: idx+model.cfg.n_heads] = q_act
        downstream_levels[idx: idx+model.cfg.n_heads] = stage_cnt
        idx += model.cfg.n_heads
        for i in range(model.cfg.n_heads):
            head_name = q_name + str(i)
            downstream_names.append(head_name)
        
        downstream_grad_cache_clean[idx: idx+model.cfg.n_heads] = k_act
        downstream_levels[idx: idx+model.cfg.n_heads] = stage_cnt
        idx += model.cfg.n_heads
        for i in range(model.cfg.n_heads):
            head_name = k_name + str(i)
            downstream_names.append(head_name)

        downstream_grad_cache_clean[idx: idx+model.cfg.n_heads] = v_act
        downstream_levels[idx: idx+model.cfg.n_heads] = stage_cnt
        idx += model.cfg.n_heads
        for i in range(model.cfg.n_heads):
            head_name = v_name + str(i)
            downstream_names.append(head_name)

    elif name.endswith(("hook_k_input", "hook_v_input")):
        continue
    else:
        downstream_grad_cache_clean[idx] = clean_grad_cache[name]
        downstream_levels[idx] = stage_cnt
        idx += 1
        downstream_names.append(name)
    stage_cnt += 1

#%% Calculate the cartesian product of stage, node for upstream and downstream
eap_scores = einops.einsum(
    upstream_diff, 
    downstream_grad_cache_clean,
    "up_nodes batch seq d_model, down_nodes batch seq d_model -> up_nodes down_nodes"
)



#%% Make explicit only upstream -> downstream (not downstream -> upstream is important)
upstream_level_matrix = einops.repeat(upstream_levels, "up_nodes -> up_nodes down_nodes", down_nodes=N_TOTAL_DOWNSTREAM_NODES)
downstream_level_matrix = einops.repeat(downstream_levels, "down_nodes -> up_nodes down_nodes", up_nodes=N_TOTAL_UPSTREAM_NODES)
mask = upstream_level_matrix > downstream_level_matrix
eap_scores = eap_scores.masked_fill(mask, value=t.nan)

px.imshow(
    eap_scores,
    x=downstream_names,
    y=upstream_names,
    labels = dict(x="downstream node", y="upstream node", color="EAP score"),
    color_continuous_scale="RdBu",
    color_continuous_midpoint=0
)
# %%
