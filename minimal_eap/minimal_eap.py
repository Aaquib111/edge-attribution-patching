#%% Imports
import sys
sys.path.append('..')

import torch as t
import einops

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
BATCH_SIZE = 15
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
downstream_hook_names = ("hook_q", "hook_k", "hook_v", "hook_q_input","hook_k_input", "hook_v_input", "hook_mlp_in", "blocks.11.hook_resid_post")










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
#%% Get idx
clean_cache




# %% Compute matrix holding all attribution scores
# edge_attribution_score = (upstream_corr - upstream_clean) * downstream_grad_clean
N_UPSTREAM_STAGES = len(clean_cache)
N_DOWNSTREAM_STAGES = len(clean_grad_cache) - 2*model.cfg.n_layers # qkv
SEQUENCE_LENGTH = clean_ds.shape[1]

# Get (upstream_corr - upstream_clean) as matrix
upstream_cache_clean = t.zeros((
    N_UPSTREAM_STAGES, 
    model.cfg.n_heads, 
    BATCH_SIZE,
    SEQUENCE_LENGTH,
    model.cfg.d_model
))
upstream_cache_corr = t.zeros_like(upstream_cache_clean)

upstream_idx = []
for stage_cnt, name in enumerate(clean_cache.keys()):
    if name.endswith("result"):
        act_clean = einops.rearrange(clean_cache[name], "b s nh dm -> nh b s dm")
        act_corr = einops.rearrange(corrupted_cache[name], "b s nh dm -> nh b s dm")
        upstream_cache_clean[stage_cnt] = act_clean
        upstream_cache_corr[stage_cnt] = act_corr
    else:
        upstream_cache_clean[stage_cnt] = clean_cache[name]
        upstream_cache_corr[stage_cnt] = corrupted_cache[name]
    upstream_idx.append(name)
upstream_diff = upstream_cache_corr - upstream_cache_clean

#%% Get downstream_grad as matrix
downstream_grad_cache_clean = t.zeros((
    N_DOWNSTREAM_STAGES,
    model.cfg.n_heads * 3, # q, k, v separate
    BATCH_SIZE,
    SEQUENCE_LENGTH,
    model.cfg.d_head
))

downstream_idx = []
stage_cnt = 0
for name in clean_grad_cache:
    if name.endswith("hook_q_input"):
        q_name = name
        k_name = name[:-7] + "k_input"
        v_name = name[:-7] + "v_input"
        q_act = einops.rearrange(clean_grad_cache[q_name], "b s nh dm -> nh b s dm")
        k_act = einops.rearrange(clean_grad_cache[k_name], "b s nh dm -> nh b s dm")
        v_act = einops.rearrange(clean_grad_cache[v_name], "b s nh dm -> nh b s dm")
        qkv_stack = t.vstack((q_act, k_act, v_act))
        print(f"{qkv_stack.shape=}")
        downstream_grad_cache_clean[stage_cnt] = qkv_stack
    elif name.endswith(("hook_k_input", "hook_v_input")):
        continue
    else:
        downstream_grad_cache_clean[stage_cnt] = clean_grad_cache[name]

    downstream_idx.append(name)
    stage_cnt += 1
    if stage_cnt == 5:
        break



#  Make explicit only upstream -> downstream (not downstream -> upstream is important)
# %%
clean_grad_cache.keys()
# %%
clean_grad_cache['blocks.0.attn.hook_q'].shape
# %%
