#%% Imports
import torch as t

from transformer_lens import HookedTransformer
from acdc.greaterthan.utils import get_all_greaterthan_things


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
N = 25
things = get_all_greaterthan_things(
    num_examples=N, metric_name="greaterthan", device=device
)
greaterthan_metric = things.validation_metric
clean_ds = things.validation_data # clean data x_i
corr_ds = things.validation_patch_data # corrupted data x_i'

print("\nClean dataset samples")
for i in range(5):
    print(model.tokenizer.decode(clean_ds[i]))

print("\nReference dataset samples")
for i in range(5):
    print(model.tokenizer.decode(corr_ds[i]))

#%% Run the model on a dataset sample to verify the setup worked
next_token_logits = model(clean_ds[3])[-1, -1]
next_token_str = model.tokenizer.decode(next_token_logits.argmax())
print(f"prompt: {model.tokenizer.decode(clean_ds[3])}")
print(f"next token: {next_token_str}")

# %% Define Hooks for upstream and downstream nodes
# Upstream nodes in {Embedding, Attn_heads, MLPs}
# Downstream nodes in {Attn_heads, MLPs, resid_final}

# %% Get the required caches for calculating EAP scores
# (2 forward passes on clean and corr ds, backward pass on clean ds)

# %% Calculate a giant matrix holding all 



# Make explicit only upstream -> downstream (not downstream -> upstream is important)