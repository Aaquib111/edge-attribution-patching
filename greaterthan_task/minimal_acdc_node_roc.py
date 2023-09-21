#%%

from IPython import get_ipython
ipython = get_ipython()
if ipython is not None:
    ipython.magic("%load_ext autoreload")
    ipython.magic("%autoreload 2")
import os
from pathlib import Path
import json
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import pandas as pd
import re
import numpy as np

#%%

# Set your root directory here
ROOT_DIR = Path("/Users/canrager/acdcpp/greaterthan_task")
assert ROOT_DIR.exists(), f"I don't think your ROOT_DIR is correct (ROOT_DIR = {ROOT_DIR})"

# %%

TASK = "greaterthan"
METRIC = "greaterthan"
FNAME = f"results/greaterthan_absval_pruned_heads.json"
FPATH = ROOT_DIR / FNAME
assert FPATH.exists(), f"I don't think your FNAME is correct (FPATH = {FPATH})"

# %%

with open(FPATH, 'r') as f:
    pruned_heads = json.load(f)
with open(ROOT_DIR /'results/greaterthan_absval_num_passes.json', 'r') as f:
    num_passes = json.load(f)

# %%

cleaned_heads = {}

for thresh in pruned_heads.keys():
    cleaned_heads[thresh] = {}
    cleaned_heads[thresh]['acdcpp'] = set()
    cleaned_heads[thresh]['acdc'] = set()

    for i in range(2):
        for head in pruned_heads[thresh][i]:
            attn_head_pttn = re.compile('^<a([0-9]+)\.([0-9]+)>$')
            matched = attn_head_pttn.match(head)
            if matched:
                head_str = f'{matched.group(1)}.{matched.group(2)}'
                if i == 0:
                    cleaned_heads[thresh]['acdcpp'].add(head_str)
                else:
                    cleaned_heads[thresh]['acdc'].add(head_str)
    
#%%
true_baseline_heads = set(["5.1", "5.5", "6.1", "6.9", "7.10", "8.8", "8.11", "9.1"])
print(len(true_baseline_heads))

all_heads = set()

for layer in range(12):
    for head in range(12):
        all_heads.add(f'{layer}.{head}')


# %%
data = {
    'Threshold': [0],
    'ACDCpp TPR': [1],
    'ACDCpp TNR': [0],
    'ACDCpp FPR': [1],
    'ACDCpp FNR': [0],
    'TPR': [1],
    'TNR': [0],
    'FPR': [1],
    'FNR': [0],
    'Num Passes': [np.inf],
}

for thresh in cleaned_heads.keys():
    data['Threshold'].append(round(float(thresh), 3)) # Correct rounding error
    # Variables prefixed with pp_ are after ADCDCpp only
    pp_heads = cleaned_heads[thresh]['acdcpp']
    heads = cleaned_heads[thresh]['acdc']
    
    pp_tp = len(pp_heads.intersection(true_baseline_heads))
    pp_tn = len((all_heads - true_baseline_heads).intersection(all_heads - pp_heads))
    pp_fp = len(pp_heads - true_baseline_heads)
    pp_fn = len(true_baseline_heads - pp_heads)

    tp = len(heads.intersection(true_baseline_heads))
    tn = len((all_heads - true_baseline_heads).intersection(all_heads - heads))
    fp = len(heads - true_baseline_heads)
    fn = len(true_baseline_heads - heads)

    pp_tpr = pp_tp / (pp_tp + pp_fn)
    pp_tnr = pp_tn / (pp_tn + pp_fp)
    pp_fpr = 1 - pp_tnr
    pp_fnr = 1 - pp_tpr

    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    fpr = 1 - tnr
    fnr = 1 - tpr

    data['ACDCpp TPR'].append(pp_tpr)
    data['ACDCpp TNR'].append(pp_tnr)
    data['ACDCpp FPR'].append(pp_fpr)
    data['ACDCpp FNR'].append(pp_fnr)

    data['TPR'].append(tpr)
    data['TNR'].append(tnr)
    data['FPR'].append(fpr)
    data['FNR'].append(fnr)

    data['Num Passes'].append(num_passes[thresh])
df = pd.DataFrame(data)
# Add thresh inf to end of df
row = [np.inf, 0, 1, 0, 1, 0, 1, 0, 1, 0]
df.loc[len(df)] = row

# %%

# We would just plot these, but sometimes points are not on the Pareto frontier

def pareto_optimal_sublist(xs, ys):
    retx, rety = [], []
    for x, y in zip(xs, ys):
        for x1, y1 in zip(xs, ys):
            if x1 > x and y1 < y:
                break
        else:
            retx.append(x)
            rety.append(y)
    indices = sorted(range(len(retx)), key=lambda i: retx[i])
    return [retx[i] for i in indices], [rety[i] for i in indices]

# %%

pareto_node_tpr, pareto_node_fpr = pareto_optimal_sublist(data['TPR'], data['FPR'])

# %%

# Thanks GPT-4 for this code

# Create the plot
plt.figure()

# Plot the ROC curve
plt.step(pareto_node_fpr, pareto_node_tpr, where='post')

# Add titles and labels
plt.title("ROC Curve of number of Nodes recovered by ACDC")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

# Show the plot
plt.show()

# %%

# Original code from https://plotly.com/python/line-and-scatter/

# I use plotly but it should be easy to adjust to matplotlib
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=list(pareto_node_fpr),
        y=list(pareto_node_tpr),
        mode="lines",
        line=dict(shape="hv"),
        showlegend=False,
    ),
)

fig.update_layout(
    title="ROC Curve of number of Nodes recovered by ACDC",
    xaxis_title="False Positive Rate",
    yaxis_title="True Positive Rate",
)

fig.show()
# %%
