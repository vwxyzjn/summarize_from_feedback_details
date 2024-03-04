from collections import defaultdict
import copy
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import wandb
import pandas as pd
from collections import deque
import seaborn as sns


wandb_tag = "refactor-chosen-rejected3"
model_sizes = {
    # "EleutherAI/pythia-410m-deduped": 410e6,
    "EleutherAI/pythia-1b-deduped": 1e9,
    "EleutherAI/pythia-2.8b-deduped": 2.8e9,
    "EleutherAI/pythia-6.9b-deduped": 6.9e9,
}

def bfs_traverse_and_collect_items(nested_dict):
    queue = deque([nested_dict])  # Start with the root dictionary
    items = []

    while queue:
        current_dict = queue.popleft()

        if isinstance(current_dict, dict):
            for key, value in current_dict.items():
                items.append((key, value))
                if isinstance(value, dict):
                    queue.append(value)

    return items

def get_runs_df(runs: List[wandb.apis.public.Run]):
    summary_list = []
    for run in runs:
        d = copy.deepcopy(run.summary._json_dict)
        for k,v in bfs_traverse_and_collect_items(run.config):
            d[k] = v
        summary_list.append(d)
    return pd.DataFrame(summary_list)

def plot_reward(df, colors, labels, ykeys, ax=None):
    # print(df[["base_model", "seed", "eval/rm/validation/accuracy"]])
    for color_idx, ykey in enumerate(ykeys):
        color = colors[color_idx]
        label = labels[color_idx]
        groupby = df[["base_model", "lr", ykey]] \
            .groupby(["base_model", "lr", ]) \
            .agg(['mean', 'std'])
        print(groupby)

        # find the highest accuracy for each model
        means = []
        stds = []
        for idx, model in enumerate(model_sizes):
            item = groupby.loc[model] \
                .sort_values(by=(ykey, "mean")).iloc[-1]
            means.append(item[(ykey, "mean")])
            stds.append(item[(ykey, "std")])

        ax.errorbar(
            model_sizes.values(),
            means,
            yerr=stds,
            label=label,
            marker='o',
            color=color,
            capsize=5,
        )



colors = list(reversed(["#add8e6", "#87ceeb", "#1e90ff", "#0000ff", "#00008b"]))
df = get_runs_df(wandb.Api().runs(
    path=f"costa-huang/tldr_summarize",
    filters={
        "$and": [
            {f"config.exp_name.value": "reward"},
            {"tags": {"$in": [wandb_tag]}},
        ]
    }
))
fig, ax = plt.subplots(figsize=(4.0, 3.0))
ykeys = ["eval/rm/validation/accuracy", "eval/rm/validation_cnndm/accuracy"]
plot_reward(df, ["#00008b", "#0000ff"], ["TL;DR Set", "CNN/DM Set"], ykeys, ax=ax)



# df2 = get_runs_df(wandb.Api().runs(
#     path=f"costa-huang/tldr_summarize",
#     filters={
#         "$and": [
#             {f"config.exp_name.value": "dpo"},
#             {"tags": {"$in": [wandb_tag]}},
#         ]
#     }
# ))
# plot_reward(df2, colors[1], "DPO Reward Modeling")
# df2 = get_runs_df(wandb.Api().runs(
#     path=f"costa-huang/tldr_summarize",
#     filters={
#         "$and": [
#             {f"config.exp_name.value": "dpo_not_completion_only"},
#             {"tags": {"$in": [wandb_tag]}},
#         ]
#     }
# ))
# plot_reward(df2, colors[2], "dpo (not completion only) reward modeling")


# # Adding the human baseline and ensemble of humans
# plt.axhline(y=0.77, color='black', linestyle='-.', label='Human baseline')
# plt.axhline(y=0.83, color='black', linestyle='--', label='Ensemble of humans')

# Setting the title and labels
plt.title("RM scaling")
plt.xlabel("Model size")
plt.ylabel("Validation accuracy")
plt.xscale("log")  # Setting the x-axis to a logarithmic scale
plt.xlim(5e8, 1e10) 
plt.legend()

# Display the plot
plt.grid(True, which="both", ls="--", c='0.7')
plt.tight_layout()
fig.savefig("images/rm_scale_plot.png")
fig.savefig("images/rm_scale_plot.pdf")
plt.clf()


# df = get_runs_df(wandb.Api().runs(
#     path=f"costa-huang/tldr_summarize",
#     filters={
#         "$and": [
#             {f"config.exp_name.value": "reward"},
#             {"tags": {"$in": [wandb_tag]}},
#         ]
#     }
# ))
# fig, ax = plt.subplots(figsize=(4.0, 3.0))
# ykeys = [
#     "eval/rm/validation/accuracy/confidence/1",
#     "eval/rm/validation/accuracy/confidence/6",
#     "eval/rm/validation/accuracy/confidence/7",
#     "eval/rm/validation/accuracy/confidence/8",
#     "eval/rm/validation/accuracy/confidence/9",
# ]
# plot_reward(df, sns.color_palette("Blues", n_colors=len(ykeys)), "Reward Modeling", ykeys, ax=ax)
# # "eval/accuracy/valid1", "eval/accuracy/valid2", 
# # Setting the title and labels
# plt.title("RM scaling")
# plt.xlabel("Model size")
# plt.ylabel("Validation accuracy")
# plt.xscale("log")  # Setting the x-axis to a logarithmic scale
# plt.xlim(5e8, 1e10) 
# plt.legend()

# # Display the plot
# plt.grid(True, which="both", ls="--", c='0.7')
# plt.tight_layout()
# fig.savefig("images/rm_scale_plot_confidence.png")
# fig.savefig("images/rm_scale_plot_confidence.pdf")
# plt.clf()


df = get_runs_df(wandb.Api().runs(
    path=f"costa-huang/tldr_summarize",
    filters={
        "$and": [
            {f"config.exp_name.value": "reward"},
            {"tags": {"$in": [wandb_tag]}},
        ]
    }
))
fig, ax = plt.subplots(figsize=(4.0, 3.0))
ykeys = ["eval/rm/validation/accuracy"]
plot_reward(df, ["#00008b"], ["Reward Modeling"], ykeys, ax=ax)
df2 = get_runs_df(wandb.Api().runs(
    path=f"costa-huang/tldr_summarize",
    filters={
        "$and": [
            {f"config.exp_name.value": "dpo"},
            {"tags": {"$in": ["refactor-chosen-rejected2"]}},
        ]
    }
))
plot_reward(df2, ["#0000ff"], ["DPO Reward Modeling"], ykeys, ax=ax)

# # Adding the human baseline and ensemble of humans
# plt.axhline(y=0.77, color='black', linestyle='-.', label='Human baseline')
# plt.axhline(y=0.83, color='black', linestyle='--', label='Ensemble of humans')

# Setting the title and labels
plt.title("RM scaling")
plt.xlabel("Model size")
plt.ylabel("Validation accuracy")
plt.xscale("log")  # Setting the x-axis to a logarithmic scale
plt.xlim(5e8, 1e10) 
plt.legend()

# Display the plot
plt.grid(True, which="both", ls="--", c='0.7')
plt.tight_layout()
fig.savefig("images/rm_scale_dpo_plot.png")
fig.savefig("images/rm_scale_dpo_plot.pdf")
plt.clf()


ykeys = [item for item in  df.columns if "eval/rm" in item]
groupby_mean = df[["base_model", *ykeys]] \
    .groupby(["base_model", ]) \
    .mean()
groupby_std = df[["base_model", *ykeys]] \
    .groupby(["base_model", ]) \
    .std()

groupby_merged = groupby_mean.copy()
for ykey in ykeys:
    groupby_merged[ykey] = groupby_mean[ykey].round(3).astype(str) + " ± " + groupby_std[ykey].round(3).astype(str)
print(groupby_merged.T.sort_index())

# raise


fig, ax = plt.subplots(figsize=(4.0, 3.0))
df = get_runs_df(wandb.Api().runs(
    path=f"costa-huang/tldr_summarize",
    filters={
        "$and": [
            {f"config.exp_name.value": "sft"},
            {"tags": {"$in": [wandb_tag]}},
        ]
    }
))
print(df[["base_model", "seed", "validation/sft/rouge/rougeL"]])
ykeys = ["validation/sft/rouge/rougeL"]
for color_idx, ykey in enumerate(ykeys):
    groupby = df[["base_model", "lr", ykey]] \
        .groupby(["base_model", "lr", ]) \
        .agg(['mean', 'std'])
    print(groupby)

    # find the highest accuracy for each model
    means = []
    stds = []
    for idx, model in enumerate(model_sizes):
        item = groupby.loc[model] \
            .sort_values(by=(ykey, "mean")).iloc[-1]
        means.append(item[(ykey, "mean")])
        stds.append(item[(ykey, "std")])

    ax.errorbar(
        model_sizes.values(),
        means,
        yerr=stds,
        label="SFT model",
        marker='o',
        color=colors[color_idx],
        capsize=5,
    )

# Setting the title and labels
plt.title("ROUGE scores scaling")
plt.xlabel("Model size")
plt.ylabel("ROUGE score")
plt.xscale("log")  # Setting the x-axis to a logarithmic scale
plt.xlim(5e8, 1e10) 
plt.legend()

# Display the plot
plt.grid(True, which="both", ls="--", c='0.7')
plt.tight_layout()
plt.savefig("images/rouge_score_plot.png")
plt.savefig("images/rouge_score_plot.pdf")


