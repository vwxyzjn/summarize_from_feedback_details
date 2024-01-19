from collections import defaultdict
import copy
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import wandb
import pandas as pd
from collections import deque


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

def plot_reward(df, color, label):
    # hack
    # models/EleutherAI/pythia-1b-deduped/sft_model_4441 -> EleutherAI/pythia-1b-deduped
    df["base_model"] = df["base_model"].apply(lambda x: "/".join(x.split("/")[1:3]))
    print(df[["base_model", "seed", "eval/rm/validation/accuracy"]])
    ykeys = [
        "eval/rm/validation/accuracy", 
        # "eval/accuracy/valid1", "eval/accuracy/valid2", "eval/confidence/7", "eval/confidence/8", "eval/confidence/9"
    ]
    for color_idx, ykey in enumerate(ykeys):
        groupby = df[["base_model", "lr", "num_train", ykey]] \
            .groupby(["base_model", "lr", "num_train", ]) \
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

        plt.errorbar(
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
            {"tags": {"$in": ["refactor111"]}},
        ]
    }
))
plot_reward(df, colors[0], "reward modeling")
df2 = get_runs_df(wandb.Api().runs(
    path=f"costa-huang/tldr_summarize",
    filters={
        "$and": [
            {f"config.exp_name.value": "dpo"},
            {"tags": {"$in": ["refactor111"]}},
        ]
    }
))
plot_reward(df2, colors[1], "dpo reward modeling")
df2 = get_runs_df(wandb.Api().runs(
    path=f"costa-huang/tldr_summarize",
    filters={
        "$and": [
            {f"config.exp_name.value": "dpo_not_completion_only"},
            {"tags": {"$in": ["refactor111"]}},
        ]
    }
))
plot_reward(df2, colors[2], "dpo (not completion only) reward modeling")


# Adding the human baseline and ensemble of humans
plt.axhline(y=0.77, color='black', linestyle='-.', label='Human baseline')
plt.axhline(y=0.83, color='black', linestyle='--', label='Ensemble of humans')

# Setting the title and labels
plt.title("RM scaling law")
plt.xlabel("Model size")
plt.ylabel("Validation accuracy")
plt.xscale("log")  # Setting the x-axis to a logarithmic scale
plt.xlim(1e8, 1e10) 
plt.legend()

# Display the plot
plt.grid(True, which="both", ls="--", c='0.7')
plt.tight_layout()
plt.savefig("rm_scale_plot.png")
plt.clf()


df = get_runs_df(wandb.Api().runs(
    path=f"costa-huang/tldr_summarize",
    filters={
        "$and": [
            {f"config.exp_name.value": "sft"},
            {"tags": {"$in": ["refactor111"]}},
        ]
    }
))
print(df[["base_model", "seed", "rouge/rougeL"]])
ykeys = ["rouge/rougeL"]
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

    plt.errorbar(
        model_sizes.values(),
        means,
        yerr=stds,
        label=ykey,
        marker='o',
        color=colors[color_idx],
        capsize=5,
    )

# Setting the title and labels
plt.title("ROUGE scores scaling law")
plt.xlabel("Model size")
plt.ylabel("ROUGE score")
plt.xscale("log")  # Setting the x-axis to a logarithmic scale
plt.xlim(1e8, 1e10) 
plt.legend()

# Display the plot
plt.grid(True, which="both", ls="--", c='0.7')
plt.tight_layout()
plt.savefig("rouge_score_plot.png")