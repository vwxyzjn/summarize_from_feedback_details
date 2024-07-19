from collections import defaultdict
from dataclasses import dataclass
import json
import os
import shlex
import subprocess
import sys
import time
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import wandb
import shutil
import json
import csv
import tyro
from transformers import AutoTokenizer
import seaborn as sns


@dataclass
class Args:
    model: str = "gpt-3.5-turbo-0125"
    wandb_tag: str = "refactor-chosen-rejected3"
    max_seeds: int = 10000

args = tyro.cli(Args)


keys = [
    # "ppo_left_padding",
    # "ppo_left_padding_new_kl_0.07",
    # "ppo_left_padding_new1_kl_0.07",
    # "ppo_left_padding_new1",
    # "ppo_left_padding_new",
    # "ppo_left_padding_kl_0.07",
    "ppo_left_padding_new_nowhiten_reward",
    # "ppo_left_padding_128",
    # "dpo",
    "sft",
    "reward",
]
runs = [
    list(wandb.Api().runs(
        path=f"costa-huang/tldr_summarize",
        filters={
            "$and": [
                {f"config.exp_name.value": key},
                {"tags": {"$in": [args.wandb_tag]}},
            ]
        }
    )) for key in keys
]
runs = [item for sublist in runs for item in sublist]

table = defaultdict(list)
for run in runs:
    if run.config["base_model"] == "EleutherAI/pythia-1.4b-deduped":
        continue
    table["Base Model"].append(run.config["base_model"].replace("_", "\_"))
    exp_name = run.config["exp_name"]
    if exp_name == "ppo_left_padding":
        exp_name = "ppo"
    table["Type"].append(exp_name)
    table["Seed"].append(run.config["seed"])
    hf_repo_url = run.config["hf_repo_url"].replace('_', '\_')
    table["\hflogo Model Checkpoint"].append(f"\href{{{hf_repo_url}}}{{\hflogo Link}}")
    wandb_url = run.url.replace('_', '\_')
    table["Tracked Wandb Logs"].append(f"\href{{{wandb_url}}}{{Link}}")

df = pd.DataFrame(table)
df_grouped = df.groupby(["Base Model", "Type", "Seed"]).agg(lambda x: x.iloc[0])
print(df_grouped.to_latex())