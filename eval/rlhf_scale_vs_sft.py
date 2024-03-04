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
from oai2 import LLMJudgeConfig, llm_judge
from transformers import AutoTokenizer
import matplotlib.colors as mcolors


@dataclass
class Args:
    model: str = "gpt-3.5-turbo-0125"
    wandb_tag: str = "refactor-chosen-rejected2"
    rejudge: bool = False

args = tyro.cli(Args)


def convert_json_to_csv(json_file_path, csv_file_path):
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)

    columns = data['columns']
    data_rows = data['data']

    with open(csv_file_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(columns)
        writer.writerows(data_rows)

def run_command(command: str):
    command_list = shlex.split(command)
    print(f"running {command}")
    subprocess.run(command_list, stderr=sys.stderr, stdout=sys.stdout)



sft_runs = list(wandb.Api().runs(
    path=f"costa-huang/tldr_summarize",
    filters={
        "$and": [
            {f"config.exp_name.value": "sft"},
            {"tags": {"$in": [args.wandb_tag]}},
        ]
    }
))
ppo_runs = list(wandb.Api().runs(
    path=f"costa-huang/tldr_summarize",
    filters={
        "$and": [
            {f"config.exp_name.value": "ppo_left_padding"},
            {"tags": {"$in": [args.wandb_tag]}},
        ]
    }
))
dpo_runs = list(wandb.Api().runs(
    path=f"costa-huang/tldr_summarize",
    filters={
        "$and": [
            {f"config.exp_name.value": "dpo"},
            {"tags": {"$in": [args.wandb_tag]}},
        ]
    }
))

pref_rates = {
    "ppo_left_padding": defaultdict(list),
    "dpo": defaultdict(list),
}
model_generations = {}
runs = sft_runs + ppo_runs + dpo_runs

seeds = set()
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-1b-deduped")
for run in runs:
    if run.state != "finished":
        continue
    folder = f'sampled_data/{args.wandb_tag}/{run.config["exp_name"]}/{run.config["base_model"]}/{run.config["seed"]}'
    os.makedirs(folder, exist_ok=True)
    print(folder)

    # if `query_responses` doesn't exist, download it
    if not os.path.exists(f'{folder}/query_responses.csv'):
        for file in run.files():
            if "table/eval/query_responses" in file.name:
                target_file = file
                break
            elif "table/eval/validation_query_responses" in file.name:
                target_file = file
                break
            elif "samples/query_responses" in file.name:
                target_file = file
                break
        print(f"downloading {target_file.name} from {run.url}")
        target_file.download()
        shutil.copyfile(target_file.name, f'{folder}/query_responses.json')
        convert_json_to_csv(f'{folder}/query_responses.json', f'{folder}/query_responses.csv')
        shutil.rmtree('media')
    seeds.add(run.config["seed"])
    model_generations[f'{run.config["exp_name"]}/{run.config["base_model"]}/{run.config["seed"]}'] = f'{folder}/query_responses.csv'

# compare dpo and ppo runs against sft
for run in ppo_runs + dpo_runs:
    if run.state != "finished":
        continue
    folder = f'sampled_data/{args.wandb_tag}/{run.config["exp_name"]}/{run.config["base_model"]}/{run.config["seed"]}'
    df = pd.read_csv(f'{folder}/query_responses.csv').drop_duplicates(subset=['query'])
    sft_df = pd.read_csv(model_generations[f'sft/{run.config["base_model"]}/{run.config["seed"]}']).drop_duplicates(subset=['query'])
    df["query"] = df["query"].map(lambda x: x.replace("[PAD]", ""))
    sft_df["query"] = sft_df["query"].map(lambda x: x.replace("[PAD]", ""))
    
    # align the queries indices
    sft_query2idx = {query: idx for idx, query in enumerate(sft_df["query"])}
    sft_df_idxes = [sft_query2idx[query] for query in df["query"] if query in sft_query2idx]
    sft_df = sft_df.iloc[sft_df_idxes].reset_index(drop=True)
    df.index = sft_df.index
    assert all(df["query"] == sft_df["query"])

    evaluated_filename = f'{folder}/query_responses_{args.model}_vs_sft11.csv'
    if not os.path.exists(evaluated_filename) or args.rejudge:
        combined_df = pd.DataFrame({
            "prompt": df["query"],
            "response0": df["postprocessed_response"],
            "response1": sft_df["response"],
        })
        judge_df = llm_judge(ljc=LLMJudgeConfig(n=-1), df=combined_df)
        judge_df.to_csv(evaluated_filename)
    
    judge_df = pd.read_csv(evaluated_filename)
    response_tokens = tokenizer.batch_encode_plus(df["postprocessed_response"].tolist())
    response_tokens_lens = [len(item) for item in response_tokens["input_ids"]]
    pref_rate = judge_df["preferred"].value_counts()["response0"] / (judge_df["preferred"].value_counts().sum())
    pref_rates[run.config["exp_name"]][run.config["base_model"]].append([pref_rate, np.mean(response_tokens_lens)])
    

model_sizes = {
    "EleutherAI/pythia-1b-deduped": 1e9,
    "EleutherAI/pythia-2.8b-deduped": 2.8e9,
    "EleutherAI/pythia-6.9b-deduped": 6.9e9,
}
exp_name_plots = []
colors = list(reversed(["#add8e6", "#FF5733"]))
for exp_name, color in zip(pref_rates, colors):
    xs = []
    ys = []
    for model_size in model_sizes:
        xs.extend([model_sizes[model_size]] * len(pref_rates[exp_name][model_size]))
        ys.extend([item[0] for item in pref_rates[exp_name][model_size]])
    plt.scatter(
        xs, 
        ys, 
        label=exp_name,
        color=color,
        alpha=0.5,
    )

# # OAI 1.3B baseline generated directly from https://github.com/openai/summarize-from-feedback/blob/master/exps/sample.py
# df = pd.read_csv("sampled_data/openai_ppo_original_1.3b/validation.csv_judged.csv")
# plt.scatter(
#     [1.3e9], 
#     [df["preferred"].value_counts()["ours"] / (len(df))], 
#     label="OpenAI 1.3b checkpoint (Stiennon et al., 2020)",
#     color="red",
#     alpha=0.5,
#     marker="*", # Change marker to a star
#     s=100 # Adjust the size of the star
# )

# Adding the human baseline and ensemble of humans
plt.axhline(y=0.5, color='black', linestyle='-.', label='reference summary')

# Setting the title and labels
plt.title("RLHF scaling")
plt.xlabel("Model size")
plt.ylabel(f"Fraction preferred to SFT summaries\n(according to {args.model})")
plt.xscale("log")  # Setting the x-axis to a logarithmic scale
plt.xlim(1e8, 1e10) 
plt.legend()

# Display the plot
plt.grid(True, which="both", ls="--", c='0.7')
plt.tight_layout()
plt.savefig("rlhf_scale_plot_sft.png")
plt.clf()
plt.close()

# plot x-axis as the token length and y-axis as the preference rate
exp_name_plots = []
import seaborn as sns
palettes = ["flare", "crest", ]
for exp_name, palette in zip(pref_rates, palettes):
    colors = sns.color_palette(palette, n_colors=len(model_sizes))
    for model_size, color in zip(model_sizes, colors):
        xs = [item[1] for item in pref_rates[exp_name][model_size]]
        ys = [item[0] for item in pref_rates[exp_name][model_size]]
        print(model_size, color, xs, ys)
        plt.scatter(
            xs, 
            ys, 
            label=f"{exp_name}/{model_size}",
            color=color,
        )
plt.xlabel("Token length")
plt.ylabel(f"Fraction preferred to SFT summaries\n(according to {args.model})")
plt.legend()
plt.grid(True, which="both", ls="--", c='0.7')
plt.tight_layout()
plt.savefig("rlhf_scale_plot_sft_token_length.png")
