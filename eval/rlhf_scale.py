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


runs = list(wandb.Api().runs(
    path=f"costa-huang/tldr_summarize",
    filters={
        "$and": [
            {f"config.exp_name.value": "ppo_left_padding"},
            {"tags": {"$in": [args.wandb_tag]}},
        ]
    }
))
runs2 = list(wandb.Api().runs(
    path=f"costa-huang/tldr_summarize",
    filters={
        "$and": [
            {f"config.exp_name.value": "dpo"},
            {"tags": {"$in": [args.wandb_tag]}},
        ]
    }
))
runs3 = list(wandb.Api().runs(
    path=f"costa-huang/tldr_summarize",
    filters={
        "$and": [
            {f"config.exp_name.value": "sft"},
            {"tags": {"$in": [args.wandb_tag]}},
        ]
    }
))
runs4 = list(wandb.Api().runs(
    path=f"costa-huang/tldr_summarize",
    filters={
        "$and": [
            {f"config.exp_name.value": "ppo_left_padding_128"},
            {"tags": {"$in": [args.wandb_tag]}},
        ]
    }
))
runs5 = list(wandb.Api().runs(
    path=f"costa-huang/tldr_summarize",
    filters={
        "$and": [
            {f"config.exp_name.value": "ppo_left_padding_new"},
            {"tags": {"$in": [args.wandb_tag]}},
        ]
    }
))
runs = runs + runs5
# runs = runs4

pref_rates = {
    "ppo_left_padding": defaultdict(list),
    "ppo_left_padding_new": defaultdict(list),
    # "ppo_left_padding_128": defaultdict(list),
    # "dpo": defaultdict(list),
}
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

    # evaluate the model
    evaluated_filename = f'{folder}/query_responses_{args.model}.csv'
    if not os.path.exists(evaluated_filename):
        run_command(f"python oai.py --csv {folder}/query_responses.csv --num_trails 1 --n -1 --output_path {evaluated_filename} ")
    else:
        print(f"skipped python oai.py --csv {folder}/query_responses.csv --num_trails 1 --n -1 --output_path {evaluated_filename} ")
        


    # read the results
    df = pd.read_csv(evaluated_filename)
    df = df.dropna()
    print(f"ours {len(df)=}")
    response_tokens = tokenizer.batch_encode_plus(df["postprocessed_response"].tolist())
    response_tokens_lens = [len(item) for item in response_tokens["input_ids"]]
    pref_rate = df["preferred"].value_counts()["ours"] / (len(df))
    base_model_idx = run.config["base_model"].find("EleutherAI")
    base_model = "/".join(run.config["base_model"][base_model_idx:].split("/")[:2])
    pref_rates[run.config["exp_name"]][base_model].append([pref_rate, np.mean(response_tokens_lens)])
    print(f"{run.config['exp_name']} {base_model} {run.config['seed']} {pref_rate=}")

model_sizes = {
    "EleutherAI/pythia-1b-deduped": 1e9,
    "EleutherAI/pythia-2.8b-deduped": 2.8e9,
    "EleutherAI/pythia-6.9b-deduped": 6.9e9,
}
exp_name_plots = []
# colors = list(reversed(["red", "#FF5733", "#FFD700"]))
colors = sns.color_palette("colorblind", n_colors=len(model_sizes))
for exp_name, color in zip(pref_rates, colors):
    xs = []
    ys = []
    for model_size in model_sizes:
        xs.extend([model_sizes[model_size]] * len(pref_rates[exp_name][model_size]))
        ys.extend([item[0] for item in pref_rates[exp_name][model_size]])
    if exp_name == "ppo_left_padding":
        label = "Our PPO Reproduction"
    elif exp_name == "dpo":
        label = "DPO"
    elif exp_name == "ppo_left_padding_128":
        label = "PPO (128 tokens)"
    plt.scatter(
        xs, 
        ys, 
        label=label,
        color=color,
        alpha=0.5,
    )

# OAI 1.3B baseline generated directly from https://github.com/openai/summarize-from-feedback/blob/master/exps/sample.py
df = pd.read_csv("sampled_data/openai_ppo_original_1.3b/validation.csv_judged.csv")
plt.scatter(
    [1.3e9], 
    [df["preferred"].value_counts()["ours"] / (len(df))], 
    label="OpenAI 1.3b PPO checkpoint\n(Stiennon et al., 2020)",
    color="red",
    alpha=0.5,
    marker="*", # Change marker to a star
    s=100 # Adjust the size of the star
)
print(f"oai {len(df)=}")

# Adding the human baseline and ensemble of humans
plt.axhline(y=0.5, color='black', linestyle='-.', label='reference summary')

# Setting the title and labels
plt.title("RLHF scaling")
plt.xlabel("Model size")
plt.ylabel(f"Fraction preferred to reference summaries\n(according to {args.model})")
plt.xscale("log")  # Setting the x-axis to a logarithmic scale
plt.xlim(5e8, 1e10)
plt.legend()

# Display the plot
plt.grid(True, which="both", ls="--", c='0.7')
plt.tight_layout()
plt.savefig("images/rlhf_scale_plot.png")
plt.savefig("images/rlhf_scale_plot.pdf")
plt.clf()
plt.close()

# plot x-axis as the token length and y-axis as the preference rate
exp_name_plots = []
palettes = ["Reds", "Blues", "Greens"]
for exp_name, palette in zip(pref_rates, palettes):
    colors = sns.color_palette(palette, n_colors=len(model_sizes))
    for model_size, color in zip(model_sizes, colors):
        xs = [item[1] for item in pref_rates[exp_name][model_size]]
        ys = [item[0] for item in pref_rates[exp_name][model_size]]
        print(model_size, color, xs, ys)
        plt.scatter(
            xs, 
            ys, 
            label=f"{exp_name}/{model_size}".replace('EleutherAI/pythia-', '').replace('-deduped', ''),
            color=color,
        )
plt.xlabel("Token length")
plt.ylabel(f"Fraction preferred to reference summaries\n(according to {args.model})")
plt.legend()
plt.grid(True, which="both", ls="--", c='0.7')
plt.tight_layout()
plt.savefig("images/rlhf_scale_plot_token_length.png")
plt.savefig("images/rlhf_scale_plot_token_length.pdf")
