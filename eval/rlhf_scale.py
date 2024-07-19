import csv
import json
import os
import shlex
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd
import seaborn as sns
import tyro
import wandb
from matplotlib import pyplot as plt
from transformers import AutoTokenizer

"""
python rlhf_scale.py  --model gpt-4-0125-preview --max-seeds 1 --no_judge
"""


@dataclass
class Args:
    model: str = "gpt-3.5-turbo-0125"
    wandb_tag: str = "refactor-chosen-rejected3"
    max_seeds: int = 10000
    judge: bool = False


args = tyro.cli(Args)


def convert_json_to_csv(json_file_path, csv_file_path):
    with open(json_file_path) as json_file:
        data = json.load(json_file)

    columns = data["columns"]
    data_rows = data["data"]

    with open(csv_file_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(columns)
        writer.writerows(data_rows)


def run_command(command: str):
    command_list = shlex.split(command)
    print(f"running {command}")
    subprocess.run(command_list, stderr=sys.stderr, stdout=sys.stdout)


dpo_runs = list(
    wandb.Api().runs(
        path=f"costa-huang/tldr_summarize",
        filters={
            "$and": [
                {f"config.exp_name.value": "dpo"},
                {"tags": {"$in": ["refactor-chosen-rejected2"]}},
            ]
        },
    )
)


keys = {
    # "ppo_left_padding": args.wandb_tag,
    "sft": args.wandb_tag,
    # "ppo_left_padding_new_kl_0.07": args.wandb_tag,
    # "ppo_left_padding_new1_kl_0.07": args.wandb_tag,
    # "ppo_left_padding_new1": args.wandb_tag,
    # "ppo_left_padding_new": args.wandb_tag,
    # "ppo_left_padding_kl_0.07": args.wandb_tag,
    "ppo_left_padding_new_nowhiten_reward": args.wandb_tag,
    "ppo_lora": args.wandb_tag,
    # "dpo_eos": args.wandb_tag,
    # "ppo": args.wandb_tag,
    # "ppo_left_padding_nowhiten_reward": args.wandb_tag,
    # "ppo_left_padding_128": args.wandb_tag,
    # "dpo": args.wandb_tag,
    # "dpo_0.5": args.wandb_tag,
    "dpo": "refactor-chosen-rejected2",
}
exp_name_map = {
    # "ppo_left_padding": "Our PPO",
    # "ppo_left_padding_new": "Our PPO\n(no unconditional KL Penalty)",
    "ppo_left_padding_new_nowhiten_reward": "PPO",
    "ppo_lora": "PPO w/ LoRA",
    # "ppo": "PPO",
    "ppo_left_padding_new": "PPO w/ reward whitening",
    # "ppo_left_padding_new1": "Our PPO 1\n(without unconditional KL Penalty)",
    "dpo": "DPO",
    "sft": "SFT",
    # "ppo_left_padding_128": "PPO (128 tokens)"
}
runs = [
    list(
        wandb.Api().runs(
            path=f"costa-huang/tldr_summarize",
            filters={
                "$and": [
                    {f"config.exp_name.value": key},
                    {"tags": {"$in": [keys[key]]}},
                ]
            },
        )
    )
    for key in keys
]

runs = {
    key: list(
        wandb.Api().runs(
            path=f"costa-huang/tldr_summarize",
            filters={
                "$and": [
                    {f"config.exp_name.value": key},
                    {"tags": {"$in": [keys[key]]}},
                ]
            },
        )
    )
    for key in keys
    if key != "ppo"
}
# # hack to fix the column names
# if "ppo" in keys:
#     runs["ppo_left_padding_new_nowhiten_reward"].append(
#         list(
#             wandb.Api().runs(
#                 path=f"costa-huang/tldr_summarize",
#                 filters={
#                     "$and": [
#                         {f"config.exp_name.value": "ppo"},
#                         {"tags": {"$in": [keys["ppo"]]}},
#                     ]
#                 },
#             )
#         )
#     )
runs = runs.values()
runs = [item for sublist in runs for item in sublist]
pref_rates = {key: defaultdict(list) for key in keys}

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-1b-deduped")
for run in runs:
    if run.state != "finished":
        continue
    folder = f'sampled_data/{run.tags[-1]}/{run.config["exp_name"]}/{run.config["base_model"]}/{run.config["seed"]}'
    os.makedirs(folder, exist_ok=True)
    print(folder)

    # if `query_responses` doesn't exist, download it
    if not os.path.exists(f"{folder}/query_responses.csv"):
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
        shutil.copyfile(target_file.name, f"{folder}/query_responses.json")
        convert_json_to_csv(f"{folder}/query_responses.json", f"{folder}/query_responses.csv")
        if run.config["exp_name"] == "sft":  # hack to fix the column names
            df = pd.read_csv(f"{folder}/query_responses.csv")
            df["postprocessed_response"] = df["response"]
            df["reference_responses"] = df["reference"]
            del df["response"], df["reference"]
            df["query"] = df["query"].apply(lambda x: x.replace("[PAD]", ""))
            df.to_csv(f"{folder}/query_responses.csv")
        shutil.rmtree("media")

    # evaluate the model
    base_model_idx = run.config["base_model"].find("EleutherAI")
    base_model = "/".join(run.config["base_model"][base_model_idx:].split("/")[:2])
    evaluated_filename = f"{folder}/query_responses_{args.model}.csv"

    hack_exp_name = run.config["exp_name"] if run.config["exp_name"] != "ppo" else "ppo_left_padding_new_nowhiten_reward"

    if not os.path.exists(evaluated_filename) and len(pref_rates[hack_exp_name][base_model]) < args.max_seeds and args.judge:
        run_command(
            f"python oai.py --model {args.model} --csv {folder}/query_responses.csv --num_trails 1 --n -1 --output_path {evaluated_filename} "
        )
    else:
        print(
            f"skipped python oai.py --model {args.model} --csv {folder}/query_responses.csv --num_trails 1 --n -1 --output_path {evaluated_filename} "
        )

    if len(pref_rates[hack_exp_name][base_model]) >= args.max_seeds:
        continue

    # read the results
    df = pd.read_csv(evaluated_filename)
    df = df.dropna()
    print(f"ours {len(df)=}")
    response_tokens = tokenizer.batch_encode_plus(df["postprocessed_response"].tolist())
    response_tokens_lens = [len(item) for item in response_tokens["input_ids"]]
    pref_rate = df["preferred"].value_counts()["ours"] / (len(df))
    pref_rates[hack_exp_name][base_model].append([pref_rate, np.mean(response_tokens_lens)])
    print(f"{hack_exp_name} {base_model} {run.config['seed']} {pref_rate=}")

model_sizes = {
    "EleutherAI/pythia-1b-deduped": 1e9,
    "EleutherAI/pythia-1.4b-deduped": 1.4e9,
    "EleutherAI/pythia-2.8b-deduped": 2.8e9,
    "EleutherAI/pythia-6.9b-deduped": 6.9e9,
}
exp_name_plots = []
# colors = list(reversed(["red", "#FF5733", "#FFD700"]))
# colors = sns.color_palette("colorblind", n_colors=len(pref_rates))
palettes = ["Reds", "Blues", "Greens", "Purples"]
fig, ax = plt.subplots(figsize=(5.0, 4.5))
i = 0
for exp_name, palette in zip(pref_rates, palettes):
    colors = sns.color_palette(palette, n_colors=1)
    color = colors[0]
    xs = []
    ys = []
    for model_size in model_sizes:
        xs.extend([model_sizes[model_size]] * len(pref_rates[exp_name][model_size]))
        ys.extend([item[0] for item in pref_rates[exp_name][model_size]])
    label = exp_name_map.get(exp_name, exp_name)

    # # hack
    # label = "" if i == 1 else label
    # color = sns.color_palette("Reds", n_colors=1)[0] if i == 1 else color

    plt.scatter(
        xs,
        ys,
        label=label,
        color=color,
        alpha=0.5,
    )
    i += 1

# OAI 1.3B baseline generated directly from https://github.com/openai/summarize-from-feedback/blob/master/exps/sample.py
if args.model == "gpt-3.5-turbo-0125":
    df = pd.read_csv("sampled_data/openai_ppo_original_1.3b/validation.csv_judged.csv")
else:
    df = pd.read_csv("sampled_data/openai_ppo_original_1.3b/validation_gpt-4-0125-preview.csv")
plt.scatter(
    [1.3e9],
    [df["preferred"].value_counts()["ours"] / (len(df))],
    label="OpenAI 1.3b PPO checkpoint\n(Stiennon et al., 2020)",
    color="red",
    alpha=0.5,
    marker="*",  # Change marker to a star
    s=100,  # Adjust the size of the star
)
response_tokens = tokenizer.batch_encode_plus(df["postprocessed_response"].tolist())
response_tokens_lens = [len(item) for item in response_tokens["input_ids"]]
oai_average_response_tokens_lens = np.mean(response_tokens_lens)
print(f"oai {len(df)=}")

# Adding the human baseline and ensemble of humans
plt.axhline(y=0.5, color="black", linestyle="-.", label="reference summary")

# Setting the title and labels
plt.title("RLHF scaling by model size")
plt.xlabel("Model size")
plt.ylabel(f"Win rate against reference summaries\n(according to {args.model})")
plt.xscale("log")  # Setting the x-axis to a logarithmic scale
plt.xlim(5e8, 1e10)
plt.legend(loc="lower right")

# Display the plot
plt.grid(True, which="both", ls="--", c="0.7")
plt.tight_layout()
plt.savefig("images/rlhf_scale_plot.png")
plt.savefig("images/rlhf_scale_plot.pdf")
plt.clf()
plt.close()
fig, ax = plt.subplots(figsize=(5.0, 4.5))
# plot x-axis as the token length and y-axis as the preference rate
exp_name_plots = []
for exp_name, palette in zip(pref_rates, palettes):
    colors = sns.color_palette(palette, n_colors=len(model_sizes))
    for model_size, color in zip(model_sizes, colors):
        xs = [item[1] for item in pref_rates[exp_name][model_size]]
        ys = [item[0] for item in pref_rates[exp_name][model_size]]
        print(model_size, color, xs, ys)
        label = exp_name_map.get(exp_name, exp_name)
        plt.scatter(
            xs,
            ys,
            label=f"{model_size}/{label}".replace("EleutherAI/pythia-", "").replace("-deduped", ""),
            color=color,
        )
plt.scatter(
    [oai_average_response_tokens_lens],
    [df["preferred"].value_counts()["ours"] / (len(df))],
    label="OpenAI 1.3b PPO checkpoint\n(Stiennon et al., 2020)",
    color="red",
    marker="*",  # Change marker to a star
)
plt.xlabel("Token length")
plt.title("RLHF scaling by token length")
plt.ylabel(f"Win rate against reference summaries\n(according to {args.model})")
# plt.figlegend(loc="lower center", ncol=2, bbox_to_anchor=(0.5, 1.0))
plt.legend()
plt.grid(True, which="both", ls="--", c="0.7")
plt.tight_layout()
plt.savefig("images/rlhf_scale_plot_token_length.png")
plt.savefig("images/rlhf_scale_plot_token_length.pdf")
