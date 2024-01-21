from collections import defaultdict
import json
import os
import shlex
import subprocess
import sys
import time
from matplotlib import pyplot as plt
import pandas as pd
import wandb
import shutil
import json
import csv

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
            {"tags": {"$in": ["refactor111"]}},
        ]
    }
))
runs2 = list(wandb.Api().runs(
    path=f"costa-huang/tldr_summarize",
    filters={
        "$and": [
            {f"config.exp_name.value": "dpo"},
            {"tags": {"$in": ["refactor111"]}},
        ]
    }
))
# runs3 = list(wandb.Api().runs(
#     path=f"costa-huang/tldr_summarize",
#     filters={
#         "$and": [
#             {f"config.exp_name.value": "sft"},
#             {"tags": {"$in": ["refactor111"]}},
#         ]
#     }
# ))
runs = runs # + runs2

pref_rates = {
    "ppo_left_padding": defaultdict(list),
    # "dpo": defaultdict(list),
}
for run in runs:
    if run.state != "finished":
        continue
    folder = f'sampled_data/{run.config["exp_name"]}/{run.config["base_model"]}/{run.config["seed"]}'
    os.makedirs(folder, exist_ok=True)
    print(folder)

    # if `query_responses` doesn't exist, download it
    if not os.path.exists(f'{folder}/query_responses.csv'):
        for file in run.files():
            if "table/eval/query_responses" in file.name:
                print(f"downloading {file.name}")
                file.download()
                break
            elif "samples/query_responses" in file.name:
                print(f"downloading {file.name}")
                file.download()
                break
        shutil.copyfile(file.name, f'{folder}/query_responses.json')
        convert_json_to_csv(f'{folder}/query_responses.json', f'{folder}/query_responses.csv')
        shutil.rmtree('media')

    # evaluate the model
    if not os.path.exists(f'{folder}/query_responses.csv_judged.csv'):
        run_command(f"python oai.py --csv {folder}/query_responses.csv --num_trails 1 --n -1")
        time.sleep(10) # rate limit stuff

    # read the results
    df = pd.read_csv(f'{folder}/query_responses.csv_judged.csv')
    df = df.dropna()
    pref_rate = df["preferred"].value_counts()["ours"] / (len(df))
    base_model_idx = run.config["base_model"].find("EleutherAI")
    base_model = "/".join(run.config["base_model"][base_model_idx:].split("/")[:2])
    pref_rates[run.config["exp_name"]][base_model].append(pref_rate)

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
        ys.extend(pref_rates[exp_name][model_size])
    plt.scatter(
        xs, 
        ys, 
        label=exp_name,
        color=color,
        alpha=0.5,
    )

# OAI 1.3B baseline generated directly from https://github.com/openai/summarize-from-feedback/blob/master/exps/sample.py
df = pd.read_csv("sampled_data/openai_ppo_original_1.3b/validation.csv_judged.csv")
plt.scatter(
    [1.3e9], 
    [df["preferred"].value_counts()["ours"] / (len(df))], 
    label="OpenAI 1.3b checkpoint (Stiennon et al., 2020)",
    color="red",
    alpha=0.5,
    marker="*", # Change marker to a star
    s=100 # Adjust the size of the star
)

# Adding the human baseline and ensemble of humans
plt.axhline(y=0.5, color='black', linestyle='-.', label='reference summary')

# Setting the title and labels
plt.title("RLHF scaling")
plt.xlabel("Model size")
plt.ylabel("Fraction preferred to ref (according to GPT-4)")
plt.xscale("log")  # Setting the x-axis to a logarithmic scale
plt.xlim(1e8, 1e10) 
plt.legend()

# Display the plot
plt.grid(True, which="both", ls="--", c='0.7')
plt.tight_layout()
plt.savefig("rlhf_scale_plot.png")
# plt.clf()