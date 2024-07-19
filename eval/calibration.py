import csv
import json
import os
import shutil
import wandb

reward_eval_runs = list(wandb.Api().runs(
    path=f"costa-huang/tldr_summarize",
    filters={
        "$and": [
            {f"config.exp_name.value": "reward_eval"},
            {"tags": {"$in": ["refactor-chosen-rejected3"]}},
        ]
    }
))

def convert_json_to_csv(json_file_path, csv_file_path):
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)

    columns = data['columns']
    data_rows = data['data']

    with open(csv_file_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(columns)
        writer.writerows(data_rows)

for run in reward_eval_runs:
    folder = f'sampled_data/{run.tags[-1]}/{run.config["exp_name"]}/{run.config["base_model"]}/{run.config["seed"]}'
    os.makedirs(folder, exist_ok=True)
    # if `query_responses` doesn't exist, download it
    if not os.path.exists(f'{folder}/query_responses.csv'):
        for file in run.files():
            if "validation/query_responses" in file.name:
                target_file = file
                break
        print(f"downloading {target_file.name} from {run.url}")
        target_file.download()
        shutil.copyfile(target_file.name, f'{folder}/query_responses.json')
        convert_json_to_csv(f'{folder}/query_responses.json', f'{folder}/query_responses.csv')
        if run.config["exp_name"] == "sft": # hack to fix the column names
            df = pd.read_csv(f'{folder}/query_responses.csv')
            df["postprocessed_response"] = df["response"]
            df["reference_responses"] = df["reference"]
            del df["response"], df["reference"]
            df["query"] = df["query"].apply(lambda x: x.replace("[PAD]", ""))
            df.to_csv(f'{folder}/query_responses.csv')
        shutil.rmtree('media')
