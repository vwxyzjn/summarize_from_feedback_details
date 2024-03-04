from collections import defaultdict
import pandas as pd
from transformers import AutoTokenizer
import os

endings = ["gpt-3.5-turbo-0125.csv", "csv_judged.csv", "_vs_sft11.csv"]

tokenizer = AutoTokenizer.from_pretrained(
    "EleutherAI/pythia-160m",
    padding_side="right",
    trust_remote_code=True,
)
# we use the padding token manually but do not resize the token embedding of the model
tokenizer.add_special_tokens({"pad_token": "[PAD]"})
max_response_token_length = 200

models = defaultdict(list)
for root, dirs, files in os.walk("sampled_data"):
    for file in files:
        if file.endswith(".csv") and not any([file.endswith(ending) for ending in endings]):
            csv = os.path.join(root, file)
            exp_name = csv.split("/")[2]
            model_name = exp_name + "/" + "/".join(csv[csv.find("EleutherAI"):].split("/")[:2])
            models[model_name].append(csv)

for model_name in sorted(models.keys()):
    print(f"analyzing {model_name}")
    for seed_idx, csv in enumerate(models[model_name]):
        df = pd.read_csv(csv)
        if "response" in df.columns:
            df["postprocessed_response"] = df["response"]

        df["token_len"] = df["postprocessed_response"].apply(
            lambda x: len(tokenizer(x)["input_ids"]))
        print(f"{seed_idx=}, {df['token_len'].mean()=}")
