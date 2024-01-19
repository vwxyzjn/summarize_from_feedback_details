from collections import defaultdict
import pandas as pd
from transformers import AutoTokenizer
import os

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
tokenizer.add_special_tokens({"pad_token": "[PAD]"})
max_response_token_length = 51

models = defaultdict(list)
for root, dirs, files in os.walk("sampled_data"):
    for file in files:
        if file.endswith(".csv") and not file.endswith("csv_judged.csv"):
            csv = os.path.join(root, file)
            exp_name = csv.split("/")[1]
            model_name = exp_name + "/" + "/".join(csv[csv.find("EleutherAI"):].split("/")[:2])
            models[model_name].append(csv)

for model_name in sorted(models.keys()):
    print(f"analyzing {model_name}")
    for seed_idx, csv in enumerate(models[model_name]):
        df = pd.read_csv(csv)
        if "response" in df.columns:
            df["postprocessed_response"] = df["response"]
        # tokenize df["postprocessed_response"]
        df["tokenized"] = df["postprocessed_response"].apply(
            lambda x: tokenizer(x, max_length=max_response_token_length, truncation=True, padding="max_length"
                                )["input_ids"])
        # determine if it ends with an EOS token
        df["ends_with_eos"] = df["tokenized"].apply(lambda x: x[-1] == tokenizer.pad_token_id)
        print(f"seed{seed_idx}: ====ends_with_eos perecentage", df["ends_with_eos"].sum() / len(df))