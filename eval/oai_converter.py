import pandas as pd
for split in ["test", "validation"]:
    df = pd.read_json(path_or_buf=f"sampled_data/openai_ppo_original_1.3b/{split}.jsonl", lines=True)
    df = df[['context', 'samples', 'ref']]
    df = df.rename(columns={'context': 'query', 'samples': 'postprocessed_response', 'ref': 'reference_responses'})
    # remove leading spaces
    df["query"] = df["query"].apply(lambda x: x.strip())
    # add leading spaces to responses to be consistent with our setup
    df["postprocessed_response"] = df["postprocessed_response"].apply(lambda x: f" {x[0]}")
    df["reference_responses"] = df["reference_responses"].apply(lambda x: f" {x}")
    df.to_csv(f"sampled_data/openai_ppo_original_1.3b/{split}.csv")
