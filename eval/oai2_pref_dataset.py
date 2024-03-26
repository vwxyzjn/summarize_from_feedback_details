import pandas as pd
import tyro
from datasets import load_dataset
from oai2 import ArgsWrapper, llm_judge

if __name__ == "__main__":
    all_args = tyro.cli(ArgsWrapper)
    args = all_args.args
    ljc = all_args.ljc

    dataset = load_dataset(
        "vwxyzjn/summarize_from_feedback_oai_preprocessing_1706381144", split="train"
    )

    dataset = dataset.select_columns(["query", "chosen", "rejected"])

    df = pd.DataFrame(dataset)

    df["chosen"] = df["chosen"].map(lambda x: x.split("<|endoftext|>")[0].strip())
    df["rejected"] = df["rejected"].map(lambda x: x.split("<|endoftext|>")[0].strip())
    df["prompt"] = df["query"].map(lambda x: x.strip())
    df["response0"] = df["chosen"].map(lambda x: x.strip())
    df["response1"] = df["rejected"].map(lambda x: x.strip())
    judge_df = llm_judge(ljc, df)
    judge_df.to_csv(args.output_path)
