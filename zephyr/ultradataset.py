import copy
import multiprocessing
import os
import time
from dataclasses import dataclass, field
from pprint import pformat
from typing import Dict, Literal, Optional

import matplotlib.pyplot as plt
import pandas as pd
import tyro
from datasets import load_dataset
from huggingface_hub import HfApi
from huggingface_hub.repocard import RepoCard
from rich.pretty import pprint
from transformers import AutoTokenizer

api = HfApi()


"""
poetry run python -i zephyr/ultradataset.py --push_to_hub
"""


@dataclass
class TaskQueryHParams:
    length: Optional[int] = None
    format_str: Optional[str] = None
    truncate_field: Optional[str] = None
    truncate_text: Optional[str] = None
    padding: Optional[Literal["empty_space", "pad_token"]] = None
    pad_token: Optional[str] = None
    pad_side: Optional[str] = None
    max_query_length: Optional[int] = None
    max_sft_query_response_length: Optional[int] = None
    max_sft_response_length: Optional[int] = None
    max_rm_query_response_length: Optional[int] = None
    max_rm_response_length: Optional[int] = None


@dataclass
class Args:
    base_model: str = "mistralai/Mistral-7B-v0.1"  #  "gpt2"
    hf_entity: Optional[str] = None
    push_to_hub: bool = False
    check_length_correctness: bool = True
    debug: bool = False
    params: TaskQueryHParams = field(
        default_factory=lambda: TaskQueryHParams(
            length=3000,
            format_str="SUBREDDIT: r/{subreddit}\n\nTITLE: {title}\n\nPOST: {post}\n\nTL;DR:",
            truncate_field="post",
            truncate_text="\n",
            padding="pad_token",
            pad_side="left",
            max_query_length=3000,
            max_sft_query_response_length=4000,
            max_sft_response_length=1500,
            max_rm_query_response_length=4500,
            max_rm_response_length=1500,
        )
    )


def ceil_div(a, b):
    return (a - 1) // b + 1


if __name__ == "__main__":
    args = tyro.cli(Args)
    if args.hf_entity is None:
        args.hf_entity = api.whoami()["name"]
        assert isinstance(args.hf_entity, str)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, padding_side="right")
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.chat_template = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
    left_tokenizer = AutoTokenizer.from_pretrained(args.base_model, padding_side="left")
    left_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    left_tokenizer.chat_template = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"

    # post init
    if args.params.padding == "empty_space":
        args.params.pad_token = tokenizer.encode(" ")
    else:
        args.params.pad_token = [tokenizer.pad_token_id]
    pprint(args)
    timestamp = int(time.time())
    sft_ds = load_dataset("HuggingFaceH4/ultrachat_200k")
    if args.debug: # reduce the dataset size; faster debugging
        for split in sft_ds.keys():
            sft_ds[split] = sft_ds[split].select(range(1000))
    def process(x):
        full_query_token = tokenizer.apply_chat_template(x["messages"][:-1], add_generation_prompt=True)
        full_query_reference_response_token = tokenizer.apply_chat_template(x["messages"])
        full_reference_response_token = full_query_reference_response_token[len(full_query_token):]
        # ensure `reference_response_token` is of length `max_sft_response_length`
        reference_response_token = full_reference_response_token[:args.params.max_sft_response_length]
        if len(reference_response_token) < args.params.max_sft_response_length:
            reference_response_token = reference_response_token + [tokenizer.pad_token_id] * (args.params.max_sft_response_length - len(reference_response_token))
        assert len(reference_response_token) == args.params.max_sft_response_length
        x["query"] = x["messages"][:-1]
        x["query_token"] = left_tokenizer.apply_chat_template(
            x["messages"][:-1],
            padding="max_length",
            max_length=args.params.max_query_length,
            add_generation_prompt=True,
        )
        x["query_reference_response"] = x["messages"]
        x["query_reference_response_token"] = tokenizer.apply_chat_template(
            x["messages"],
            padding="max_length",
            max_length=args.params.max_sft_query_response_length,
            truncation=True,
        )
        x["query_reference_response_token_len"] = len(full_query_reference_response_token)
        x["query_token_len"] = len(full_query_token)
        x["reference_response"] = x["messages"][-1]
        x["reference_response_token"] = reference_response_token
        x["reference_response_token_len"] = len(full_reference_response_token)
        return x
    sft_ds = sft_ds.map(process, load_from_cache_file=False, num_proc=1 if args.debug else multiprocessing.cpu_count())
    del sft_ds["test_gen"], sft_ds["train_gen"]
    os.makedirs("dataset_visuals", exist_ok=True)
    fig, axs = plt.subplots(len(sft_ds), 3, figsize=(16, 8))
    for idx, split in enumerate(sft_ds.keys()):
        df = sft_ds[split].to_pandas()
        print(f'{df["query_reference_response_token_len"].mean()=}')
        print(f'{df["query_token_len"].mean()=}')
        # plot the length distribution
        axs[idx][0].hist(df["query_reference_response_token_len"], bins=100)
        axs[idx][0].set_title(f"{split} -- query_reference_response_token_len")
        axs[idx][1].hist(df["query_token_len"], bins=100)
        axs[idx][1].set_title(f"{split} -- query_token_len")
        axs[idx][2].hist(df["reference_response_token_len"], bins=100)
        axs[idx][2].set_title(f"{split} -- reference_response_token_len")
    fig.tight_layout()
    fig.savefig("dataset_visuals/ultrachat_200k.png")

    # based on the length distribution, we can set a max length using --params.max_sft_query_response_length
    for split in sft_ds.keys():
        sft_ds[split] = sft_ds[split].filter(
            lambda x: x["query_reference_response_token_len"] <= args.params.max_sft_query_response_length \
                and x["query_token_len"] <= args.params.max_query_length \
                and x["reference_response_token_len"] <= args.params.max_sft_response_length,
            num_proc=1 if args.debug else multiprocessing.cpu_count(),
        )
    if args.push_to_hub:
        sft_dataset_hf_path = f"{args.hf_entity}/ultrachat_200k_filtered_{timestamp}"
        sft_ds.push_to_hub(sft_dataset_hf_path)
        sft_card = RepoCard.load(sft_dataset_hf_path, repo_type="dataset")
        sft_card.text = f"""\
# Args

```python
{pformat(vars(args))}
```
"""
        sft_card.push_to_hub(sft_dataset_hf_path, repo_type="dataset")

    label_ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized")
    # label_ds = label_ds.remove_columns(["test_gen", "test_sft", "train_gen", "train_sft", "train_gen"])
    del label_ds["test_gen"], label_ds["test_sft"], label_ds["train_gen"], label_ds["train_sft"]
    if args.debug: # reduce the dataset size; faster debugging
        for split in label_ds.keys():
            label_ds[split] = label_ds[split].select(range(1000))
    
    def process(x):
        # x["chosen"] = x["chosen"]
        # x["rejected"] = x["rejected"]
        full_query_token = tokenizer.apply_chat_template(x["messages"][:-1], add_generation_prompt=True)
        full_query_chosen_token = tokenizer.apply_chat_template(x["chosen"])
        full_query_rejected_token = tokenizer.apply_chat_template(x["rejected"])
        full_rejected_token = full_query_rejected_token[len(full_query_token):]
        full_chosen_token = full_query_chosen_token[len(full_query_token):]
        # ensure `rejected_token` is of length `max_rm_response_length`
        rejected_token = full_rejected_token[:args.params.max_rm_response_length]
        if len(rejected_token) < args.params.max_rm_response_length:
            rejected_token = rejected_token + [tokenizer.pad_token_id] * (args.params.max_rm_response_length - len(rejected_token))
        assert len(rejected_token) == args.params.max_rm_response_length
        # ensure `chosen_token` is of length `max_rm_response_length`
        chosen_token = full_chosen_token[:args.params.max_rm_response_length]
        if len(chosen_token) < args.params.max_rm_response_length:
            chosen_token = chosen_token + [tokenizer.pad_token_id] * (args.params.max_rm_response_length - len(chosen_token))
        assert len(chosen_token) == args.params.max_rm_response_length
        x["query"] = x["messages"][:-1]
        x["query_token"] = left_tokenizer.apply_chat_template(
            x["messages"][:-1],            
            padding="max_length",
            max_length=args.params.max_query_length,
            add_generation_prompt=True,
        )
        x["query_token_len"] = len(full_query_token)
        x["query_chosen_token"] = tokenizer.apply_chat_template(
            x["chosen"],
            padding="max_length",
            max_length=args.params.max_rm_query_response_length,
            truncation=True,
        )
        x["query_chosen_token_len"] = len(full_query_chosen_token)
        x["chosen_token"] = chosen_token
        x["chosen_token_len"] = len(full_chosen_token)
        x["query_rejected_token"] = tokenizer.apply_chat_template(
            x["rejected"],
            padding="max_length",
            max_length=args.params.max_rm_query_response_length,
            truncation=True,
        )
        x["query_rejected_token_len"] = len(full_query_rejected_token)
        x["rejected_token"] = full_rejected_token
        x["rejected_token_len"] = len(full_rejected_token)
        return x
    
    label_ds = label_ds.map(process, load_from_cache_file=False, num_proc=1 if args.debug else multiprocessing.cpu_count())

    # visualize the length distribution
    fig, axs = plt.subplots(len(label_ds), 5, figsize=(16, 8))
    for idx, split in enumerate(label_ds.keys()):
        df = label_ds[split].to_pandas()
        axs[idx][0].hist(df["query_token_len"], bins=100)
        axs[idx][0].set_title(f"{split} -- query_token_len")
        axs[idx][1].hist(df["query_chosen_token_len"], bins=100)
        axs[idx][1].set_title(f"{split} -- query_chosen_token_len")
        axs[idx][2].hist(df["query_rejected_token_len"], bins=100)
        axs[idx][2].set_title(f"{split} -- query_rejected_token_len")
        axs[idx][3].hist(df["chosen_token_len"], bins=100)
        axs[idx][3].set_title(f"{split} -- chosen_token_len")
        axs[idx][4].hist(df["rejected_token_len"], bins=100)
        axs[idx][4].set_title(f"{split} -- rejected_token_len")
    fig.tight_layout()
    fig.savefig("dataset_visuals/ultrafeedback_binarized.png")

    # based on the length distribution, we can set a max length using --params.max_rm_query_response_length
    for split in label_ds.keys():
        label_ds[split] = label_ds[split].filter(
            lambda x: x["query_chosen_token_len"] <= args.params.max_rm_query_response_length \
                and x["query_rejected_token_len"] <= args.params.max_rm_query_response_length \
                and x["query_token_len"] <= args.params.max_query_length \
                and x["chosen_token_len"] <= args.params.max_rm_response_length \
                and x["rejected_token_len"] <= args.params.max_rm_response_length,
            num_proc=1 if args.debug else multiprocessing.cpu_count(),
        )


    if args.push_to_hub:
        rm_dataset_hf_path = f"{args.hf_entity}/ultrafeedback_binarized_{timestamp}"
        label_ds.push_to_hub(rm_dataset_hf_path)

    if args.push_to_hub:
        print(f"{__file__=}")
        for hf_path in [rm_dataset_hf_path, sft_dataset_hf_path]:
            api.upload_folder(
                folder_path="dataset_visuals",
                path_in_repo="dataset_visuals",
                repo_id=hf_path,
                repo_type="dataset",
            )
            api.upload_file(
                path_or_fileobj=__file__,
                path_in_repo="create_dataset.py",
                repo_id=hf_path,
                repo_type="dataset",
            )
            print(f"✨ Pushed to hub: https://huggingface.co/datasets/{hf_path}")

