from collections import defaultdict
from dataclasses import dataclass
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import load_dataset
from rich.console import Console
from rich.table import Table
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    PretrainedConfig,
    PreTrainedModel,
)

@dataclass
class RunRecord:
    wandb_url: str
    hf_repo_url: str
    hf_repo_id: str
    revision: str

console = Console()

if not os.path.exists("release_runs.csv"):
    import wandb
    keys = {
        "sft": "refactor-chosen-rejected3",
        "reward": "refactor-chosen-rejected3",
        "ppo_left_padding_new_nowhiten_reward": "refactor-chosen-rejected3",
        "dpo": "refactor-chosen-rejected2",
    }
    runs = []
    for exp_name, tag in keys.items():
        runs.extend(list(wandb.Api().runs(
            path=f"costa-huang/tldr_summarize",
            filters={
                "$and": [
                    {f"config.exp_name.value": exp_name},
                    {"tags": {"$in": [tag]}},
                ]
            }
        )))
    table = defaultdict(list)
    for i in range(len(runs)):
        table["base_model"].append(runs[i].config["base_model"])
        table["exp"].append(runs[i].config["exp_name"])
        table["seed"].append(runs[i].config["seed"])
        table["wandb_url"].append(runs[i].url)
        table["hf_repo_url"].append(runs[i].config["hf_repo_url"])
        table["hf_repo_id"].append(runs[i].config["hf_repo_id"])
        table["revision"].append(runs[i].config["run_name"])
    df = pd.DataFrame(table)
    df.to_csv("release_runs.csv", index=False)
else:
    df = pd.read_csv("release_runs.csv")


df = df.groupby(["base_model", "exp", "seed"]).agg(lambda x: x.tolist()[0])

# feel free to change the base_model, exp, and seed; the seeds are 44413, 55513, 66613, 77713
sft_record = RunRecord(**df.loc[("EleutherAI/pythia-1b-deduped", "sft", 55513)])
ppo_record = RunRecord(**df.loc[("EleutherAI/pythia-1b-deduped", "ppo_left_padding_new_nowhiten_reward", 55513)])
dpo_record = RunRecord(**df.loc[("EleutherAI/pythia-1b-deduped", "dpo", 55513)])
# rm_record = RunRecord(**df.loc[("EleutherAI/pythia-1b-deduped", "reward", 55513)])
rm_record = RunRecord(**df.loc[("EleutherAI/pythia-6.9b-deduped", "reward", 55513)]) # larger (in some sense gold) RM

######
# RM model definition
######

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.normal_(layer.weight, std=std)
    torch.nn.init.constant_(layer.bias, val=bias_const)
    return layer


class ScalarModelConfig(PretrainedConfig):
    def __init__(
        self,
        base_model: str = "EleutherAI/pythia-160m",
        base_config: PretrainedConfig = AutoConfig.from_pretrained("EleutherAI/pythia-160m"),
        hidden_size: int = 768,
        bias: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_model = base_model
        self.base_config = base_config
        self.hidden_size = hidden_size
        self.bias = bias


class ScalarModel(PreTrainedModel):
    config_class = ScalarModelConfig

    def __init__(self, config: ScalarModelConfig):
        super().__init__(config)
        self.config = config
        self.lm_backbone = AutoModel.from_pretrained(
            config.base_model,
            config=self.config.base_config,
            trust_remote_code=True,
        )
        self.scalar_head = layer_init(
            nn.Linear(self.config.hidden_size, 1),
            std=1 / np.sqrt(self.config.hidden_size + 1),
        )

    def forward(self, **kwargs):
        output = self.lm_backbone(**kwargs)
        reward = self.scalar_head(output.hidden_states[-1]) - self.config.bias
        return reward


######
# Utility functions
######


def generate(lm_backbone, queries, tokenizer, generation_config):
    """generate in a way that does not affect padding tokens"""
    context_length = queries.shape[1]
    attention_mask = queries != tokenizer.pad_token_id
    input_ids = torch.masked_fill(queries, ~attention_mask, 0)
    output = lm_backbone.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        # position_ids=attention_mask.cumsum(1) - attention_mask.long(), # generation collapsed if this was turned on. TODO: why does generation collapse with this?
        generation_config=generation_config,
        return_dict_in_generate=True,
    )
    return torch.cat((queries, output.sequences[:, context_length:]), dim=1)


def forward(model, query_responses, tokenizer):
    attention_mask = query_responses != tokenizer.pad_token_id
    position_ids = attention_mask.cumsum(1) - attention_mask.long()
    input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
    return model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        return_dict=True,
        output_hidden_states=True,
    )


def get_reward(model, query_responses, tokenizer):
    attention_mask = query_responses != tokenizer.pad_token_id
    input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
    reward_logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_dict=True,
        output_hidden_states=True,
    )
    sequence_lengths = (torch.eq(query_responses, tokenizer.pad_token_id).long().argmax(-1) - 1).to(query_responses.device)
    # https://github.com/huggingface/transformers/blob/dc68a39c8111217683bf49a4912d0c9018bab33d/src/transformers/models/gpt2/modeling_gpt2.py#L1454
    return reward_logits[torch.arange(reward_logits.size(0), device=reward_logits.device), sequence_lengths], reward_logits


def print_rich_table(title: str, df: pd.DataFrame, console: Console) -> Table:
    table = Table(show_lines=True)
    for column in df.columns:
        table.add_column(column)
    for _, row in df.iterrows():
        table.add_row(*row.astype(str).tolist())
    console.rule(f"[bold red]{title}")
    console.print(table)


######
# Start
######

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-1b-deduped")
tokenizer.add_special_tokens({"pad_token": "[PAD]"})
response_length = 80
validation_generation_config = GenerationConfig(
    max_new_tokens=response_length,
    temperature=(0.01 + 1e-7),
    top_k=0.0,
    top_p=1.0,
    do_sample=True,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

sft_dataset = load_dataset("vwxyzjn/summarize_from_feedback_tldr_3_filtered_oai_preprocessing_1706381144")
base_model: PreTrainedModel = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-1b-deduped").to(device)

console.print("loading", sft_record)
sft_model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
    sft_record.hf_repo_id,
    revision=sft_record.revision,
    trust_remote_code=True,
).to(device)
console.print("loading", ppo_record)
ppo_model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
    ppo_record.hf_repo_id,
    revision=ppo_record.revision,
    trust_remote_code=True,
).to(device)
console.print("loading", dpo_record)
dpo_model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
    dpo_record.hf_repo_id,
    revision=dpo_record.revision,
    trust_remote_code=True,
).to(device)
console.print("loading", rm_record)
scalar_model_config = ScalarModelConfig.from_pretrained(
    rm_record.hf_repo_id,
    revision=rm_record.revision,
    trust_remote_code=True,
)
# hack to remove the path
# models/EleutherAI/pythia-6.9b-deduped/sft_model_55513 -> EleutherAI/pythia-6.9b-deduped
original_model = "/".join(scalar_model_config.base_config["_name_or_path"].split("/")[1:3])
scalar_model_config.base_config["_name_or_path"] = original_model
scalar_model_config.base_model = original_model
rm: PreTrainedModel = ScalarModel.from_pretrained(
    rm_record.hf_repo_id,
    revision=rm_record.revision,
    trust_remote_code=True,
    config=scalar_model_config,
).to(device)

nchecks = 4
colors = {
    0: "on blue",
    1: "on yellow",
    2: "on yellow",
    3: "on red",
}
latex_colors = {
    0: "\sethlcolor{LightBlue}",
    1: "\sethlcolor{LightYellow}",
    2: "\sethlcolor{LightYellow}",
    3: "\sethlcolor{LightRed}",
}
include_logits = False
for i in range(len(sft_dataset["validation"])):
    rich_table = defaultdict(list)
    latex_table = defaultdict(list)
    query = torch.LongTensor(sft_dataset["validation"][i : i + 1]["query_token"]).to(device)
    context_length = query.shape[1]
    query_reference_response = torch.cat((query, torch.LongTensor(tokenizer.encode(sft_dataset["validation"][i]["reference_response"])).to(device).unsqueeze(0)), dim=1)
    for table in [rich_table, latex_table]:
        table["Type"].append("Query")
        table["Content"].append(tokenizer.decode(query[0], skip_special_tokens=True))
        table["Score (RM)"].append("N/A")
    with torch.no_grad():
        model_stats = defaultdict(list)
        for aligned_model, model_name in zip(
            [sft_model, ppo_model, dpo_model],
            ["SFT Model Response", "PPO Model Response", "DPO Model Response"],
        ):
            aligned_model_query_response = generate(aligned_model, query, tokenizer, validation_generation_config)
            aligned_model_response = aligned_model_query_response[:, context_length:]
            aligned_model_reward, aligned_model_reward_logits = get_reward(rm, aligned_model_query_response, tokenizer)
            aligned_model_reward_logits = aligned_model_reward_logits.squeeze(-1)[:, context_length-1:]

            # AI2 visualization https://allenai.github.io/re-align/tds.html
            aligned_model_output = forward(aligned_model, aligned_model_query_response, tokenizer)
            base_model_output = forward(base_model, aligned_model_query_response, tokenizer)
            aligned_model_logits = aligned_model_output.logits[:, context_length - 1 : -1]
            _, aligned_model_topk_indices = aligned_model_logits.topk(10)
            base_model_logits = base_model_output.logits[:, context_length - 1 : -1]
            _, base_model_topk_indices = base_model_logits.topk(10)
            aligned_model_topk_indices[:, :, 0:1].expand(-1, -1, nchecks)
            matches = aligned_model_topk_indices[:, :, 0:1].expand(-1, -1, nchecks) == base_model_topk_indices[:, :, 0:nchecks]
            matched = matches.sum(2)
            match_idx = matches.float().argmax(2)
            final_matches = torch.where(matched > 0, match_idx, nchecks - 1)
            stats = torch.stack([(final_matches == i).sum(1) for i in range(nchecks)]).T

            final_matches = final_matches.tolist()
            aligned_model_response = aligned_model_response.tolist()
            for table in [rich_table, latex_table]:
                table["Type"].append(model_name)
            latex_table["Content"].append(
                "".join(
                    [
                        f"{latex_colors[jt]}"  "\hl{" f"{tokenizer.decode(it)}" "}"
                        for it, jt in zip(aligned_model_response[0], final_matches[0])
                    ]
                )
            )
            rich_table["Content"].append(
                "".join(
                    [
                        f"[{colors[jt]}]{tokenizer.decode(it)}[/{colors[jt]}]"
                        for it, jt in zip(aligned_model_response[0], final_matches[0])
                    ]
                )
            )
            for table in [rich_table, latex_table]:
                table["Score (RM)"].append(str(round(aligned_model_reward[0][0].item(), 4)))
                if include_logits:
                    table["Type"].append(f"{model_name} Reward Logits")
                    table["Content"].append([round(logit, 4) for logit in aligned_model_reward_logits[0].tolist()])
                    table["Score (RM)"].append(str(round(aligned_model_reward[0][0].item(), 4)))
            # table["Type"].append("Matched Color Counts")
            # table["Content"].append(stats[0])
        reference_reward, reference_reward_logits = get_reward(rm, query_reference_response, tokenizer)
        reference_reward_logits = reference_reward_logits.squeeze(-1)[:, context_length-1:]
        for table in [rich_table, latex_table]:
            table["Type"].append("Reference response")
            table["Content"].append(sft_dataset["validation"][i]["reference_response"])
            table["Score (RM)"].append(str(round(reference_reward[0][0].item(), 4)))
            if include_logits:
                table["Type"].append("Reference Reward Logits")
                table["Content"].append([round(logit, 4) for logit in reference_reward_logits[0].tolist()])
                table["Score (RM)"].append(str(round(reference_reward[0][0].item(), 4)))

        base_model_query_response = generate(base_model, query, tokenizer, validation_generation_config)
        base_model_response = base_model_query_response[:, context_length:]
        base_model_reward, base_model_reward_logits = get_reward(rm, base_model_query_response, tokenizer)
        base_model_reward_logits = base_model_reward_logits.squeeze(-1)[:, context_length-1:]
        for table in [rich_table, latex_table]:
            table["Type"].append("Base Model Response")
            table["Content"].append(tokenizer.decode(base_model_response[0], skip_special_tokens=True))
            table["Score (RM)"].append(str(round(base_model_reward[0][0].item(), 4)))
            if include_logits:
                table["Type"].append("Base Model Reward Logits")
                table["Content"].append([round(logit, 4) for logit in base_model_reward_logits[0].tolist()])
                table["Score (RM)"].append(str(round(base_model_reward[0][0].item(), 4)))


    rich_df = pd.DataFrame(rich_table)
    latex_df = pd.DataFrame(latex_table)
    print_rich_table("Results", rich_df, console)
    # print(latex_df.to_latex(index=False))
    if input("Continue? (press `n` to stop) ") == "n":
        break
