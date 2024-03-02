from collections import defaultdict

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
    return reward_logits[torch.arange(reward_logits.size(0), device=reward_logits.device), sequence_lengths]


def print_rich_table(title: str, df: pd.DataFrame, console: Console) -> Table:
    table = Table(show_lines=True)
    for column in df.columns:
        table.add_column(column)
    for _, row in df.iterrows():
        table.add_row(*row.astype(str).tolist())
    console.rule(f"[bold red]{title}")
    console.print(table)


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-1b-deduped")
tokenizer.add_special_tokens({"pad_token": "[PAD]"})
response_length = 128
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
rm_dataset = load_dataset("vwxyzjn/summarize_from_feedback_oai_preprocessing_1706381144")

# https://wandb.ai/costa-huang/tldr_summarize/runs/a0rutstb
# https://huggingface.co/vwxyzjn/EleutherAI_pythia-1b-deduped__sft__tldr/tree/sft__55513__1706646024
sft_model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
    "vwxyzjn/EleutherAI_pythia-1b-deduped__sft__tldr",
    revision="sft__55513__1706646024",
    trust_remote_code=True,
).to(device)

# https://wandb.ai/costa-huang/tldr_summarize/runs/ulekmmac
# https://huggingface.co/vwxyzjn/EleutherAI_pythia-1b-deduped__ppo_left_padding__tldr/tree/ppo_left_padding__55513__1706746254
ppo_model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
    "vwxyzjn/EleutherAI_pythia-1b-deduped__ppo_left_padding__tldr",
    revision="ppo_left_padding__55513__1706746254",
    trust_remote_code=True,
).to(device)

# https://wandb.ai/costa-huang/tldr_summarize/runs/tewm564g
# https://huggingface.co/vwxyzjn/EleutherAI_pythia-1b-deduped__dpo__tldr/tree/dpo__55513__1707379566
dpo_model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
    "vwxyzjn/EleutherAI_pythia-1b-deduped__dpo__tldr",
    revision="dpo__55513__1707379566",
    trust_remote_code=True,
).to(device)

# # https://wandb.ai/costa-huang/tldr_summarize/runs/jsj57urt
# # https://huggingface.co/vwxyzjn/EleutherAI_pythia-1b-deduped__reward__tldr/tree/reward__55513__1706651113
# scalar_model_config = ScalarModelConfig.from_pretrained(
#     "vwxyzjn/EleutherAI_pythia-1b-deduped__reward__tldr",
#     revision="reward__55513__1706651113",
#     trust_remote_code=True,
# )
# # hack to remove the path
# # models/EleutherAI/pythia-1b-deduped/sft_model_55513 -> EleutherAI/pythia-1b-deduped
# original_model = "/".join(scalar_model_config.base_config["_name_or_path"].split("/")[1:3])
# scalar_model_config.base_config["_name_or_path"] = original_model
# scalar_model_config.base_model = original_model
# rm: PreTrainedModel = ScalarModel.from_pretrained(
#     "vwxyzjn/EleutherAI_pythia-1b-deduped__reward__tldr",
#     revision="reward__55513__1706651113",
#     trust_remote_code=True,
#     config=scalar_model_config,
# ).to(device)

# "Gold" RM (a much larger model)
# https://wandb.ai/costa-huang/tldr_summarize/runs/ddw0ixx9
# https://huggingface.co/vwxyzjn/EleutherAI_pythia-6.9b-deduped__reward__tldr/tree/reward__55513__1706651113
scalar_model_config = ScalarModelConfig.from_pretrained(
    "vwxyzjn/EleutherAI_pythia-6.9b-deduped__reward__tldr",
    revision="reward__55513__1706651113",
    trust_remote_code=True,
)
# hack to remove the path
# models/EleutherAI/pythia-6.9b-deduped/sft_model_55513 -> EleutherAI/pythia-6.9b-deduped
original_model = "/".join(scalar_model_config.base_config["_name_or_path"].split("/")[1:3])
scalar_model_config.base_config["_name_or_path"] = original_model
scalar_model_config.base_model = original_model
rm: PreTrainedModel = ScalarModel.from_pretrained(
    "vwxyzjn/EleutherAI_pythia-6.9b-deduped__reward__tldr",
    revision="reward__55513__1706651113",
    trust_remote_code=True,
    config=scalar_model_config,
).to(device)


console = Console()
for i in range(len(sft_dataset["validation"])):
    table = defaultdict(list)
    query = torch.Tensor(sft_dataset["validation"][i]["query_token"]).to(device).long()
    query_reference_response = torch.Tensor(sft_dataset["validation"][i]["query_reference_response_token"]).to(device).long()
    with torch.no_grad():
        sft_query_response = generate(sft_model, query.unsqueeze(0), tokenizer, validation_generation_config)
        sft_response = sft_query_response[:, query.shape[0] :]
        ppo_query_response = generate(ppo_model, query.unsqueeze(0), tokenizer, validation_generation_config)
        ppo_response = ppo_query_response[:, query.shape[0] :]
        dpo_query_response = generate(dpo_model, query.unsqueeze(0), tokenizer, validation_generation_config)
        dpo_response = dpo_query_response[:, query.shape[0] :]
        sft_reward = get_reward(rm, sft_query_response, tokenizer)
        ppo_reward = get_reward(rm, ppo_query_response, tokenizer)
        dpo_reward = get_reward(rm, dpo_query_response, tokenizer)
        reference_reward = get_reward(rm, query_reference_response.unsqueeze(0), tokenizer)

    # print results
    table["type"].append("Query")
    table["content"].append(tokenizer.decode(query, skip_special_tokens=True))
    table["score (RM)"].append("-")
    table["type"].append("SFT response")
    table["content"].append(tokenizer.decode(sft_response[0]))
    table["score (RM)"].append(sft_reward[0][0].item())
    table["type"].append("PPO response")
    table["content"].append(tokenizer.decode(ppo_response[0]))
    table["score (RM)"].append(ppo_reward[0][0].item())
    table["type"].append("DPO response")
    table["content"].append(tokenizer.decode(dpo_response[0]))
    table["score (RM)"].append(dpo_reward[0][0].item())
    table["type"].append("Reference response")
    table["content"].append(sft_dataset["validation"][i]["reference_response"])
    table["score (RM)"].append(reference_reward[0][0].item())
    df = pd.DataFrame(table)
    print_rich_table("Results", df, console)
    if input("Continue? (press `n` to stop) ") == "n":
        break
