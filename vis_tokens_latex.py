from collections import defaultdict

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

import torch
import torch.nn as nn

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    PreTrainedModel,
    AutoModelForCausalLM,
    GenerationConfig,
    AutoConfig,
    PretrainedConfig,
    AutoModel,
)


######
# Utility functions
######


class ScalarModelConfig(PretrainedConfig):
    def __init__(
        self,
        base_model: str = "EleutherAI/pythia-160m",
        base_config: PretrainedConfig = AutoConfig.from_pretrained(
            "EleutherAI/pythia-160m"
        ),
        hidden_size: int = 768,
        bias: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_model = base_model
        self.base_config = base_config
        self.hidden_size = hidden_size
        self.bias = bias


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.normal_(layer.weight, std=std)
    torch.nn.init.constant_(layer.bias, val=bias_const)
    return layer


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


def print_rich_table(title: str, df: pd.DataFrame, console: Console) -> Table:
    table = Table(show_lines=True)
    for column in df.columns:
        table.add_column(column)
    for _, row in df.iterrows():
        table.add_row(*row.astype(str).tolist())
    console.rule(f"[bold red]{title}")
    console.print(table)


def get_reward(model, query_responses, tokenizer):
    attention_mask = query_responses != tokenizer.pad_token_id
    input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
    reward_logits = model(
        input_ids=input_ids.to("cpu"),
        attention_mask=attention_mask.to("cpu"),
        return_dict=True,
        output_hidden_states=True,
    )
    sequence_lengths = (
        torch.eq(query_responses, tokenizer.pad_token_id).long().argmax(-1) - 1
    ).to("cpu")

    # https://github.com/huggingface/transformers/blob/dc68a39c8111217683bf49a4912d0c9018bab33d/src/transformers/models/gpt2/modeling_gpt2.py#L1454
    return reward_logits[
        torch.arange(reward_logits.size(0), device=reward_logits.device),
        sequence_lengths,
    ]


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

sft_dataset = load_dataset(
    "vwxyzjn/summarize_from_feedback_tldr_3_filtered_oai_preprocessing_1706381144"
)

# base
base_model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
    "EleutherAI/pythia-1b-deduped"
).to(device)

scalar_model_config = ScalarModelConfig.from_pretrained(
    "vwxyzjn/EleutherAI_pythia-6.9b-deduped__reward__tldr",
    revision="reward__55513__1706651113",
    trust_remote_code=True,
)
# hack to remove the path
# models/EleutherAI/pythia-6.9b-deduped/sft_model_55513 -> EleutherAI/pythia-6.9b-deduped
original_model = "/".join(
    scalar_model_config.base_config["_name_or_path"].split("/")[1:3]
)
scalar_model_config.base_config["_name_or_path"] = original_model
scalar_model_config.base_model = original_model
rm: PreTrainedModel = ScalarModel.from_pretrained(
    "vwxyzjn/EleutherAI_pythia-6.9b-deduped__reward__tldr",
    revision="reward__55513__1706651113",
    trust_remote_code=True,
    config=scalar_model_config,
).to("cpu")

compare = "ppo_model"

if compare == "sft_model":
    # # https://wandb.ai/costa-huang/tldr_summarize/runs/a0rutstb
    # # https://huggingface.co/vwxyzjn/EleutherAI_pythia-1b-deduped__sft__tldr/tree/sft__55513__1706646024
    compare_model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        "vwxyzjn/EleutherAI_pythia-1b-deduped__sft__tldr",
        revision="sft__55513__1706646024",
        trust_remote_code=True,
    ).to(device)
elif compare == "ppo_model":
    # https://wandb.ai/costa-huang/tldr_summarize/runs/ulekmmac
    # https://huggingface.co/vwxyzjn/EleutherAI_pythia-1b-deduped__ppo_left_padding__tldr/tree/ppo_left_padding__55513__1706746254
    compare_model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        "vwxyzjn/EleutherAI_pythia-1b-deduped__ppo_left_padding__tldr",
        revision="ppo_left_padding__55513__1706746254",
        trust_remote_code=True,
    ).to(device)
else:
    # https://wandb.ai/costa-huang/tldr_summarize/runs/tewm564g
    # https://huggingface.co/vwxyzjn/EleutherAI_pythia-1b-deduped__dpo__tldr/tree/dpo__55513__1707379566
    compare_model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        "vwxyzjn/EleutherAI_pythia-1b-deduped__dpo__tldr",
        revision="dpo__55513__1707379566",
        trust_remote_code=True,
    ).to(device)

# compared_models = {
#     "base_model": base_model,
#     "ppo_model": ppo_model,
#     "sft_model": sft_model,
#     "dpo_model": dpo_model,
# }

nchecks = 4
colors = {
    0: "\sethlcolor{LightOrchid}",
    1: "\sethlcolor{LightYellowGreen}",
    2: "\sethlcolor{LightYellowOrange}",
    3: "\sethlcolor{LightSalmon}",
}
top_k = 10


console = Console()
for i in range(len(sft_dataset["validation"])):
    table = defaultdict(list)
    query = (
        torch.Tensor(sft_dataset["validation"][i : i + 1]["query_token"])
        .to(device)
        .long()
    )
    context_length = query.shape[1]
    query_reference_response = (
        torch.Tensor(sft_dataset["validation"][i]["reference_response_token"])
        .to(device)
        .long()
    )
    import pdb

    pdb.set_trace()
    with torch.no_grad():
        base_model = base_model
        aligned_model = compare_model

        context_length = query.shape[1]
        aligned_model_query_response = generate(
            aligned_model, query, tokenizer, validation_generation_config
        )
        aligned_model_response = aligned_model_query_response[:, context_length:]
        base_model_query_response = generate(
            base_model, query, tokenizer, validation_generation_config
        )
        base_model_response = base_model_query_response[:, context_length:]

        reward = get_reward(rm, aligned_model_response, tokenizer)

        aligned_model_output = forward(
            aligned_model, aligned_model_query_response, tokenizer
        )
        base_model_output = forward(base_model, aligned_model_query_response, tokenizer)
        aligned_model_logits = aligned_model_output.logits[:, context_length - 1 : -1]
        _, aligned_model_topk_indices = aligned_model_logits.topk(top_k)
        base_model_logits = base_model_output.logits[:, context_length - 1 : -1]
        _, base_model_topk_indices = base_model_logits.topk(top_k)

        aligned_model_topk_indices[:, :, 0:1].expand(-1, -1, nchecks)
        matches = (
            aligned_model_topk_indices[:, :, 0:1].expand(-1, -1, nchecks)
            == base_model_topk_indices[:, :, 0:nchecks]
        )
        matched = matches.sum(2)
        match_idx = matches.float().argmax(2)
        final_matches = torch.where(matched > 0, match_idx, nchecks - 1)
        stats = torch.stack([(final_matches == i).sum(1) for i in range(nchecks)]).T

    final_matches = final_matches.tolist()
    aligned_model_response = aligned_model_response.tolist()
    table["type"].append("Query")
    table["content"].append(tokenizer.decode(query[0], skip_special_tokens=True))
    table["type"].append(f"{compare} Response")
    table["content"].append(
        "".join(
            [
                f"{colors[jt]}" "\hl{" f"{tokenizer.decode(it)}" "}"
                for it, jt in zip(aligned_model_response[0], final_matches[0])
            ]
        )
    )
    table["type"].append("score (RM)")
    table["content"].append(round(reward[0][0].item(), 2))
    table["type"].append("Matched Color Counts")
    table["content"].append(stats[0].cpu().numpy())
    table["type"].append("Refernce Model Response")
    table["content"].append(
        tokenizer.decode(
            sft_dataset["validation"][i]["reference_response_token"],
            skip_special_tokens=True,
        )
    )
    table["type"].append("Base Model Response")
    table["content"].append(
        tokenizer.decode(base_model_response[0], skip_special_tokens=True)
    )

    df = pd.DataFrame(table)

    print(df.to_latex(index=False, header=False))
    if input("Continue? (press `n` to stop) ") == "n":
        break
