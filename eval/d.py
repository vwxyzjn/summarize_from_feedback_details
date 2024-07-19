import numpy as np
import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModel,
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


import wandb

runs = list(wandb.Api().runs(
    path=f"costa-huang/tldr_summarize",
    filters={
        "$and": [
            {f"config.exp_name.value": "reward"},
            {"tags": {"$in": ["refactor-chosen-rejected3"]}},
        ]
    }
))

for run in runs:
    if run.state != "finished":
        continue
    hf_repo_url = run.config['hf_repo_url']
    hf_repo_id = run.config["hf_repo_id"]
    revision = run.config["run_name"]
    print(f"{run.config['base_model']}__{run.config['seed']}: {run.config['hf_repo_url']}")
    # scalar_model_config = ScalarModelConfig.from_pretrained(
    #     hf_repo_id,
    #     revision=revision,
    #     trust_remote_code=True,
    # )
    # # hack to remove the path
    # # models/EleutherAI/pythia-6.9b-deduped/sft_model_55513 -> EleutherAI/pythia-6.9b-deduped
    # original_model = "/".join(scalar_model_config.base_config["_name_or_path"].split("/")[1:3])
    # scalar_model_config.base_config["_name_or_path"] = original_model
    # scalar_model_config.base_model = original_model
    # rm: PreTrainedModel = ScalarModel.from_pretrained(
    #     hf_repo_id,
    #     revision=revision,
    #     trust_remote_code=True,
    #     config=scalar_model_config,
    # )
    # raise