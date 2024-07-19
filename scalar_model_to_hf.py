from dataclasses import dataclass

import pandas as pd
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
)

AutoTokenizer
AutoModelForSequenceClassification
AutoModelForCausalLM


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
        self.scalar_head = nn.Linear(self.config.hidden_size, 1)

    def forward(self, **kwargs):
        output = self.lm_backbone(**kwargs)
        reward = self.scalar_head(output.hidden_states[-1]) - self.config.bias
        return reward


@dataclass
class RunRecord:
    wandb_url: str
    hf_repo_url: str
    hf_repo_id: str
    revision: str


df = pd.read_csv("release_runs.csv")
df = df.groupby(["base_model", "exp", "seed"]).agg(lambda x: x.tolist()[0])

# feel free to change the base_model, exp, and seed; the seeds are 44413, 55513, 66613, 77713
sft_record = RunRecord(**df.loc[("EleutherAI/pythia-1b-deduped", "sft", 77713)])
# sft_model = AutoModelForCausalLM.from_pretrained(
#     sft_record.hf_repo_id,
#     revision=sft_record.revision,
#     trust_remote_code=True,
# )
# sft_model.push_to_hub(sft_record.hf_repo_id.replace("vwxyzjn", "cleanrl"), use_temp_dir=True)
# sft_tokenizer = AutoTokenizer.from_pretrained(sft_record.hf_repo_id, revision=sft_record.revision)
# sft_tokenizer.push_to_hub(sft_record.hf_repo_id.replace("vwxyzjn", "cleanrl"), use_temp_dir=True)


rm_record = RunRecord(**df.loc[("EleutherAI/pythia-2.8b-deduped", "reward", 77713)])
rm_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-2.8b-deduped")
# scalar_model_config = ScalarModelConfig.from_pretrained(
#     rm_record.hf_repo_id,
#     revision=rm_record.revision,
#     trust_remote_code=True,
# )
# # hack to remove the path
# # models/EleutherAI/pythia-6.9b-deduped/sft_model_55513 -> EleutherAI/pythia-6.9b-deduped
# original_model = "/".join(scalar_model_config.base_config["_name_or_path"].split("/")[1:3])
# scalar_model_config.base_config["_name_or_path"] = original_model
# scalar_model_config.base_model = original_model
# src_scalar_model = ScalarModel.from_pretrained(
#     rm_record.hf_repo_id,
#     revision=rm_record.revision,
#     trust_remote_code=True,
#     config=scalar_model_config,
# )

# des_model = AutoModelForSequenceClassification.from_pretrained(original_model, num_labels=1)
# lm_backbone = getattr(des_model, des_model.base_model_prefix)
# print("loading")
# lm_backbone.load_state_dict(src_scalar_model.lm_backbone.state_dict())
# lm_head_state_dict = src_scalar_model.scalar_head.state_dict()
# del lm_head_state_dict["bias"]
# des_model.score.load_state_dict(lm_head_state_dict)

# des_model.push_to_hub(rm_record.hf_repo_id.replace("vwxyzjn", "cleanrl"), use_temp_dir=True)
rm_tokenizer.push_to_hub(rm_record.hf_repo_id.replace("vwxyzjn", "cleanrl"), use_temp_dir=True)
