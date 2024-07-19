from dataclasses import dataclass
import multiprocessing
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
import numpy as np
import datasets
import torch
import torch.nn as nn
from transformers import (
    DataCollatorWithPadding,
    PreTrainedTokenizer,
    PreTrainedTokenizerBase,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EvalPrediction,
    TrainerCallback,
    GenerationConfig,
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)
from transformers.trainer_utils import seed_worker
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
import torch.nn.functional as F

INVALID_LOGPROB = 1.0

class PPOConfig(TrainingArguments):
    # # various batch sizes
    # world_size: Optional[int] = None
    # """The number of processes (GPUs) to use"""
    num_updates: Optional[int] = None
    """The number of updates to train"""
    # gradient_accumulation_steps: int = 64
    # """The number of gradient accumulation steps"""
    # per_device_train_batch_size: Optional[int] = 1
    # """The micro batch size per GPU (HF's `per_device_train_batch_size`)"""
    total_episodes: Optional[int] = 1000000
    """The total number of episodes in the dataset"""
    micro_batch_size: Optional[int] = None
    """The micro batch size across devices (HF's `per_device_train_batch_size` * `world_size`)"""
    local_batch_size: Optional[int] = None
    """The batch size per GPU (HF's `per_device_train_batch_size` * `gradient_accumulation_steps`)"""
    batch_size: Optional[int] = None
    """The batch size across devices (HF's `per_device_train_batch_size` * `world_size` * `gradient_accumulation_steps`)"""
    nminibatches: int = 1
    """Number of minibatches to split a batch into"""
    local_mini_batch_size: Optional[int] = None
    """the mini batch size per GPU"""
    mini_batch_size: Optional[int] = None
    """the mini batch size across GPUs"""
    local_eval_batch_size: int = 2
    """per rank eval batch size"""
    # local_rollout_forward_batch_size: int = 64
    # """per rank no grad forward pass in the rollout phase"""

    # other args
    # base_model: str = "EleutherAI/pythia-160m"
    # """the name of the pretrained model to use"""
    # query_dataset: str = "vwxyzjn/summarize_from_feedback_tldr_3_filtered_oai_preprocessing_1706381144"
    # """the query dataset"""

    dataset_text_field: str = "prompt"
    """the name of the text field in the dataset"""
    debug: bool = False
    """whether to run in debug mode"""
    nminibatches: int = 1
    """the number of minibatches to split a batch into"""
    noptepochs: int = 4
    """the number of epochs to train"""
    vf_coef: float = 0.1
    """the value function coefficient"""
    cliprange: float = 0.2
    """the clip range"""
    cliprange_value: float = 0.2
    """the clip range for the value function"""
    gamma: float = 1
    """the discount factor"""
    lam: float = 0.95
    """the lambda value for GAE"""
    whiten_rewards: bool = False
    """whether to whiten the rewards"""
    kl_coef: float = 0.05
    """the KL coefficient"""

    # response_length: int = 53
    # """the length of the response"""
    truncate_token: Literal["eos"] = "eos"
    """the truncate token"""
    truncate_token_id: Optional[int] = None
    """the truncation token id"""
    temperature: float = 0.7
    """the sampling temperature"""
    penalty_reward_value: int = -1
    """the reward value for responses that do not contain `truncate_token_id`"""
    non_eos_penalty: bool = True
    """whether to penalize responses that do not contain `truncate_token_id`"""
    # offload: bool = False
    # """Whether to offload ref policy and reward model to CPU"""
    # reward_model_path: str = ""
    # """the path to the reward model"""
    # sft_model_path: str = "EleutherAI/pythia-160m"
    # """the path to the sft model"""


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.normal_(layer.weight, std=std)
    torch.nn.init.constant_(layer.bias, val=bias_const)
    return layer


def exact_div(a, b):
    q = a // b
    if a != q * b:
        raise ValueError(f"Inexact division: {a} / {b} = {a / b}")
    return q

# taken from https://github.com/vwxyzjn/direct-preference-optimization/blob/f8b8c0f49dc92a430bae41585f9d467d3618fe2f/utils.py#L99
def disable_dropout(model: torch.nn.Module):
    """Disable dropout in a model."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0


def first_true_indices(bools, dtype=torch.long):
    """
    Takes an N-dimensional bool tensor and returns an (N-1)-dimensional tensor of integers giving
    the position of the first True in each "row".

    Returns the length of the rows (bools.size(-1)) if no element is True in a given row.
    """
    row_len = bools.size(-1)
    zero_or_index = row_len * (~bools).type(dtype) + torch.arange(row_len, dtype=dtype, device=bools.device)
    return torch.min(zero_or_index, dim=-1).values


def truncate_response(responses, truncate_token_id, pad_token_id):
    trunc_idxs = first_true_indices(responses == truncate_token_id).unsqueeze(-1)
    new_size = [1] * (len(responses.size()) - 1) + [responses.shape[1]]
    idxs = torch.arange(responses.shape[1], device=responses.device).view(*new_size)
    postprocessed_responses = torch.masked_fill(responses, idxs > trunc_idxs, pad_token_id)
    return postprocessed_responses



def masked_mean(values, mask, axis=None):
    """Compute mean of tensor with a masked values."""
    if axis is not None:
        return (values * mask).sum(axis=axis) / mask.sum(axis=axis)
    else:
        return (values * mask).sum() / mask.sum()


def masked_var(values, mask, unbiased=True):
    """Compute variance of tensor with masked values."""
    mean = masked_mean(values, mask)
    centered_values = values - mean
    variance = masked_mean(centered_values**2, mask)
    if unbiased:
        mask_sum = mask.sum()
        if mask_sum == 0:
            raise ValueError(
                "The sum of the mask is zero, which can happen when `mini_batch_size=1`;"
                "try increase the `mini_batch_size` or `gradient_accumulation_steps`"
            )
        # note that if mask_sum == 1, then there is a division by zero issue
        # to avoid it you just need to use a larger minibatch_size
        bessel_correction = mask_sum / (mask_sum - 1)
        variance = variance * bessel_correction
    return variance


def masked_whiten(values, mask, shift_mean=True):
    """Whiten values with masked values."""
    mean, var = masked_mean(values, mask), masked_var(values, mask, False)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened


def get_reward(model, query_responses, tokenizer, context_length):
    attention_mask = query_responses != tokenizer.pad_token_id
    # position_ids = attention_mask.cumsum(1) - attention_mask.long()  # exclusive cumsum
    lm_backbone = getattr(model, model.base_model_prefix)
    input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
    output = lm_backbone(
        input_ids=input_ids,
        attention_mask=attention_mask,
        # position_ids=position_ids,
        return_dict=True,
        output_hidden_states=True,
    )
    reward_logits = model.score(output.hidden_states[-1])
    sequence_lengths = first_true_indices(query_responses[:, context_length:] == tokenizer.pad_token_id) - 1 + context_length
    # https://github.com/huggingface/transformers/blob/dc68a39c8111217683bf49a4912d0c9018bab33d/src/transformers/models/gpt2/modeling_gpt2.py#L1454
    return (
        reward_logits,
        reward_logits[
            torch.arange(reward_logits.size(0), device=reward_logits.device),
            sequence_lengths,
        ].squeeze(-1),
        sequence_lengths,
    )


# taken from https://github.com/OpenLMLab/MOSS-RLHF/blob/40b91eb2f2b71b16919addede0341d2bef70825d/ppo/ppo_trainer.py#L29
# we did this we can do a single `model = accelerator.prepare(model)`
class PolicyAndValueWrapper(nn.Module):
    def __init__(self, policy, value_model) -> None:
        super().__init__()
        self.policy = policy
        self.value_model = value_model

    def forward(self, **kwargs):
        return self.policy(**kwargs), self.value_model(**kwargs)
    

class PolicyAndSharedValueWrapper(nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy
        self.scalar_head = layer_init(
            nn.Linear(policy.config.hidden_size, 1),
            std=1 / np.sqrt(policy.config.hidden_size + 1),
        )

    def forward(self, **kwargs):
        output = self.policy(**kwargs)
        return output, self.scalar_head(output.hidden_states[-1])


class PPOTrainer(Trainer):
    def __init__(
            self,
            args: PPOConfig,
            tokenizer: PreTrainedTokenizer,
            policy: nn.Module,
            ref_policy: nn.Module,
            reward_model: nn.Module,
            train_dataset: Dataset,
            train_generation_config: GenerationConfig,
            value_model: Optional[nn.Module] = None,
            data_collator: Optional[DataCollatorWithPadding] = None,
            eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
            eval_generation_config: Optional[GenerationConfig] = None,
            compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
            # less commonly used
            model_init: Optional[Callable[[torch.nn.Module], None]] = None,
            callbacks: Optional[List[TrainerCallback]] = None,
            optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
            preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        ) -> None:
        self.args = args
        self.tokenizer = tokenizer
        self.policy = policy
        self.ref_policy = ref_policy
        self.reward_model = reward_model
        self.train_generation_config = train_generation_config
        self.value_model = value_model
        for module in [policy, ref_policy, reward_model]:
            disable_dropout(module)
        if self.value_model is None:
            self.model = PolicyAndSharedValueWrapper(policy)
            self.separate_value_network = False
        else:
            disable_dropout(value_model)
            self.model = PolicyAndValueWrapper(policy, value_model)
            self.separate_value_network = True
        self.eval_generation_config = eval_generation_config
        if eval_generation_config is None:
            self.eval_generation_config = train_generation_config

        # super().__init__(
        #     args=args,
        #     model= self.model,
        #     data_collator=data_collator,
        #     train_dataset=self.prepare_dataset(train_dataset, tokenizer),
        #     eval_dataset=self.prepare_dataset(eval_dataset, tokenizer),
        #     tokenizer=tokenizer,
        #     model_init=model_init,
        #     compute_metrics=compute_metrics,
        #     callbacks=callbacks,
        #     optimizers=optimizers,
        #     preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        # )
        self.optimizer, self.lr_scheduler = optimizers
        self.preprocess_logits_for_metrics = preprocess_logits_for_metrics
        self.callbacks = callbacks
        self.model_init = model_init

        self.create_accelerator_and_postprocess()
        self.train_dataset = self.prepare_dataset(train_dataset, tokenizer)
        self.eval_dataset = self.prepare_dataset(eval_dataset, tokenizer)
        self.data_collator = data_collator
        self.model = self.accelerator.prepare(self.model)

        self.device = self.model.policy.device
        self.ref_policy = self.ref_policy.to(self.device)
        self.reward_model = self.reward_model.to(self.device)

        # runtime batch sizes
        args.local_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps * args.nminibatches
        args.micro_batch_size = int(args.per_device_train_batch_size * args.world_size)
        args.batch_size = int(args.local_batch_size * args.world_size)
        args.mini_batch_size = exact_div(args.batch_size, args.nminibatches)
        args.local_mini_batch_size = exact_div(args.local_batch_size, args.nminibatches)
        args.num_updates = args.total_episodes // args.batch_size


    def prepare_dataset(self, dataset, tokenizer):
        """pre-tokenize the dataset before training; only collate during training"""
        def tokenize(element):
            outputs = tokenizer(
                element[self.args.dataset_text_field],
                padding=False,
            )
            return {"input_ids": outputs["input_ids"], "attention_mask": outputs["attention_mask"]}
        return dataset.map(
            tokenize,
            remove_columns=dataset.column_names,
            batched=True,
            num_proc=1 if self.args.debug else 4 #multiprocessing.cpu_count(),
        )

    def get_train_dataloader(self) -> DataLoader:
        """
        Costa: it's removing the dataset altogether in the `get_train_dataloader` function, so overriding it
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator

        dataloader_params = {
            "batch_size": self.args.local_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    @staticmethod
    def generate(model, input_ids, attention_mask, generation_config):
        """generate in a way that does not affect padding tokens"""
        context_length = input_ids.shape[1]
        input_ids = torch.masked_fill(input_ids, ~attention_mask.bool(), 0)
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # position_ids=attention_mask.cumsum(1) - attention_mask.long(), # generation collapsed if this was turned on. TODO: why does generation collapse with this?
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True
        )
        logits = torch.stack(output.scores, 1)
        return torch.cat((input_ids, output.sequences[:, context_length:]), dim=1), logits

    @staticmethod
    def forward(model, query_responses, tokenizer):
        attention_mask = query_responses != tokenizer.pad_token_id
        # position_ids = attention_mask.cumsum(1) - attention_mask.long()
        input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
        return model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # position_ids=position_ids,
            return_dict=True,
            output_hidden_states=True,
        )


    def train(self):
        model = self.model
        args = self.args
        stats_shape = (args.noptepochs, args.nminibatches, args.gradient_accumulation_steps)
        approxkl_stats = torch.zeros(stats_shape, device=self.device)
        pg_clipfrac_stats = torch.zeros(stats_shape, device=self.device)
        pg_loss_stats = torch.zeros(stats_shape, device=self.device)
        vf_loss_stats = torch.zeros(stats_shape, device=self.device)
        vf_clipfrac_stats = torch.zeros(stats_shape, device=self.device)
        entropy_stats = torch.zeros(stats_shape, device=self.device)
        ratio_stats = torch.zeros(stats_shape, device=self.device)
        train_dataloader = self.get_train_dataloader()
        iter_dataloader = iter(train_dataloader)
        self.create_optimizer_and_scheduler(num_training_steps=args.num_updates)
        model.train()
        for update in range(1, args.num_updates + 1):
            inputs = next(iter_dataloader)
            with torch.no_grad():
                queries, attention_masks = inputs["input_ids"], inputs["attention_mask"]
                context_length = queries.shape[1]
                query_responses = []
                responses = []
                postprocessed_responses = []
                logprobs = []
                ref_logprobs = []
                values = []
                scores = []
                sequence_lengths = []
                for i in range(0, queries.shape[0], args.per_device_train_batch_size):
                    query = queries[i : i + args.per_device_train_batch_size]
                    attention_mask = attention_masks[i : i + args.per_device_train_batch_size]
                    query_response, logits = self.generate(self.accelerator.unwrap_model(model).policy, query, attention_mask, self.train_generation_config)
                    response = query_response[:, context_length:]

                    # use the logits during generation directly, instead of using the following
                    # all_logprob = F.log_softmax(logits, dim=-1)
                    # logprob = torch.gather(all_logprob, 2, response.unsqueeze(-1)).squeeze(-1)
                    output = self.forward(self.policy, query_response, self.tokenizer)
                    logits = output.logits[:, context_length - 1 : -1]
                    logits /= args.temperature + 1e-7
                    all_logprob = F.log_softmax(logits, dim=-1)
                    logprob = torch.gather(all_logprob, 2, response.unsqueeze(-1)).squeeze(-1)
                    del logits, all_logprob
                    torch.cuda.empty_cache()

                    ref_output = self.forward(self.ref_policy, query_response, self.tokenizer)
                    ref_logits = ref_output.logits[:, context_length - 1 : -1]
                    ref_logits /= args.temperature + 1e-7
                    ref_all_logprob = F.log_softmax(ref_logits, dim=-1)
                    ref_logprob = torch.gather(ref_all_logprob, 2, response.unsqueeze(-1)).squeeze(-1)
                    del ref_output, ref_logits, ref_all_logprob
                    torch.cuda.empty_cache()

                    # Response Processing 1. truncate response after the first occurrence of `truncate_token_id`
                    postprocessed_response = response
                    if args.truncate_token_id:
                        postprocessed_response = truncate_response(response, args.truncate_token_id, self.tokenizer.pad_token_id)
                        

                    # Response Processing 2. run reward model on the truncated responses
                    postprocessed_query_response = torch.cat((query, postprocessed_response), 1)
                    sequence_length = first_true_indices(postprocessed_response == self.tokenizer.pad_token_id) - 1
                    if self.separate_value_network:
                        full_value, _, _ = get_reward(
                            self.accelerator.unwrap_model(model).critic, query_response, self.tokenizer, context_length
                        )
                    else:
                        _, full_value = self.forward(model, query_response, self.tokenizer)
                    value = full_value[:, context_length - 1 : -1].squeeze(-1)
                    _, score, _ = get_reward(self.reward_model, postprocessed_query_response, self.tokenizer, context_length)

                    query_responses.append(query_response)
                    responses.append(response)
                    postprocessed_responses.append(postprocessed_response)
                    logprobs.append(logprob)
                    ref_logprobs.append(ref_logprob)
                    values.append(value)
                    sequence_lengths.append(sequence_length)
                    scores.append(score)
                query_responses = torch.cat(query_responses, 0)
                responses = torch.cat(responses, 0)
                postprocessed_responses = torch.cat(postprocessed_responses, 0)
                logprobs = torch.cat(logprobs, 0)
                ref_logprobs = torch.cat(ref_logprobs, 0)
                values = torch.cat(values, 0)
                sequence_lengths = torch.cat(sequence_lengths, 0)
                scores = torch.cat(scores, 0)
                del (logprob, ref_logprob, full_value, value, score)
                torch.cuda.empty_cache()

                # Response Processing 3. filter response. Ensure that the sample contains truncate_token_id
                # responses not passing that filter will receive a low (fixed) score
                # only query humans on responses that pass that filter
                contain_eos_token = torch.any(postprocessed_responses == self.tokenizer.eos_token_id, dim=-1)
                if args.non_eos_penalty:
                    scores = torch.where(contain_eos_token, scores, torch.full_like(scores, args.penalty_reward_value))

                # be very careful with `padding_mask_p1`; see https://excalidraw.com/#json=LWnzG4w2k5DjF_EOL_xPt,e2w3a-hFJ_gX5vOfeyXGTw
                sequence_lengths_p1 = sequence_lengths + 1
                response_idxs = torch.arange(responses.shape[1], device=responses.device).repeat(responses.shape[0], 1)
                padding_mask = response_idxs > sequence_lengths.unsqueeze(1)
                padding_mask_p1 = response_idxs > (sequence_lengths_p1.unsqueeze(1))
                logprobs = torch.masked_fill(logprobs, padding_mask, INVALID_LOGPROB)
                ref_logprobs = torch.masked_fill(ref_logprobs, padding_mask, INVALID_LOGPROB)
                values = torch.masked_fill(values, padding_mask_p1, 0)

                # 4. compute rewards
                kl = logprobs - ref_logprobs
                non_score_reward = -args.kl_coef * kl
                rewards = non_score_reward.clone()
                actual_start = torch.arange(rewards.size(0), device=rewards.device)
                actual_end = torch.where(sequence_lengths_p1 < rewards.size(1), sequence_lengths_p1, sequence_lengths)
                rewards[[actual_start, actual_end]] += scores

                # 5. whiten rewards
                if args.whiten_rewards:
                    rewards = masked_whiten(rewards, mask=~padding_mask_p1, shift_mean=False)
                    rewards = torch.masked_fill(rewards, padding_mask_p1, 0)

                # 6. compute advantages and returns
                lastgaelam = 0
                advantages_reversed = []
                gen_length = responses.shape[1]
                for t in reversed(range(gen_length)):
                    nextvalues = values[:, t + 1] if t < gen_length - 1 else 0.0
                    delta = rewards[:, t] + args.gamma * nextvalues - values[:, t]
                    lastgaelam = delta + args.gamma * args.lam * lastgaelam
                    advantages_reversed.append(lastgaelam)
                advantages = torch.stack(advantages_reversed[::-1], axis=1)
                returns = advantages + values
                advantages = masked_whiten(advantages, ~padding_mask)
                advantages = torch.masked_fill(advantages, padding_mask, 0)
                torch.cuda.empty_cache()

            # Do multiple epochs of PPO training, with a fresh random shuffle in each epoch
            for ppo_epoch_idx in range(args.noptepochs):
                b_inds = np.random.permutation(args.local_batch_size)
                minibatch_idx = 0
                for mini_batch_start in range(0, args.local_batch_size, args.local_mini_batch_size):
                    mini_batch_end = mini_batch_start + args.local_mini_batch_size
                    mini_batch_inds = b_inds[mini_batch_start:mini_batch_end]
                    gradient_accumulation_idx = 0
                    for micro_batch_start in range(0, args.local_mini_batch_size, args.per_device_train_batch_size):
                        with self.accelerator.accumulate(model):
                            micro_batch_end = micro_batch_start + args.per_device_train_batch_size
                            micro_batch_inds = mini_batch_inds[micro_batch_start:micro_batch_end]
                            mb_return = returns[micro_batch_inds]
                            mb_advantage = advantages[micro_batch_inds]
                            mb_values = values[micro_batch_inds]
                            mb_responses = responses[micro_batch_inds]
                            mb_query_responses = query_responses[micro_batch_inds]
                            mb_logprobs = logprobs[micro_batch_inds]

                            output, vpred_temp = self.forward(model, mb_query_responses, self.tokenizer)
                            logits = output.logits[:, context_length - 1 : -1]
                            logits /= args.temperature + 1e-7
                            new_all_logprobs = F.log_softmax(logits, dim=-1)
                            new_logprobs = torch.gather(new_all_logprobs, 2, mb_responses.unsqueeze(-1)).squeeze(-1)
                            new_logprobs = torch.masked_fill(new_logprobs, padding_mask[micro_batch_inds], INVALID_LOGPROB)
                            vpred = vpred_temp[:, context_length - 1 : -1].squeeze(-1)
                            vpred = torch.masked_fill(vpred, padding_mask_p1[micro_batch_inds], 0)
                            vpredclipped = torch.clamp(
                                vpred,
                                mb_values - args.cliprange_value,
                                mb_values + args.cliprange_value,
                            )
                            vf_losses1 = torch.square(vpred - mb_return)
                            vf_losses2 = torch.square(vpredclipped - mb_return)
                            vf_loss_max = torch.max(vf_losses1, vf_losses2)
                            vf_loss = 0.5 * masked_mean(vf_loss_max, ~padding_mask_p1[micro_batch_inds])
                            vf_clipfrac = masked_mean((vf_losses2 > vf_losses1).float(), ~padding_mask_p1[micro_batch_inds])
                            logprobs_diff = new_logprobs - mb_logprobs
                            ratio = torch.exp(logprobs_diff)
                            pg_losses = -mb_advantage * ratio
                            pg_losses2 = -mb_advantage * torch.clamp(ratio, 1.0 - args.cliprange, 1.0 + args.cliprange)
                            pg_loss_max = torch.max(pg_losses, pg_losses2)
                            pg_loss = masked_mean(pg_loss_max, ~padding_mask[micro_batch_inds])
                            pg_clipfrac = masked_mean((pg_losses2 > pg_losses).float(), ~padding_mask[micro_batch_inds])
                            loss = pg_loss + args.vf_coef * vf_loss
                            self.accelerator.backward(loss)
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            with torch.no_grad():
                                pg_clipfrac = (pg_losses2 > pg_losses).float().mean()
                                prob_dist = torch.nn.functional.softmax(logits, dim=-1)
                                entropy = torch.logsumexp(logits, dim=-1) - torch.sum(prob_dist * logits, dim=-1)
                                approxkl = 0.5 * (logprobs_diff**2).mean()
                                approxkl_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = approxkl
                                pg_clipfrac_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = pg_clipfrac
                                pg_loss_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = pg_loss
                                vf_loss_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = vf_loss
                                vf_clipfrac_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = vf_clipfrac
                                entropy_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = entropy.mean()
                                ratio_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = ratio.mean()
                        gradient_accumulation_idx += 1
                    minibatch_idx += 1
                    # del everything and empty cache
                    # fmt: off
                    del (
                        output, vpred_temp, logits, new_all_logprobs, new_logprobs, vpred, vpredclipped,
                        vf_losses1, vf_losses2, vf_loss, vf_clipfrac, logprobs_diff, ratio, pg_losses, pg_losses2,
                        pg_loss, loss, pg_clipfrac, prob_dist, entropy, approxkl, mb_return,
                        mb_advantage, mb_values, mb_responses, mb_query_responses, mb_logprobs,
                    )
                    # fmt: on
                    torch.cuda.empty_cache()
            
            if update > 1 and (update - 1) % args.gradient_accumulation_steps == 0:
                self.lr_scheduler.step()

            if self.accelerator.is_main_process:
                print(
                    "ppo_epoch_idx",
                    ppo_epoch_idx,
                    "approxkl",
                    approxkl_stats[: ppo_epoch_idx + 1].mean().item(),
                    "pg_loss",
                    pg_loss_stats[: ppo_epoch_idx + 1].mean().item(),
                    "pg_clipfrac",
                    pg_clipfrac_stats[: ppo_epoch_idx + 1].mean().item(),
                    "ratio",
                    ratio_stats[: ppo_epoch_idx + 1].mean().item(),
                )
            with torch.no_grad():
                mean_kl = kl.sum(1).mean()
                mean_entropy = (-logprobs).sum(1).mean()
                mean_non_score_reward = non_score_reward.sum(1).mean()
                metrics = {}
                metrics["objective/kl"] = self.accelerator.gather(mean_kl).mean().item()
                metrics["objective/entropy"] = self.accelerator.gather(mean_entropy).mean().item()
                metrics["objective/non_score_reward"] = self.accelerator.gather(mean_non_score_reward).mean().item()
                metrics["objective/score_total"] = self.accelerator.gather(mean_non_score_reward + scores.mean()).mean().item()
                metrics["objective/scores"] = self.accelerator.gather(scores.mean()).mean().item()
                metrics["ppo/policy/approxkl_avg"] = self.accelerator.gather(approxkl_stats).mean().item()
                metrics["ppo/policy/clipfrac_avg"] = self.accelerator.gather(pg_clipfrac_stats).mean().item()
                metrics["ppo/loss/policy_avg"] = self.accelerator.gather(pg_loss_stats).mean().item()
                metrics["ppo/loss/value_avg"] = self.accelerator.gather(vf_loss_stats).mean().item()
                metrics["ppo/val/clipfrac_avg"] = self.accelerator.gather(vf_clipfrac_stats).mean().item()
                metrics["ppo/policy/entropy_avg"] = self.accelerator.gather(entropy_stats).mean().item()
                metrics["ppo/val/ratio"] = self.accelerator.gather(ratio_stats).mean().item()
                metrics["ppo/val/ratio_var"] = self.accelerator.gather(ratio_stats).var().item()
                print(metrics)
                self.log(metrics)
            del kl, mean_kl, mean_entropy, mean_non_score_reward, scores
            torch.cuda.empty_cache()


if __name__ == "__main__":
    # create a PPOTrainer
    base_model = "EleutherAI/pythia-1b-deduped"
    model_config = AutoConfig.from_pretrained(base_model)
    left_tokenizer = AutoTokenizer.from_pretrained(base_model, padding_side="left") # for generation
    left_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if left_tokenizer.chat_template is None:
        # a default chat template to simply concatenate the messages
        left_tokenizer.chat_template = "{% for message in messages %}{{' ' + message['content']}}{% endfor %}{{ eos_token }}"
    response_length = 53
    train_generation_config = GenerationConfig(
        max_new_tokens=response_length,
        pad_token_id=left_tokenizer.pad_token_id,
        eos_token_id=left_tokenizer.eos_token_id,
        temperature=(0.7 + 1e-7),
        top_k=0.0,
        top_p=1.0,
        do_sample=True,
    )
    from datasets import load_dataset

    raw_datasets = load_dataset("trl-internal-testing/descriptiveness-sentiment-trl-style", split="descriptiveness")
    def process(row):
        row["chosen"] = left_tokenizer.apply_chat_template(row["chosen"], tokenize=False).strip()
        row["rejected"] = left_tokenizer.apply_chat_template(row["rejected"], tokenize=False).strip()
        return row
    raw_datasets = raw_datasets.map(process, load_from_cache_file=False)
    eval_samples = 20
    train_dataset = raw_datasets.select(range(len(raw_datasets) - eval_samples))
    eval_dataset = raw_datasets.select(range(len(raw_datasets) - eval_samples, len(raw_datasets)))
    trainer = PPOTrainer(
        args=PPOConfig(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=64,
            learning_rate=3e-6,
            logging_steps=1,
            evaluation_strategy="epoch",
            num_train_epochs=1,
            output_dir="minimal/ppo",
            report_to=None,
        ),
        policy=AutoModelForCausalLM.from_pretrained(base_model),
        ref_policy=AutoModelForCausalLM.from_pretrained(base_model),
        reward_model=AutoModelForSequenceClassification.from_pretrained("minimal/reward", num_labels=1),
        tokenizer=left_tokenizer,
        train_dataset=train_dataset,
        train_generation_config=train_generation_config,
        data_collator=DataCollatorWithPadding(left_tokenizer),
        eval_dataset=eval_dataset,
    )
    trainer.train()




    # d = trainer.accelerator.prepare(DataLoader(trainer.train_dataset, batch_size=5, collate_fn=DataCollatorWithPadding(left_tokenizer)))
    # d = trainer.accelerator.prepare(
    #     DataLoader(
    #         trainer.train_dataset, 
    #         batch_size=5, 
    #         collate_fn=DataCollatorWithPadding(left_tokenizer)
    # ))


    # train_dataset = trainer.train_dataset
    # if isinstance(trainer.train_dataset, datasets.Dataset):
    #     train_dataset = trainer._remove_unused_columns(train_dataset, description="training")
    # else:
    #     data_collator = trainer._get_collator_with_removed_columns(trainer.data_collator, description="training")

    # dataloader_params = {
    #     "batch_size": trainer._train_batch_size,
    #     "collate_fn": data_collator,
    #     "num_workers": trainer.args.dataloader_num_workers,
    #     "pin_memory": trainer.args.dataloader_pin_memory,
    #     "persistent_workers": trainer.args.dataloader_persistent_workers,
    # }

    # if not isinstance(train_dataset, torch.utils.data.IterableDataset):
    #     dataloader_params["sampler"] = trainer._get_train_sampler()
    #     dataloader_params["drop_last"] = trainer.args.dataloader_drop_last
    #     # dataloader_params["worker_init_fn"] = seed_worker
    #     dataloader_params["prefetch_factor"] = trainer.args.dataloader_prefetch_factor
    # d = trainer.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))
    # print("haha")

# trainer.train
# trainer.log_metrics
# trainer.save_metrics
# trainer.save_state
# trainer.evaluate
# trainer.log_metrics
# trainer.save_metrics
# trainer.save_model
# trainer.create_model_card
# trainer.model
# trainer.model