SEED=55513
if [ -z "$MODEL" ]; then
    # MODEL=EleutherAI/pythia-6.9b-deduped
    # MODEL=EleutherAI/pythia-2.8b-deduped
    MODEL=EleutherAI/pythia-1b-deduped
    # MODEL=EleutherAI/pythia-410m-deduped
fi
LR=3e-6
REWARD_MODEL_PATH=vwxyzjn/EleutherAI_pythia-1b-deduped__reward__tldr
REWARD_MODEL_REVISION=reward__${SEED}__1708628552
SFT_MODEL_PATH=vwxyzjn/EleutherAI_pythia-1b-deduped__sft__tldr
SFT_MODEL_REVISION=sft__${SEED}__1708611267
POLICY_MODEL_PATH=models/$MODEL/policy_model_$SEED

# vary the following parameters to fit your GPU memory
local_rollout_forward_batch_size=2 # smaller fits better on GPU
gradient_accumulation_steps=64 # bigger fits better on GPU
local_micro_batch_size=1 # smaller fits better on GPU
local_eval_batch_size=1 # smaller fits better on GPU

# 1. you want to make sure gradient_accumulation_steps * local_micro_batch_size = 64
# so you have the same hyperparameters as the paper
# 2. if you are running on a single GPU, you want to make sure 
# gradient_accumulation_steps * local_micro_batch_size = 512 to have the same hyperparameters
#
# poetry run accelerate launch --config_file deepspeed.yaml \
#     summarize_from_feedback_details/sft.py \
#     --task.query_dataset=cleanrl/summarize_from_feedback_tldr_3_filtered_oai_preprocessing_1704563162 \
#     --base_model=$MODEL \
#     --lr=$LR \
#     --deepspeed \
#     --track \
#     --output_dir=$SFT_MODEL_PATH \
#     --push_to_hub \
#     --run_eval \
#     --seed=$SEED
#  
# poetry run accelerate launch --config_file deepspeed.yaml \
#     summarize_from_feedback_details/reward.py \
#     --label_dataset=cleanrl/summarize_from_feedback_oai_preprocessing_1704563162 \
#     --base_model=$SFT_MODEL_PATH \
#     --lr=$LR \
#     --deepspeed \
#     --run_eval \
#     --track \
#     --output_dir=$REWARD_MODEL_PATH \
#     --push_to_hub \
#     --local_eval_batch_size=$local_eval_batch_size \
#     --seed=$SEED

# proper left padding
poetry run accelerate launch --config_file deepspeed_1gpu.yaml \
    summarize_from_feedback_details/ppo_left_padding_new_lora.py \
    --ppo.no_whiten_rewards \
    --query_dataset=cleanrl/summarize_from_feedback_tldr_3_filtered_oai_preprocessing_1704563162 \
    --local_rollout_forward_batch_size=$local_rollout_forward_batch_size \
    --gradient_accumulation_steps=$gradient_accumulation_steps \
    --local_micro_batch_size=$local_micro_batch_size \
    --base_model=$MODEL \
    --sft_model_path=$SFT_MODEL_PATH \
    --sft_model_revision=$SFT_MODEL_REVISION \
    --reward_model_path=$REWARD_MODEL_PATH \
    --reward_model_revision=$REWARD_MODEL_REVISION \
    --lr=$LR \
    --deepspeed \
    --run_eval \
    --push_to_hub \
    --track \
    --output_dir=$POLICY_MODEL_PATH \
    --seed=$SEED

 
