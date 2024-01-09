#!/bin/bash
#SBATCH --partition=hopper-prod
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-gpu=10
#SBATCH --ntasks=1
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --requeue
#SBATCH --array=0-3 # %25
#SBATCH --exclusive

export WANDB_TAGS=refactor111,no-tag-$(git rev-parse --short HEAD)

MODELS=("EleutherAI/pythia-6.9b-deduped")
SEEDS=(44413 55513 66613 77713)
MODEL_INDEX=$((SLURM_ARRAY_TASK_ID / 4))
SEED_INDEX=$((SLURM_ARRAY_TASK_ID % 4))
MODEL=${MODELS[$MODEL_INDEX]}
SEED=${SEEDS[$SEED_INDEX]}

echo "Running task $SLURM_ARRAY_TASK_ID with SEED: $SEED and MODEL: $MODEL"

# module load cuda/12.2 

if [ -z "$SEED" ]; then
    SEED=1
fi
if [ -z "$MODEL" ]; then
    # MODEL=EleutherAI/pythia-6.9b-deduped
    MODEL=EleutherAI/pythia-2.8b-deduped
    # MODEL=EleutherAI/pythia-1b-deduped
    # MODEL=EleutherAI/pythia-410m-deduped
fi
if [ -z "$LR" ]; then
    LR=3e-6
fi

REWARD_MODEL_PATH=models/$MODEL/reward_model_$SEED
SFT_MODEL_PATH=models/$MODEL/sft_model_$SEED
POLICY_MODEL_PATH=models/$MODEL/policy_model_$SEED

if [ "$MODEL" = "EleutherAI/pythia-1b-deduped" ]; then
    local_rollout_forward_batch_size=64
    gradient_accumulation_steps=4
    local_micro_batch_size=16
    local_eval_batch_size=8
fi
if [ "$MODEL" = "EleutherAI/pythia-2.8b-deduped" ]; then
    local_rollout_forward_batch_size=32
    gradient_accumulation_steps=16
    local_micro_batch_size=4
    local_eval_batch_size=8
fi
if [ "$MODEL" = "EleutherAI/pythia-6.9b-deduped" ]; then
    local_rollout_forward_batch_size=2
    gradient_accumulation_steps=64
    local_micro_batch_size=1
    local_eval_batch_size=1
fi

srun poetry run accelerate launch --config_file deepspeed.yaml \
    summarize_from_feedback_details/sft.py \
    --task.query_dataset=cleanrl/summarize_from_feedback_tldr_3_filtered_oai_preprocessing_1704563162 \
    --base_model=$MODEL \
    --lr=$LR \
    --deepspeed \
    --track \
    --output_dir=$SFT_MODEL_PATH \
    --push_to_hub \
    --run_eval \
    --seed=$SEED
 
srun poetry run accelerate launch --config_file deepspeed.yaml \
    summarize_from_feedback_details/reward.py \
    --label_dataset=cleanrl/summarize_from_feedback_oai_preprocessing_1704563162 \
    --base_model=$SFT_MODEL_PATH \
    --lr=$LR \
    --deepspeed \
    --run_eval \
    --track \
    --output_dir=$REWARD_MODEL_PATH \
    --push_to_hub \
    --local_eval_batch_size=$local_eval_batch_size \
    --seed=$SEED

# proper left padding
srun poetry run accelerate launch --config_file deepspeed.yaml \
    summarize_from_feedback_details/ppo_left_padding.py \
    --task.query_dataset=cleanrl/summarize_from_feedback_tldr_3_filtered_oai_preprocessing_1704563162 \
    --local_rollout_forward_batch_size=$local_rollout_forward_batch_size \
    --gradient_accumulation_steps=$gradient_accumulation_steps \
    --local_micro_batch_size=$local_micro_batch_size \
    --base_model=$MODEL \
    --sft_model_path=$SFT_MODEL_PATH \
    --reward_model_path=$REWARD_MODEL_PATH \
    --lr=$LR \
    --deepspeed \
    --run_eval \
    --push_to_hub \
    --track \
    --output_dir=$POLICY_MODEL_PATH \
    --seed=$SEED
 