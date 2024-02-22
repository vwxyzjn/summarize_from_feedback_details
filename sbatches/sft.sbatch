#!/bin/bash
#SBATCH --partition=hopper-prod
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-gpu=10
#SBATCH --ntasks=1
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --requeue
#SBATCH --array=0-11 # %25
#SBATCH --exclusive

module load cuda/12.2
export WANDB_TAGS=refactor-chosen-rejected3,no-tag-$(git rev-parse --short HEAD)
MODELS=("EleutherAI/pythia-6.9b-deduped" "EleutherAI/pythia-2.8b-deduped" "EleutherAI/pythia-1b-deduped")
SEEDS=(44413 55513 66613 77713)
MODEL_INDEX=$((SLURM_ARRAY_TASK_ID / 4))
SEED_INDEX=$((SLURM_ARRAY_TASK_ID % 4))
MODEL=${MODELS[$MODEL_INDEX]}
SEED=${SEEDS[$SEED_INDEX]}

echo "Running task $SLURM_ARRAY_TASK_ID with SEED: $SEED and MODEL: $MODEL"

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
DPO_POLICY_MODEL_PATH=models/$MODEL/dpo_policy_model_$SEED

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
    local_eval_batch_size=1
fi
if [ "$MODEL" = "EleutherAI/pythia-6.9b-deduped" ]; then
    local_rollout_forward_batch_size=2
    gradient_accumulation_steps=64
    local_micro_batch_size=1
    local_eval_batch_size=1
fi

srun poetry run accelerate launch --config_file deepspeed.yaml \
    summarize_from_feedback_details/sft.py \
    --base_model=$MODEL \
    --lr=$LR \
    --deepspeed \
    --track \
    --output_dir=$SFT_MODEL_PATH \
    --push_to_hub \
    --run_eval \
    --seed=$SEED
 