if [ -z "$MODEL" ]; then
    # MODEL=EleutherAI/pythia-6.9b-deduped
    # MODEL=EleutherAI/pythia-2.8b-deduped
    MODEL=EleutherAI/pythia-1b-deduped
    # MODEL=EleutherAI/pythia-410m-deduped
fi
LR=3e-6
SEED=55513
REWARD_MODEL_PATH=vwxyzjn/EleutherAI_pythia-1b-deduped__reward__tldr
REWARD_MODEL_REVISION=reward__${SEED}__1708628552
SFT_MODEL_PATH=vwxyzjn/EleutherAI_pythia-1b-deduped__sft__tldr
SFT_MODEL_REVISION=sft__${SEED}__1708611267
OUTPUT_PATH=/home/toolkit/summarize_from_feedback_details/models/$MODEL/policy_model_$SEED

DEBUG=${DEBUG:-false}
TRACK_ARG=$([ "$DEBUG" = false ] && echo "--track" || echo "")

FP16=${FP16:-false}
NGPUS=${NGPUS:-4}

if [ "$FP16" = true ]; then
    DS_CONFIG=deepspeed_4gpu_fp16.yaml
    local_rollout_forward_batch_size=32 # smaller fits better on GPU
    gradient_accumulation_steps=64 # bigger fits better on GPU
    local_micro_batch_size=8 # smaller fits better on GPU
    local_eval_batch_size=1 # smaller fits better on GPU
else
    DS_CONFIG=deepspeed_1gpu.yaml
    gradient_accumulation_steps=4 # bigger fits better on GPU
    local_micro_batch_size=16 # smaller fits better on GPU
    local_eval_batch_size=8 # smaller fits better on GPU
fi

# dpo
python -m poetry run accelerate launch --config_file $DS_CONFIG \
    summarize_from_feedback_details/dpo_lora.py \
    --gradient_accumulation_steps=$gradient_accumulation_steps \
    --local_micro_batch_size=$local_micro_batch_size \
    --base_model=$MODEL \
    --sft_model_path=$SFT_MODEL_PATH \
    --sft_model_revision=$SFT_MODEL_REVISION \
    --lr=$LR \
    --deepspeed \
    --run_eval \
    --push_to_hub \
    --wandb_entity="mnoukhov" \
    --output_dir=$OUTPUT_PATH \
    --seed=$SEED $TRACK_ARG
