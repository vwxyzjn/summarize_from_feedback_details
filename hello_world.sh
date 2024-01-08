SEED=1
MODEL=EleutherAI/pythia-410m-deduped
LR=3e-6
REWARD_MODEL_PATH=models/$MODEL/reward_model_$SEED
SFT_MODEL_PATH=models/$MODEL/sft_model_$SEED
POLICY_MODEL_PATH=models/$MODEL/policy_model_$SEED

poetry run accelerate launch --config_file deepspeed.yaml \
    summarize_from_feedback_details/summarize/sft.py \
    --task.query_dataset=cleanrl/summarize_from_feedback_tldr_3_filtered_oai_preprocessing_1704563162 \
    --base_model=$MODEL \
    --lr=$LR \
    --deepspeed \
    --track \
    --output_dir=$SFT_MODEL_PATH \
    --push_to_hub \
    --run_eval \
    --seed=$SEED
 
poetry run accelerate launch --config_file deepspeed.yaml \
    summarize_from_feedback_details/summarize/reward.py \
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
poetry run accelerate launch --config_file deepspeed.yaml \
    summarize_from_feedback_details/summarize/ppo_left_padding.py \
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

 