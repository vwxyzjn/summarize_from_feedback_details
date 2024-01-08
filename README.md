# summarize_from_feedback_details

The follow-up work of https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo

STATUS: WIP (work in progress); some experiments are still running.

Prerequisites:
* A slurm cluster of 8xH100 box (unless we add LoRA)


## Get started

Install the dependencies

```
# with poetry (recommended)
poetry install
# or with pip
pip install -r requirements.txt
```

To run a hello world example, you can run the `hello_world.sh` script. For the full-scaling laws experiment, you can run 

```
sbatch benchmark.sbatch
```

It uses our pre-built TL;DR datasets:

* SFT dataset: [cleanrl/summarize_from_feedback_tldr_3_filtered_oai_preprocessing_1704563162](https://huggingface.co/datasets/cleanrl/summarize_from_feedback_tldr_3_filtered_oai_preprocessing_1704563162)
* Preference dataset: [cleanrl/summarize_from_feedback_oai_preprocessing_1704563162](https://huggingface.co/datasets/cleanrl/summarize_from_feedback_oai_preprocessing_1704563162)



### (Optionally) You can recreate the dataset


```bash
poetry run python summarize_from_feedback_details/tldr_dataset.py \
    --base_model=EleutherAI/pythia-1b-deduped \
    --tldr_params.max_sft_response_length=53 \
    --tldr_params.max_sft_query_response_length=562 \
    --tldr_params.max_rm_response_length=169 \
    --tldr_params.max_rm_query_response_length=638 \
    --cnndm_params.max_rm_response_length=155 \
    --cnndm_params.max_rm_query_response_length=2021 \
    --tldr_params.padding="pad_token" \
    --cnndm_params.padding="pad_token"
    # --push_to_hub # you can optionally push to hub
```

Note that these datasets use the same OpenAI processing as the original paper ([summarize-from-feedback/tasks.py#L98-L165](https://github.com/openai/summarize-from-feedback/blob/700967448d10004279f138666442bf1497d0e705/summarize_from_feedback/tasks.py#L98-L165)); it does things like

* make sure query is only 512 tokens (pad if shorter, and ''smartly truncate'' if longer, e.g., like it will truncate at before the last `\n` instead of a hard truncation.)
* make sure response tokens is limited