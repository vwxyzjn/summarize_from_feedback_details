import json
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

import json
import csv
from transformers import AutoTokenizer
import seaborn as sns


seeds = [44413, 55513, 66613, 77713]
base_models = ["EleutherAI/pythia-1b-deduped", "EleutherAI/pythia-2.8b-deduped", "EleutherAI/pythia-6.9b-deduped"]

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-1b-deduped")

length_by = "char"  # "token" or "char"
nrows, ncols = len(seeds), len(base_models)  # For a total of 12 subplots
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 15))  # Adjust the figure size as necessary

for seed_idx, seed in enumerate(seeds):
    for base_model_idx, base_model in enumerate(base_models):

        data_dict = {
            "PPO": {
                "file_name": f"sampled_data/refactor-chosen-rejected3/ppo_left_padding_new_nowhiten_reward/{base_model}/{seed}/query_responses_gpt-3.5-turbo-0125.csv",
                "legend": "PPO",
                "color": "red",
            },
            # "PPO w/ whiten rewards": {
            #     "file_name": f"sampled_data/refactor-chosen-rejected3/ppo_left_padding_new/{base_model}/{seed}/query_responses_gpt-3.5-turbo-0125.csv",
            #     "legend": "PPO w/ whiten rewards",
            #     "color": "blue",
            # },
            "PPO OpenAI checkpoint": {
                "file_name": f"sampled_data/openai_ppo_original_1.3b/validation.csv_judged.csv",
                "legend": "PPO OpenAI checkpoint",
                "color": "blue",
            },
            "PPO new": {
                "file_name": f"sampled_data/refactor-chosen-rejected3/ppo/EleutherAI/pythia-1.4b-deduped/{seed}/query_responses_gpt-3.5-turbo-0125.csv",
                "legend": "PPO new",
                "color": "purple",
            },
            "SFT": {
                "file_name": f"sampled_data/refactor-chosen-rejected3/sft/{base_model}/{seed}/query_responses_gpt-3.5-turbo-0125.csv",
                "legend": "SFT",
                "color": "green",
            },
        }


        for key in data_dict:
            df = pd.read_csv(data_dict[key]["file_name"])
            df = df.dropna()
            clean_ref_responses = [res[: res.find("<|endoftext|>")] for res in df["reference_responses"].tolist()]

            if length_by == "token":
                response_tokens = [
                    tokenizer.encode(res, add_special_tokens=False) for res in df["postprocessed_response"].tolist()
                ]
                response_tokens_lens = [len(item) for item in response_tokens]

                refrence_tokens = [tokenizer.encode(res, add_special_tokens=False) for res in clean_ref_responses]
                refrence_tokens_lens = [len(item) for item in refrence_tokens]
            else:
                response_tokens_lens = [len(res) for res in df["postprocessed_response"].tolist()]
                refrence_tokens_lens = [len(res) for res in clean_ref_responses]

            log_summary_len_by_ref_len = np.log(np.array(response_tokens_lens) / np.array(refrence_tokens_lens))

            log_summary_len_by_ref_len_and_summaries = list(zip(log_summary_len_by_ref_len, df["postprocessed_response"].tolist(), clean_ref_responses))
            log_summary_len_by_ref_len_and_summaries.sort(key=lambda x: x[0])
            # make 10 bins equally spaced
            log_len_ratio_bins = np.linspace(log_summary_len_by_ref_len.min(), log_summary_len_by_ref_len.max(), 10)

            prefered_values = df["preferred"].to_list()

            bin_values = [[] for _ in range(len(log_len_ratio_bins))]
            bin_indices = np.digitize(log_summary_len_by_ref_len, log_len_ratio_bins)
            for i, (log_len_ratio, preferred) in enumerate(zip(log_summary_len_by_ref_len, prefered_values)):
                bin_values[bin_indices[i] - 1].append(preferred)

            # count how many "ours" are in each bin (normalized)
            ZERO_DIVISION = 1e-10
            bin_preferred_fractions = [bin.count("ours") / (len(bin) + ZERO_DIVISION) for bin in bin_values]
            # plt.plot(log_len_ratio_bins, bin_preferred_fractions, marker="o")
            plot_df = pd.DataFrame(
                {
                    "log(summary length / reference length)": log_len_ratio_bins,
                    "fraction of summaries preferred to ref": bin_preferred_fractions,
                }
            )
            ax = sns.regplot(
                x="log(summary length / reference length)",
                y="fraction of summaries preferred to ref",
                data=plot_df,
                # logistic=True,
                y_jitter=0.03,
                x_bins=10,
                ci=None,
                # hue=None,
                color=data_dict[key]["color"],
                label=data_dict[key]["legend"],
                line_kws={"linestyle": "--"},
                # order=2
                ax=axes[seed_idx, base_model_idx],
            )
        ax = axes[seed_idx, base_model_idx]
        ax.axhline(y=0.5, color='black', linestyle='-.', label='reference summary')
        ax.axvline(x=0.0, color='black', linestyle='-.')
        if seed_idx == len(seeds) - 1 and base_model_idx == len(base_models) - 1:
            ax.legend()

        # plt.legend()
        if seed_idx == len(seeds) - 1:
            ax.set_xlabel("log(summary length / reference length)")
        else:
            ax.set_xlabel("")


        if base_model_idx == 0:
            ax.set_ylabel("Winrate against reference summaries\n(according to gpt-3.5-turbo-0125)")
        else:
            ax.set_ylabel("")

        if seed_idx == 0:
            ax.set_title(f"{base_model} seed {seed}")
        else:
            ax.set_title(f"seed {seed}")
        
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)

plt.tight_layout()
fig.savefig(f"images/full_rlhf_log_length_by_{length_by}.png")
fig.savefig(f"images/full_rlhf_log_length_by_{length_by}.pdf")
