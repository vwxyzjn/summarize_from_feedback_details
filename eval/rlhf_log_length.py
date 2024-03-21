import json
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

import json
import csv
from transformers import AutoTokenizer
import seaborn as sns

seed = 55513
length_by = "token"  # "token" or "char"
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-1b-deduped")
data_dict = {
    "PPO": {
        "file_name": f"sampled_data/refactor-chosen-rejected3/ppo_left_padding/EleutherAI/pythia-6.9b-deduped/{seed}/query_responses_gpt-3.5-turbo-0125.csv",
        "legend": "PPO",
        "color": "red",
    },
    "DPO": {
        "file_name": f"sampled_data/refactor-chosen-rejected2/dpo/EleutherAI/pythia-6.9b-deduped/{seed}/query_responses_gpt-3.5-turbo-0125.csv",
        "legend": "DPO",
        "color": "blue",
    },
    "SFT": {
        "file_name": f"sampled_data/refactor-chosen-rejected3/sft/EleutherAI/pythia-6.9b-deduped/{seed}/query_responses_gpt-3.5-turbo-0125.csv",
        "legend": "SFT",
        "color": "green",
    },
}


fig = plt.figure(figsize=(7, 5))
for key in data_dict:
    df = pd.read_csv(data_dict[key]["file_name"])
    df = df.dropna()
    clean_ref_responses = [res[: res.find("<|endoftext|>")] for res in df["reference_responses"].tolist()]

    print(tokenizer.encode(df["postprocessed_response"].tolist()[0]))
    print(tokenizer.encode(clean_ref_responses[0]))

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
    # make 10 bins equally spaced
    log_len_ratio_bins = np.linspace(log_summary_len_by_ref_len.min(), log_summary_len_by_ref_len.max(), 10)

    prefered_values = df["preferred"].to_list()

    print(f"{log_len_ratio_bins=}")
    print(f"{prefered_values[:10]=}")
    print(f"{np.mean(response_tokens_lens)=}")
    print(f"{np.mean(refrence_tokens_lens)=}")
    print(f"{np.mean(log_summary_len_by_ref_len)=}")
    print(f"{len(response_tokens_lens)=}")
    print(f"{len(refrence_tokens_lens)=}")

    bin_values = [[] for _ in range(len(log_len_ratio_bins))]
    bin_indices = np.digitize(log_summary_len_by_ref_len, log_len_ratio_bins)
    print(f"{bin_indices[:10]=}")
    for i, (log_len_ratio, preferred) in enumerate(zip(log_summary_len_by_ref_len, prefered_values)):
        bin_values[bin_indices[i] - 1].append(preferred)

    # count how many "ours" are in each bin (normalized)
    ZERO_DIVISION = 1e-10
    bin_preferred_fractions = [bin.count("ours") / (len(bin) + ZERO_DIVISION) for bin in bin_values]
    print(f"{bin_preferred_fractions=}")

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
        logistic=True,
        y_jitter=0.03,
        x_bins=10,
        ci=None,
        # hue=None,
        color=data_dict[key]["color"],
        label=data_dict[key]["legend"],
        line_kws={"linestyle": "--"},
    )

plt.legend()
plt.xlabel("log(summary length / reference length)", fontsize=16)
plt.ylabel("fraction of summaries preferred to ref", fontsize=16)

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.tight_layout()
plt.savefig(f"images/rlhf_log_length_by_{length_by}.png")
