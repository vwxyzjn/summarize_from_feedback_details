import glob
import datasets
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

datasets.disable_progress_bar()


def get_rm_agree_data(csv_files):
    datas = datasets.load_dataset("csv", data_files=csv_files)["train"]
    total_num = len(datas)

    datas_preferred_reference = datas.filter(
        lambda example: example["preferred"] == "reference"
    )
    agreed_preferred_reference = np.sum(
        np.array(datas_preferred_reference["scores"])
        < np.array(datas_preferred_reference["reference_scores"])
    )

    datas_preferred_ours = datas.filter(lambda example: example["preferred"] == "ours")
    agreed_preferred_ours = np.sum(
        np.array(datas_preferred_ours["scores"])
        > np.array(datas_preferred_ours["reference_scores"])
    )

    return agreed_preferred_reference, agreed_preferred_ours, total_num


def check_rm_agree(directory_path):
    csv_files = [
        file_name
        for file_name in glob.glob(directory_path + "/**/*.csv", recursive=True)
        if "gpt-3.5-turbo-0125" in file_name
    ]

    total_agreed_ratio = []
    for file in csv_files:
        agreed_preferred_reference, agreed_preferred_ours, total_num = (
            get_rm_agree_data([file])
        )
        agreed_ratio = (agreed_preferred_reference + agreed_preferred_ours) / total_num
        print(f"Agreed ratio for {file} is {agreed_ratio}")
        total_agreed_ratio.append(agreed_ratio)

    return total_agreed_ratio


model_sizes = {
    "pythia-1b-deduped": 1e9,
    "pythia-2.8b-deduped": 2.8e9,
    "pythia-6.9b-deduped": 6.9e9,
}


need_check_paths = {
    "ppo_left_padding": {
        "pythia-1b-deduped": "sampled_data/refactor-chosen-rejected3/ppo_left_padding/EleutherAI/pythia-1b-deduped",
        "pythia-2.8b-deduped": "sampled_data/refactor-chosen-rejected3/ppo_left_padding/EleutherAI/pythia-2.8b-deduped",
        "pythia-6.9b-deduped": "sampled_data/refactor-chosen-rejected3/ppo_left_padding/EleutherAI/pythia-6.9b-deduped",
    },
    # ppo_left_padding_new
    "ppo_left_padding_new": {
        "pythia-1b-deduped": "sampled_data/refactor-chosen-rejected3/ppo_left_padding_new/EleutherAI/pythia-1b-deduped",
        "pythia-2.8b-deduped": "sampled_data/refactor-chosen-rejected3/ppo_left_padding_new/EleutherAI/pythia-1b-deduped",
    },
}

colors = sns.color_palette("colorblind", n_colors=len(need_check_paths))
for exp_name_path, color in zip(need_check_paths.items(), colors):
    if exp_name_path[0] == "ppo_left_padding":
        label = "Our PPO Reproduction"
    elif exp_name_path[0] == "ppo_left_padding_new":
        label = "Our PPO Reproduction New"

    xs = []
    ys_mean = []
    ys_std = []

    for model_name, path in exp_name_path[1].items():
        xs.append(model_sizes[model_name])
        ys_mean.append(np.mean((check_rm_agree(path))))
        ys_std.append(np.std((check_rm_agree(path))))

    plt.errorbar(
        xs,
        ys_mean,
        yerr=ys_std,
        label=label,
        marker="o",
        color=color,
        capsize=5,
    )

# Setting the title and labels
plt.title("Reward Model scaling")
plt.xlabel("Reward Model size")
plt.ylabel(f"RM consistency to gpt-3.5")
plt.xscale("log")  # Setting the x-axis to a logarithmic scale
plt.xlim(5e8, 1e10)
plt.legend()

# Display the plot
plt.grid(True, which="both", ls="--", c="0.7")
plt.tight_layout()
plt.savefig("rm_scale_plot.png")
