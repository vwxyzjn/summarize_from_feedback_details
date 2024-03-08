import glob
import datasets
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

datasets.disable_progress_bar()


def check_rm_agree(directory_path):
    csv_files = [
        file_name
        for file_name in glob.glob(directory_path + "/**/*.csv", recursive=True)
        if "gpt-3.5-turbo-0125" in file_name
    ]
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

    agreed_ratio = (agreed_preferred_reference + agreed_preferred_ours) / total_num

    return agreed_ratio


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
    xs = []
    ys = []

    for model_name, path in exp_name_path[1].items():
        xs.append(model_sizes[model_name])
        ys.append(check_rm_agree(path))

    if exp_name_path[0] == "ppo_left_padding":
        label = "Our PPO Reproduction"
    elif exp_name_path[0] == "ppo_left_padding_new":
        label = "Our PPO Reproduction New"

    plt.scatter(
        xs,
        ys,
        label=label,
        color=color,
        alpha=0.5,
    )


# Adding the human baseline and ensemble of humans
# plt.axhline(y=0.5, color="black", linestyle="-.", label="reference summary")

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
plt.savefig('rm_scale_plot.png')