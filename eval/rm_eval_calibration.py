import math
from dataclasses import dataclass
from typing import Literal

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tyro


@dataclass
class Args:
    eval_on: Literal["gpt-3.5", "dataset"] = "dataset"


def pd_xand(a, b):
    return (a & b) | (~a & ~b)


def get_margin_df_gpt35(df):
    margin_df = pd.DataFrame()
    margin_df["margin"] = df["scores"] - df["reference_scores"]
    margin_df["accuracy"] = pd_xand(
        (df["preferred"] == "ours"), (df["scores"] > df["reference_scores"])
    ).astype(float)
    return margin_df


def get_margin_df_dataset(df):
    margin_df = pd.DataFrame()
    margin_df["margin"] = df["chosen_rewards"] - df["rejected_rewards"]
    margin_df["accuracy"] = df["accuracy"].astype(float)
    return margin_df


def score_calibration(df):
    # our negative range doesn't seem to go as low as positive goes high
    # maybe we should cut off our max margin at 99.5% quantile
    margin_max = df["margin"].quantile(0.995)
    # margin_max = df['margin'].max()
    margin_max_rounded = np.round(margin_max * 2) / 2
    interval_tuples = [(i, i + 0.5) for i in np.arange(0, margin_max_rounded, 0.5)]
    bins = pd.IntervalIndex.from_tuples(interval_tuples)
    df["margin_bins"] = pd.cut(df["margin"].abs(), bins)

    calibration_df = df.groupby("margin_bins")["accuracy"].mean()
    calibration_df = calibration_df.reset_index()
    calibration_df["middle"] = calibration_df["margin_bins"].apply(lambda x: x.mid)

    return calibration_df


def plot_score_calibration(
    file_format,
    preprocess_func,
    title,
    save_fname,
    skip_model_seeds={},
):
    models = [
        "EleutherAI/pythia-1b-deduped",
        "EleutherAI/pythia-2.8b-deduped",
        "EleutherAI/pythia-6.9b-deduped",
    ]
    colors = ["#8ac6e4", "#2b8cbe", "#08306b"]
    SEEDS = [44413, 55513, 66613, 77713]

    for model, color in zip(models, colors):
        margin_dfs = []
        for seed in SEEDS:
            if model in skip_model_seeds and seed in skip_model_seeds[model]:
                continue
            eval_fname = file_format.format(model=model, seed=seed)
            margin_df = preprocess_func(pd.read_csv(eval_fname))
            calibration_df = score_calibration(margin_df)
            margin_dfs.append(calibration_df)

        sns.lineplot(
            data=pd.concat(margin_dfs).reset_index(),
            x="middle",
            y="accuracy",
            estimator="mean",
            errorbar="sd",
            color=color,
        )

    perfect_calibration = margin_dfs[-1]["middle"].apply(
        lambda x: 1 / (1 + math.exp(-x))
    )
    sns.lineplot(
        x=margin_dfs[-1]["middle"],
        y=perfect_calibration,
        color="black",
        label="perfect calibration",
    )

    plt.title(title)
    plt.xlabel("Score difference")
    plt.ylabel("Accuracy")
    plt.gca().invert_yaxis()

    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(
        boundaries=[0, 1, 2, 3], ncolors=4
    )  # Adjust boundaries to match the number of colors
    cb = plt.colorbar(
        cm.ScalarMappable(norm=norm, cmap=cmap), ax=plt.gca(), orientation="vertical"
    )
    cb.set_ticks([0.5, 1.5, 2.5])
    cb.set_ticklabels(["1b", "2.8b", "6.9b"])
    cb.set_label("Parameters", rotation=270, labelpad=15)
    plt.legend(loc="lower right")

    plt.grid(True, which="both", ls="--", c="0.7")
    plt.tight_layout()
    plt.savefig(save_fname)
    plt.savefig(save_fname.replace("png", "pdf"))
    plt.clf()
    plt.close()


if __name__ == "__main__":
    args = tyro.cli(Args)

    if args.eval_on == "dataset":
        plot_score_calibration(
            file_format="sampled_data/refactor-chosen-rejected3/reward_eval/{model}/{seed}/query_responses.csv",
            preprocess_func=get_margin_df_dataset,
            title="Calibration with TLDR Preferences Data",
            save_fname="images/calibration_tldr.png",
            # missing file
            skip_model_seeds={"EleutherAI/pythia-2.8b-deduped": [44413]},
        )

    else:
        plot_score_calibration(
            file_format="sampled_data/refactor-chosen-rejected3/ppo_left_padding/{model}/{seed}/query_responses_gpt-3.5-turbo-0125.csv",
            preprocess_func=get_margin_df_gpt35,
            title="Calibration with GPT-3.5 Turbo on Generated vs SFT Data",
            save_fname="images/calibration_gpt35.png",
            # these values seem off, very low accuracy, could be bad seeds
            skip_model_seeds={"EleutherAI/pythia-1b-deduped": [44413, 66613]},
        )
