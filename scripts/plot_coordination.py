from typing import Union

import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from formatting import set_size

from coordination.model.utils.beta_coordination_blending_latent_vocalics import BetaCoordinationLatentVocalicsDataset


def generate_plots(inference_path: str, dataset_path: str, width: str, out_dir: str):
    with open(inference_path, "rb") as f:
        inference_summaries = pickle.load(f)

    with open(dataset_path, "rb") as f:
        dataset = BetaCoordinationLatentVocalicsDataset.from_latent_vocalics_dataset(pickle.load(f))

    os.makedirs(out_dir, exist_ok=True)

    for i, summary in tqdm(enumerate(inference_summaries), desc="Trial"):
        fig, ax = plt.subplots(1, 1, figsize=set_size(width, fraction=1, subplots=(1, 1)))

        x_values = np.arange(len(summary.coordination_mean))
        y_values = summary.coordination_mean
        lower_y_values = y_values - np.sqrt(summary.coordination_var)
        upper_y_values = y_values + np.sqrt(summary.coordination_var)

        times_with_obs = [t for t, mask in enumerate(dataset.vocalics_mask[i]) if mask == 1]
        ax.scatter(times_with_obs, np.ones(len(times_with_obs)) * 1.05, marker="s", s=2, color="tab:purple")

        ax.plot(x_values, y_values, linestyle="--", marker="o", color="tab:red", linewidth=0.5, markersize=2)
        ax.fill_between(x_values, lower_y_values, upper_y_values, color="tab:pink", alpha=0.5)
        ax.set_xlim([-0.1, x_values[-1] + 0.1])
        ax.set_ylim([-0.05, 1.1])
        ax.set_xlabel(r"Time Step (seconds)")
        ax.set_ylabel(r"Coordination")
        fig.savefig(f"{out_dir}/{dataset.series[i].uuid}.pdf", format='pdf', bbox_inches='tight')
        plt.close()


def float_or_str(value):
    try:
        return float(value)
    except:
        return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plots estimated coordination levels in different trials from an inference summaries file."
    )

    parser.add_argument("--inference_path", type=str, required=True,
                        help="Path to an inference summaries pickled file.")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to the dataset used to train the coordination model. We need that to get the trial "
                             "ids associated with each inference.")
    parser.add_argument("--width", type=float_or_str, required=True,
                        help="Float value or pre-defined style name")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Directory where the plots must be saved.")

    args = parser.parse_args()

    generate_plots(inference_path=args.inference_path,
                   dataset_path=args.dataset_path,
                   width=args.width,
                   out_dir=args.out_dir)
