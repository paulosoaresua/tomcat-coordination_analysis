from typing import List

import argparse
import os
from datetime import datetime
from multiprocessing import Pool

import os
import pickle

from ast import literal_eval
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from scripts.formatting import set_size

from coordination.model.vocalics_semantic_model import VocalicsSemanticModel
from coordination.model.utils.beta_coordination_blending_latent_vocalics import BetaCoordinationLatentVocalicsDataset, \
    BetaCoordinationLatentVocalicsParticlesSummary, BetaCoordinationLatentVocalicsDataSeries


def fit(dataset_path: str, initial_coordination: float, burn_in: int,
        num_samples: int, num_chains: int, num_inference_jobs: int, num_trial_jobs: int, seed: int, out_dir: str,
        ref_date: str, data_idx: int, features: List[str], link: bool):
    """
    We fit parameters and coordination for each individual trial as the parameters of the model might vary per team.
    """

    with open(dataset_path, "rb") as f:
        evidence = BetaCoordinationLatentVocalicsDataset.from_latent_vocalics_dataset(pickle.load(f))

    evidence.keep_vocalic_features(features)

    evidence.normalize_per_subject()

    if ref_date is None or len(ref_date) == 0:
        ref_date = datetime.now().strftime("%Y.%m.%d--%H.%M.%S")

    out_dir = f"{out_dir}/{ref_date}"

    # Fit and save each one of the trials
    inference_data = []
    result_table = []

    os.makedirs(out_dir, exist_ok=True)

    trials = [data_idx] if data_idx >= 0 else evidence.num_trials
    for i in trials:
        print("")
        print(f"Trial {i + 1}/{evidence.num_trials}: {evidence.series[i].uuid}")

        if not link:
            evidence.series[i].disable_speech_semantic_links()

        idata = fit_helper(evidence=evidence.series[i],
                           initial_coordination=initial_coordination,
                           burn_in=burn_in,
                           num_samples=num_samples,
                           num_chains=num_chains,
                           num_jobs=num_inference_jobs,
                           seed=seed)

        isummary = BetaCoordinationLatentVocalicsParticlesSummary.from_inference_data(idata)

        inference_data.append(idata)
        result_table.append(
            [evidence.series[i].uuid, isummary.coordination_mean.tolist(), np.sqrt(isummary.coordination_var).tolist(),
             isummary.coordination_mean.mean(), np.sqrt(isummary.coordination_var).mean()])

        # plot_coordination(f"{out_dir}/plots", evidence.series[i], isummary)

    df = pd.DataFrame(result_table,
                      columns=["experiment_id", "coordination_means", "coordination_stds", "avg_coordination_mean",
                               "avg_coordination_std"])

    if data_idx >= 0:
        df.to_csv(f"{out_dir}/inference_table_{data_idx}.csv")
        with open(f"{out_dir}/inference_data_{data_idx}.pkl", "wb") as f:
            pickle.dump(inference_data, f)
    else:
        df.to_csv(f"{out_dir}/inference_table.csv")
        with open(f"{out_dir}/inference_data.pkl", "wb") as f:
            pickle.dump(inference_data, f)


def fit_helper(evidence: BetaCoordinationLatentVocalicsDataSeries, initial_coordination: float, burn_in: int,
               num_samples: int, num_chains: int,
               num_jobs: int, seed: int):
    model = VocalicsSemanticModel(
        initial_coordination=initial_coordination,
        num_vocalic_features=evidence.observed_vocalics.num_features,
        num_subjects=3
    )

    model.parameters.reset()
    idata = model.fit(evidence=evidence,
                      burn_in=burn_in,
                      num_samples=num_samples,
                      num_chains=num_chains,
                      num_jobs=num_jobs,
                      seed=seed)
    return idata


# def plot_coordination(out_dir: str, series: BetaCoordinationLatentVocalicsDataSeries, isummary):
#     os.makedirs(out_dir, exist_ok=True)
#     fig, ax = plt.subplots(1, 1, figsize=set_size(800, fraction=1, subplots=(1, 1)))
#
#     x_values = np.arange(len(isummary.coordination_mean))
#     y_values = isummary.coordination_mean
#     lower_y_values = y_values - np.sqrt(isummary.coordination_var)
#     upper_y_values = y_values + np.sqrt(isummary.coordination_var)
#
#     times_with_obs = [t for t, mask in enumerate(series.vocalics_mask) if mask == 1]
#     times_with_links = series.speech_semantic_links_times
#     ax.scatter(times_with_obs, np.ones(len(times_with_obs)) * 1.05, marker="s", s=2, color="tab:purple")
#     ax.scatter(times_with_links, np.ones(len(times_with_links)) * 1.03, marker="s", s=2, color="black")
#
#     ax.plot(x_values, y_values, linestyle="--", marker="o", color="tab:red", linewidth=0.5, markersize=2)
#     ax.fill_between(x_values, lower_y_values, upper_y_values, color="tab:pink", alpha=0.5)
#     ax.set_xlim([-0.1, x_values[-1] + 0.1])
#     ax.set_ylim([-0.05, 1.1])
#     ax.set_xlabel(r"Time Step (seconds)")
#     ax.set_ylabel(r"Coordination")
#     fig.savefig(f"{out_dir}/{series.uuid}.pdf", format='pdf', bbox_inches='tight')
#     plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trains a coordination model on a dataset of observed vocalic features over time."
    )

    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to the .pkl file containing the data.")
    parser.add_argument("--c0", type=float, required=True, help="Assumed initial coordination value.")
    parser.add_argument("--burn_in", type=int, required=True, help="Number of discarded samples per chain.")
    parser.add_argument("--n_samples", type=int, required=True, help="Number of samples per chain.")
    parser.add_argument("--n_chains", type=int, required=True, help="Number of independent chains.")
    parser.add_argument("--n_i_jobs", type=int, required=False, default=1, help="Number of jobs during inference.")
    parser.add_argument("--n_t_jobs", type=int, required=False, default=1, help="Number of jobs to split the trials.")
    parser.add_argument("--seed", type=int, required=False, default=0, help="Random seed for replication.")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory where to save the model.")
    parser.add_argument("--ref_date", type=str, required=False, default="",
                        help="Name of the folder inside out_dir where to save inference artifacts. If not informed, the "
                             "program will create a folder with the timestamp at the execution time.")
    parser.add_argument("--data_idx", type=int, required=False, default=-1,
                        help="Data index. If different than -1, it will perform inference in the experiment at the informed index. "
                             "It can be used to execute inferences on different trials in parallel since PyMC can not be spawned in parallel "
                             "by the main process.")
    parser.add_argument("--features", type=str, required=False, default="pitch, intensity, jitter, shimmer",
                        help="List of vocalic features to consider. It can be any subset of the default value.")
    parser.add_argument("--link", type=int, required=False, default=0,
                        help="Whether to use a model that considers speech semantic link.")

    args = parser.parse_args()


    def format_feature_name(name: str):
        return name.strip().lower()


    fit(dataset_path=args.data_path,
        initial_coordination=args.c0,
        burn_in=args.burn_in,
        num_samples=args.n_samples,
        num_chains=args.n_chains,
        num_inference_jobs=args.n_i_jobs,
        num_trial_jobs=args.n_t_jobs,
        seed=args.seed,
        out_dir=args.out_dir,
        ref_date=args.ref_date,
        data_idx=args.data_idx,
        features=list(map(format_feature_name, args.features.split(","))),
        link=args.link > 0)
