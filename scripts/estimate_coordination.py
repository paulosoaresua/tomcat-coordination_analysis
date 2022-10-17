import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

from coordination.common.dataset import InputFeaturesDataset
from coordination.model.gaussian_coordination_blending_latent_vocalics import \
    GaussianCoordinationBlendingInferenceLatentVocalics
from coordination.plot.coordination import plot_coordination_estimation


def estimate_coordination(dataset_path: str, out_dir: str, num_particles: int, seed: int, num_jobs: int):
    input_features: InputFeaturesDataset
    scores: np.ndarray
    with open(dataset_path, "rb") as f:
        dataset = pickle.load(f)

    input_features, scores = dataset

    if input_features.num_trials == 0:
        raise Exception("The dataset is empty.")

    # Pre-defined parameters
    num_features = input_features.series[0].vocalics.num_features

    MEAN_COORDINATION_PRIOR = 0
    STD_COORDINATION_DRIFT = 0.05
    STD_COORDINATION_PRIOR = 0.01  # The process starts with no coordination
    MEAN_PRIOR_VOCALICS = np.zeros(num_features)
    STD_PRIOR_VOCALICS = np.ones(num_features)
    STD_COORDINATED_VOCALICS = np.ones(num_features)
    STD_OBSERVED_VOCALICS = np.ones(num_features) * 0.5

    model = GaussianCoordinationBlendingInferenceLatentVocalics(
        mean_prior_coordination=MEAN_COORDINATION_PRIOR,
        std_prior_coordination=STD_COORDINATION_PRIOR,
        std_coordination_drifting=STD_COORDINATION_DRIFT,
        mean_prior_latent_vocalics=MEAN_PRIOR_VOCALICS,
        std_prior_latent_vocalics=STD_PRIOR_VOCALICS,
        std_coordinated_latent_vocalics=STD_COORDINATED_VOCALICS,
        std_observed_vocalics=STD_OBSERVED_VOCALICS,
        fix_coordination_on_second_half=False,
        num_particles=num_particles,
        seed=seed,
        show_progress_bar=True
    )

    params = model.predict(input_features)

    # Estimate and plot coordination in each trial
    plot_dir = f"{out_dir}/coordination/plots"
    os.makedirs(plot_dir, exist_ok=True)
    for i, series in enumerate(input_features.series):
        fig = plot_coordination_estimation(params[i][0], np.sqrt(params[i][1]), series, "tab:orange")
        plt.savefig(f"{plot_dir}/{series.uuid}.png")
        plt.close(fig)

    path = f"{out_dir}/coordination/estimation/estimates.pkl"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(params, f)

    # Save the original dataset used to estimate coordination
    path = f"{out_dir}/coordination/estimation/dataset.pkl"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(dataset, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Estimates coordination for a series of trials."
    )

    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path containing the dataset to use in the model.")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Directory directory where the results must be saved.")
    parser.add_argument("--n_particles", type=int, required=False, default=10000,
                        help="Number of particles for inference.")
    parser.add_argument("--seed", type=int, required=False, default=0,
                        help="Random seed.")
    parser.add_argument("--n_jobs", type=int, required=False, default=1,
                        help="Number of threads for parallelism.")

    args = parser.parse_args()
    estimate_coordination(args.dataset_path, args.out_dir, args.n_particles, args.seed, args.n_jobs)
