import argparse
import pickle

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import BayesianRidge

from coordination.common.dataset import InputFeaturesDataset
from coordination.plot.regression import plot_coordination_vs_score_regression


def estimate_regression(coordination_path: str, dataset_path: str, plot_out_path: str):
    input_features: InputFeaturesDataset
    scores: np.ndarray
    with open(dataset_path, "rb") as f:
        input_features, scores = pickle.load(f)

    with open(coordination_path, "rb") as f:
        params = pickle.load(f)

    coordination = np.array([np.mean(means) for means, variances in params])

    r, p = pearsonr(coordination, scores)
    regressor = BayesianRidge(tol=1e-6, fit_intercept=True, compute_score=True, alpha_init=1, lambda_init=1e-3)
    regressor.fit(coordination[:, np.newaxis], scores)

    label = f"$r = {r:.2f}, p = {p:.4f}$"
    fig = plot_coordination_vs_score_regression(coordination, scores, regressor, label)
    plt.savefig(plot_out_path)
    plt.tight_layout()
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Creates a dataset of vocalics for mission 1, mission 2 and both missions for all the serialized "
                    "trials in a given folder."
    )

    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path containing the dataset to use in the model.")
    parser.add_argument("--plot_out_path", type=str, required=True,
                        help="Path where the regression plot must be saved.")
    parser.add_argument("--coordination_path", type=str, required=True,
                        help="Path containing estimated coordination.")

    args = parser.parse_args()
    estimate_regression(args.coordination_path, args.dataset_path, args.plot_out_path)
