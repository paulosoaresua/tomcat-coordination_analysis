from typing import Any, Dict

import argparse
import random
import os
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_validate as sklearn_cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from torch.utils.tensorboard import SummaryWriter

from coordination.common.dataset import InputFeaturesDataset, IndexToDatasetTransformer
from coordination.model.coordination_transformer import CoordinationTransformer
from coordination.model.gaussian_coordination_blending_latent_vocalics import \
    GaussianCoordinationBlendingInferenceLatentVocalics
from coordination.plot.regression import plot_coordination_vs_score_regression

from tqdm import tqdm


class ProgressTracker:

    def __init__(self, num_test_splits: int, out_dir: str):
        self.out_dir = out_dir
        self.fold = 1
        self._pbar = tqdm(total=num_test_splits, desc="Tuning & Testing...", position=0)

    def update(self, tuning_results: Dict[str, Any], test_results: Dict[str, Any], regression_plot: plt.figure):
        results_filepath = f"{self.out_dir}/tuning/results/fold_{self.fold}.pkl"
        os.makedirs(os.path.dirname(results_filepath), exist_ok=True)
        with open(results_filepath, "wb") as f:
            pickle.dump(tuning_results, f)

        results_filepath = f"{self.out_dir}/test/results/fold_{self.fold}.pkl"
        os.makedirs(os.path.dirname(results_filepath), exist_ok=True)
        with open(results_filepath, "wb") as f:
            pickle.dump(test_results, f)

        plot_filepath = f"{self.out_dir}/test/plots/regression/fold_{self.fold}.png"
        os.makedirs(os.path.dirname(plot_filepath), exist_ok=True)
        plt.savefig(plot_filepath, format="png")
        plt.close(regression_plot)

        self._pbar.update()
        self.fold += 1


class CrossValidationHelper:

    def __init__(self, num_outer_splits: int, progress_tracker: ProgressTracker):
        self.num_outer_splits = num_outer_splits
        self.progress_tracker = progress_tracker

    @staticmethod
    def tuning_estimation(estimator, X, y):
        """
        Estimates the MSE in the validation set. The inner cross validation splits the training into training and
        validation to perform hyperparameter tuning.
        """

        y_hat, _ = estimator.predict(X=X, return_std=True)
        mse = mean_squared_error(y, y_hat)

        return mse

    def testing_estimation(self, estimator, X, y):
        best_estimator = estimator.best_estimator_

        indexer = best_estimator.steps[0][1]
        coordination = best_estimator.steps[1][1]
        regressor = best_estimator.steps[2][1]

        y_hat, _ = best_estimator.predict(X=X, return_std=True)
        mse = mean_squared_error(y, y_hat)
        nll = regressor.scores_[-1]

        if len(y) >= 2:
            r, p = pearsonr(coordination.output_.flatten(), y)
        else:
            r = 0
            p = -1

        # Plot regression
        fig = plot_coordination_vs_score_regression(coordination.output_.flatten(), y, regressor,
                                                    f"$r = {r:.2f}; p-value = {p:.2f}$")
        tunning_results = {
            "results": pd.DataFrame(estimator.cv_results_),
            "average_coordinations": coordination.output_.flatten()
        }
        test_results = {
            "trials": [series.uuid for series in indexer.transform(X).series]
        }
        self.progress_tracker.update(tunning_results, test_results, fig)

        return {
            "mse": mse,
            "pearson-r": r,
            "pearson-p": p,
            "nll": nll
        }

    def __copy__(self):
        return CrossValidationHelper(self.num_outer_splits, self.progress_tracker)

    def __deepcopy__(self, memo):
        return CrossValidationHelper(self.num_outer_splits, self.progress_tracker)


def cross_validate(dataset_path: str, out_dir: str, num_particles: int, cv: int, seed: int, num_jobs: int):
    input_features: InputFeaturesDataset
    scores: np.ndarray
    with open(dataset_path, "rb") as f:
        input_features, scores = pickle.load(f)

    if input_features.num_trials == 0:
        raise Exception("The dataset is empty.")

    # Pre-defined parameters
    num_features = input_features.series[0].vocalics.num_features

    MEAN_COORDINATION_PRIOR = 0
    STD_COORDINATION_PRIOR = 1E-16  # The process starts with no coordination
    MEAN_PRIOR_VOCALICS = np.zeros(num_features)
    STD_PRIOR_VOCALICS = np.ones(num_features)
    STD_COORDINATED_VOCALICS = np.ones(num_features)
    STD_OBSERVED_VOCALICS = np.ones(num_features)

    # Standard deviation of coordination drifting will be estimated via parameter tuning and cross validation.
    STD_COORDINATION_DRIFT = 0

    # Hyper-parameters to try
    param_grid = {
        "score_regressor__alpha_init": [0.001, 0.1, 1],
        "score_regressor__lambda_init": [0.001, 0.1, 1],
        "coordination__std_coordination_drifting": [0.01, 0.05, 0.1, 0.2],
    }

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
        seed=0
    )

    # model.configure_tensorboard("/Users/paulosoares/code/tomcat-coordination/data/tensorboard")
    regressor = BayesianRidge(tol=1e-6, fit_intercept=True, compute_score=True)
    coordination_transformer = CoordinationTransformer(model)

    pipeline = Pipeline([
        ("index_2_dataset", IndexToDatasetTransformer(input_features)),
        ("coordination", coordination_transformer),
        ("score_regressor", regressor),
    ])

    tracker = ProgressTracker(2, out_dir)
    helper = CrossValidationHelper(cv, tracker)

    np.random.seed(seed)
    random.seed(seed)
    clf = GridSearchCV(estimator=pipeline,
                       param_grid=param_grid,
                       cv=2,
                       scoring=helper.tuning_estimation,
                       n_jobs=num_jobs)

    start = time.time()

    # We run this always with 1 thread to avoid race condition in the progress tracker. The internal cross validation
    # can use multiple threads.
    test_results = sklearn_cross_validate(
        estimator=clf,
        X=np.arange(input_features.num_trials)[:, np.newaxis],
        y=scores,
        cv=2,
        scoring=helper.testing_estimation
    )
    end = time.time()

    results_filepath = f"{out_dir}/test/results/final.pkl"
    os.makedirs(os.path.dirname(results_filepath), exist_ok=True)
    pd.DataFrame(test_results).to_pickle(results_filepath)

    print(f"{(end - start)} seconds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Creates a dataset of vocalics for mission 1, mission 2 and both missions for all the serialized "
                    "trials in a given folder."
    )

    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path containing the dataset to use in the model.")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Directory directory where the results of the cross validation must be saved.")
    parser.add_argument("--n_particles", type=int, required=False, default=10000,
                        help="Number of particles for inference.")
    parser.add_argument("--cv", type=int, required=False, default=5,
                        help="Number of folds.")
    parser.add_argument("--seed", type=int, required=False, default=0,
                        help="Random seed.")
    parser.add_argument("--n_jobs", type=int, required=False, default=1,
                        help="Number of threads for parallelism.")

    args = parser.parse_args()
    cross_validate(args.dataset_path, args.out_dir, args.n_particles, args.cv, args.seed, args.n_jobs)
