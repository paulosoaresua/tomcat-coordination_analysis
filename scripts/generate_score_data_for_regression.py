from typing import List

import argparse
from glob import glob
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from coordination.component.speech.vocalics_component import SegmentationMethod, VocalicsComponent
from coordination.entity.trial import Trial
from coordination.inference.vocalics import ContinuousCoordinationInferenceFromVocalics
from coordination.plot.coordination import add_discrete_coordination_bar
from scripts.utils import configure_log


def estimate(trials_dir: str, data_dir: str, plot_coordination: bool):
    logs_dir = f"{data_dir}/logs"
    plots_dir = f"{data_dir}/plots"

    os.makedirs(logs_dir, exist_ok=True)
    if plot_coordination:
        os.makedirs(plots_dir, exist_ok=True)

    # Constants
    NUM_TIME_STEPS = 17 * 60  # (17 minutes of mission in seconds)
    M = int(NUM_TIME_STEPS / 2)  # We assume coordination in the second half of the period is constant
    NUM_FEATURES = 2  # Pitch and Intensity

    # Common parameters
    MEAN_PRIOR_VOCALICS = np.zeros(NUM_FEATURES)
    STD_PRIOR_VOCALICS = np.ones(NUM_FEATURES)
    STD_COORDINATED_VOCALICS = np.ones(NUM_FEATURES)

    # Parameters of the continuous model
    MEAN_COORDINATION_PRIOR = 0
    STD_COORDINATION_PRIOR = 0  # The process starts with no coordination
    STD_COORDINATION_DRIFT = 0.1  # Coordination drifts by a little

    ANTI_PHASE_FUNCTION = lambda x, s: -x if s == 0 else x
    EITHER_PHASE_FUNCTION = lambda x, s: np.abs(x)

    data_table = {
        "trial": [],
        "estimation_name": [],
        "aggregation": [],
        "mean": [],
        "variance": [],
        "means": [],
        "variances": [],
        "score": []
    }

    if not os.path.exists(trials_dir):
        raise Exception(f"Directory {trials_dir} does not exist.")

    filepaths = list(glob(f"{trials_dir}/T*"))
    for i, filepath in tqdm(enumerate(filepaths)):
        if os.path.isdir(filepath):
            trial = Trial.from_directory(filepath)

            configure_log(True, f"{logs_dir}/{trial.metadata.number}")
            logging.getLogger().setLevel(logging.INFO)
            vocalics_component = VocalicsComponent.from_vocalics(trial.vocalics,
                                                                 segmentation_method=SegmentationMethod.KEEP_ALL)

            vocalic_series = vocalics_component.sparse_series(NUM_TIME_STEPS, trial.metadata.mission_start)
            vocalic_series.normalize_per_subject()

            inference_engines = [
                (
                    "continuous_in_phase_fixed_second_half",
                    "last",
                    ContinuousCoordinationInferenceFromVocalics(vocalic_series=vocalic_series,
                                                                mean_prior_coordination=MEAN_COORDINATION_PRIOR,
                                                                std_prior_coordination=STD_COORDINATION_PRIOR,
                                                                std_coordination_drifting=STD_COORDINATION_DRIFT,
                                                                mean_prior_vocalics=MEAN_PRIOR_VOCALICS,
                                                                std_prior_vocalics=STD_PRIOR_VOCALICS,
                                                                std_coordinated_vocalics=STD_COORDINATED_VOCALICS)
                ),
                (
                    "continuous_anti_phase_fixed_second_half",
                    "last",
                    ContinuousCoordinationInferenceFromVocalics(vocalic_series=vocalic_series,
                                                                mean_prior_coordination=MEAN_COORDINATION_PRIOR,
                                                                std_prior_coordination=STD_COORDINATION_PRIOR,
                                                                std_coordination_drifting=STD_COORDINATION_DRIFT,
                                                                mean_prior_vocalics=MEAN_PRIOR_VOCALICS,
                                                                std_prior_vocalics=STD_PRIOR_VOCALICS,
                                                                std_coordinated_vocalics=STD_COORDINATED_VOCALICS,
                                                                f=ANTI_PHASE_FUNCTION)
                ),
                (
                    "continuous_either_phase_fixed_second_half",
                    "last",
                    ContinuousCoordinationInferenceFromVocalics(vocalic_series=vocalic_series,
                                                                mean_prior_coordination=MEAN_COORDINATION_PRIOR,
                                                                std_prior_coordination=STD_COORDINATION_PRIOR,
                                                                std_coordination_drifting=STD_COORDINATION_DRIFT,
                                                                mean_prior_vocalics=MEAN_PRIOR_VOCALICS,
                                                                std_prior_vocalics=STD_PRIOR_VOCALICS,
                                                                std_coordinated_vocalics=STD_COORDINATED_VOCALICS,
                                                                f=EITHER_PHASE_FUNCTION)
                ),
                (
                    "continuous_in_phase_variable",
                    "all",
                    ContinuousCoordinationInferenceFromVocalics(vocalic_series=vocalic_series,
                                                                mean_prior_coordination=MEAN_COORDINATION_PRIOR,
                                                                std_prior_coordination=STD_COORDINATION_PRIOR,
                                                                std_coordination_drifting=STD_COORDINATION_DRIFT,
                                                                mean_prior_vocalics=MEAN_PRIOR_VOCALICS,
                                                                std_prior_vocalics=STD_PRIOR_VOCALICS,
                                                                std_coordinated_vocalics=STD_COORDINATED_VOCALICS,
                                                                fix_coordination_on_second_half=False)
                ),
                (
                    "continuous_anti_phase_variable",
                    "all",
                    ContinuousCoordinationInferenceFromVocalics(vocalic_series=vocalic_series,
                                                                mean_prior_coordination=MEAN_COORDINATION_PRIOR,
                                                                std_prior_coordination=STD_COORDINATION_PRIOR,
                                                                std_coordination_drifting=STD_COORDINATION_DRIFT,
                                                                mean_prior_vocalics=MEAN_PRIOR_VOCALICS,
                                                                std_prior_vocalics=STD_PRIOR_VOCALICS,
                                                                std_coordinated_vocalics=STD_COORDINATED_VOCALICS,
                                                                f=ANTI_PHASE_FUNCTION,
                                                                fix_coordination_on_second_half=False)
                ),
                (
                    "continuous_either_phase_variable",
                    "all",
                    ContinuousCoordinationInferenceFromVocalics(vocalic_series=vocalic_series,
                                                                mean_prior_coordination=MEAN_COORDINATION_PRIOR,
                                                                std_prior_coordination=STD_COORDINATION_PRIOR,
                                                                std_coordination_drifting=STD_COORDINATION_DRIFT,
                                                                mean_prior_vocalics=MEAN_PRIOR_VOCALICS,
                                                                std_prior_vocalics=STD_PRIOR_VOCALICS,
                                                                std_coordinated_vocalics=STD_COORDINATED_VOCALICS,
                                                                f=EITHER_PHASE_FUNCTION,
                                                                fix_coordination_on_second_half=False)
                )
            ]

            for estimation_name, aggregation, inference_engine in inference_engines:
                params = inference_engine.estimate_means_and_variances()
                means = params[0]
                variances = params[1]

                mean = means[-1] if aggregation == "last" else np.mean(means)
                variance = variances[-1] if aggregation == "last" else np.var(means)

                data_table["trial"].append(trial.metadata.number)
                data_table["estimation_name"].append(estimation_name)
                data_table["aggregation"].append(aggregation)
                data_table["mean"].append(mean)
                data_table["variance"].append(variance)
                data_table["means"].append(means)
                data_table["variances"].append(variances)
                data_table["score"].append(trial.metadata.team_score)

                if plot_coordination:
                    filepath = f"{plots_dir}/{estimation_name}/{trial.metadata.number}.png"
                    plot(means, np.sqrt(variances), vocalic_series.mask, filepath)

    df = pd.DataFrame(data_table)
    df.to_pickle(f"{data_dir}/score_regression_data.pkl")


def plot(means: np.ndarray, stds: np.ndarray, masks: List[int], filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    fig = plt.figure(figsize=(20, 6))
    plt.plot(range(len(means)), means, marker="o", color="tab:orange", linestyle="--")
    plt.fill_between(range(len(means)), means - stds, means + stds, color='tab:orange', alpha=0.2)
    times, masks = list(zip(*[(t, mask) for t, mask in enumerate(masks) if mask > 0 and t < len(means)]))
    plt.scatter(times, masks, color="tab:green", marker="+")
    plt.xlabel("Time Steps (seconds)")
    plt.ylabel("Coordination")
    add_discrete_coordination_bar(main_ax=fig.gca(),
                                  coordination_series=[np.where(means > 0.5, 1, 0)],
                                  coordination_colors=["tab:orange"],
                                  labels=["Coordination"])

    plt.savefig(filepath, format="png")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Estimates coordination and plot estimates from a list of trials."
    )
    parser.add_argument("--trials_dir", type=str, required=True, help="Directory where serialized trials are located.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory where the data must be saved.")
    parser.add_argument("--plot_coordination", action="store_true", required=False, default=False,
                        help="Whether plots must be generated. If so, they will be saved under data_dir/plots.")

    args = parser.parse_args()

    estimate(args.trials_dir, args.data_dir, args.plot_coordination)
