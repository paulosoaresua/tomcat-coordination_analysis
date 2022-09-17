import argparse
from glob import glob
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from coordination.component.speech.vocalics_component import SegmentationMethod, VocalicsComponent
from coordination.entity.trial import Trial
from coordination.inference.vocalics import ContinuousCoordinationInferenceFromVocalics
from coordination.plot.coordination import add_discrete_coordination_bar
from scripts.utils import configure_log


def plot(trials_dir: str, plots_dir: str):
    logs_dir = f"{plots_dir}/logs"

    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

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

    if not os.path.exists(trials_dir):
        raise Exception(f"Directory {trials_dir} does not exist.")

    filepaths = glob(f"{trials_dir}/T*")
    for i, filepath in tqdm(enumerate(filepaths)):
        if os.path.isdir(filepath):
            trial = Trial.from_directory(filepath)

            configure_log(True, f"{logs_dir}/{trial.metadata.number}")
            logging.getLogger().setLevel(logging.INFO)
            vocalics_component = VocalicsComponent.from_vocalics(trial.vocalics,
                                                                 segmentation_method=SegmentationMethod.KEEP_ALL)

            vocalic_series = vocalics_component.sparse_series(NUM_TIME_STEPS, trial.metadata.mission_start)
            vocalic_series.normalize_per_subject()

            inference_engine = ContinuousCoordinationInferenceFromVocalics(vocalic_series=vocalic_series,
                                                                           mean_prior_coordination=MEAN_COORDINATION_PRIOR,
                                                                           std_prior_coordination=STD_COORDINATION_PRIOR,
                                                                           std_coordination_drifting=STD_COORDINATION_DRIFT,
                                                                           mean_prior_vocalics=MEAN_PRIOR_VOCALICS,
                                                                           std_prior_vocalics=STD_PRIOR_VOCALICS,
                                                                           std_coordinated_vocalics=STD_COORDINATED_VOCALICS)

            params = inference_engine.estimate_means_and_variances()
            mean_cs = params[0];
            var_cs = params[1]
            fig = plt.figure(figsize=(20, 6))
            plt.plot(range(M + 1), mean_cs, marker="o", color="tab:orange", linestyle="--")
            plt.fill_between(range(M + 1), mean_cs - np.sqrt(var_cs), mean_cs + np.sqrt(var_cs), color='tab:orange',
                             alpha=0.2)
            times, masks = list(zip(*[(t, mask) for t, mask in enumerate(vocalic_series.mask) if mask > 0 and t <= M]))
            plt.scatter(times, masks, color="tab:green", marker="+")
            plt.xlabel("Time Steps (seconds)")
            plt.ylabel("Coordination")
            plt.title("Continuous Coordination Inference", fontsize=14, weight="bold")
            add_discrete_coordination_bar(main_ax=fig.gca(),
                                          coordination_series=[np.where(mean_cs > 0.5, 1, 0)],
                                          coordination_colors=["tab:orange"],
                                          labels=["Coordination"])

            plt.savefig(f"{plots_dir}/{trial.metadata.number}.png", format="png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Estimates coordination and plot estimates from a list of trials."
    )
    parser.add_argument("--trials_dir", type=str, required=True, help="Directory where serialized trials are located.")
    parser.add_argument("--plots_dir", type=str, required=True, help="Directory where the plots must be saved.")

    args = parser.parse_args()

    plot(args.trials_dir, args.plots_dir)