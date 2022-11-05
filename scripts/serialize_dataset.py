import argparse
from glob import glob
import numpy as np
import os
import pickle

from tqdm import tqdm

from coordination.common.dataset import EvidenceDataset, SeriesData
from coordination.common.log import configure_log
from coordination.component.speech.vocalics_component import SegmentationMethod, VocalicsComponent
from coordination.entity.trial import Trial


def serialize_dataset(trials_dir: str, out_dir: str, time_steps: int, no_overlap: bool):
    if not os.path.exists(trials_dir):
        raise Exception(f"Directory {trials_dir} does not exist.")

    logs_dir = f"{out_dir}/logs"
    os.makedirs(logs_dir, exist_ok=True)

    mission1_series = []
    mission2_series = []
    all_missions_series = []
    mission1_scores = []
    mission2_scores = []
    all_missions_scores = []

    trials = glob(f"{trials_dir}/T*")
    pbar = tqdm(total=len(trials))
    for trial_dir in trials:
        trial = Trial.from_directory(trial_dir)
        pbar.set_description(f"{trial.metadata.number}")

        configure_log(True, f"{logs_dir}/{trial.metadata.number}.txt")
        segmentation = SegmentationMethod.TRUNCATE_CURRENT if no_overlap else SegmentationMethod.KEEP_ALL
        vocalics_component = VocalicsComponent.from_vocalics(trial.vocalics, segmentation_method=segmentation)

        vocalic_series = vocalics_component.sparse_series(time_steps, trial.metadata.mission_start)
        vocalic_series.normalize_per_subject()

        if trial.order == 1:
            mission1_series.append(SeriesData(vocalic_series, trial.metadata.number))
            mission1_scores.append(trial.metadata.team_score)
        else:
            mission2_series.append(SeriesData(vocalic_series, trial.metadata.number))
            mission2_scores.append(trial.metadata.team_score)

        all_missions_series.append(SeriesData(vocalic_series, trial.metadata.number))
        all_missions_scores.append(trial.metadata.team_score)

        pbar.update()

    mission1_features = EvidenceDataset(mission1_series)
    mission2_features = EvidenceDataset(mission2_series)
    all_missions_features = EvidenceDataset(all_missions_series)

    mission1_dataset = (mission1_features, np.array(mission1_scores))
    mission2_dataset = (mission2_features, np.array(mission2_scores))
    all_missions_dataset = (all_missions_features, np.array(all_missions_scores))

    with open(f"{out_dir}/mission1_dataset.pkl", "wb") as f:
        pickle.dump(mission1_dataset, f)

    with open(f"{out_dir}/mission2_dataset.pkl", "wb") as f:
        pickle.dump(mission2_dataset, f)

    with open(f"{out_dir}/all_missions_dataset.pkl", "wb") as f:
        pickle.dump(all_missions_dataset, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Creates a dataset of vocalics for mission 1, mission 2 and both missions for all the serialized "
                    "trials in a given folder."
    )

    parser.add_argument("--trials_dir", type=str, required=True,
                        help="Directory containing a list of serialized trials.")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Directory directory where the serialized datasets must be saved.")
    parser.add_argument("--time_steps", type=int, required=True,
                        help="Number of time steps (seconds) in each series.")
    parser.add_argument("--no_overlap", action="store_true", required=False, default=False,
                        help="Whether utterances must be trimmed so that they don't overlap.")

    args = parser.parse_args()
    serialize_dataset(args.trials_dir, args.out_dir, args.time_steps, args.no_overlap)
