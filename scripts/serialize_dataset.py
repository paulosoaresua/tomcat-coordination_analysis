import argparse
from glob import glob
import os
import pickle

from tqdm import tqdm

from coordination.model.coordination_blending_latent_vocalics import LatentVocalicsDataset, LatentVocalicsDataSeries
from coordination.common.log import configure_log
from coordination.component.speech.semantics_component import SemanticsComponent
from coordination.component.speech.vocalics_component import SegmentationMethod, VocalicsComponent
from coordination.entity.trial import Trial


def serialize_dataset(trials_dir: str, out_dir: str, time_steps: int, no_overlap: bool, semantic_window_size: int):
    if not os.path.exists(trials_dir):
        raise Exception(f"Directory {trials_dir} does not exist.")

    logs_dir = f"{out_dir}/logs"
    os.makedirs(logs_dir, exist_ok=True)

    trials = glob(f"{trials_dir}/T*")

    mission1_series = []
    mission2_series = []
    all_missions_series = []

    GENDER_MAP = {"M": 0, "F": 1, "NB": 2, "PNA": 3}

    for i, trial_dir in tqdm(enumerate(trials), desc="Trial:"):
        trial = Trial.from_directory(trial_dir)

        configure_log(True, f"{logs_dir}/{trial.metadata.number}.txt")
        segmentation = SegmentationMethod.TRUNCATE_CURRENT if no_overlap else SegmentationMethod.KEEP_ALL
        vocalics_component = VocalicsComponent.from_vocalics(trial.vocalics, segmentation_method=segmentation)
        speech_semantics_component = SemanticsComponent.from_vocalics(trial.vocalics, semantic_window_size)

        observed_vocalics = vocalics_component.sparse_series(time_steps, trial.metadata.mission_start)
        speech_semantic_links = speech_semantics_component.to_array(time_steps, trial.metadata.mission_start)

        genders = {}
        ages = {}
        process_surveys = {}
        satisfaction_surveys = {}
        for player in trial.metadata.subject_id_map.values():
            genders[player.avatar_color] = GENDER_MAP[player.gender]
            ages[player.avatar_color] = player.age
            process_surveys[player.avatar_color] = player.team_process_scale_survey_answers
            satisfaction_surveys[player.avatar_color] = player.team_satisfaction_survey_answers

        series = LatentVocalicsDataSeries(
            uuid=trial.metadata.number,
            observed_vocalics=observed_vocalics,
            team_score=trial.metadata.team_score,
            team_process_surveys=process_surveys,
            team_satisfaction_surveys=satisfaction_surveys,
            genders=genders,
            ages=ages,
            features=trial.vocalics.features,
            speech_semantic_links=speech_semantic_links
        )

        if trial.order == 1:
            mission1_series.append(series)
        else:
            mission2_series.append(series)

        all_missions_series.append(series)

    mission1_dataset = LatentVocalicsDataset(series=mission1_series)
    mission2_dataset = LatentVocalicsDataset(series=mission2_series)
    all_missions_dataset = LatentVocalicsDataset(series=all_missions_series)

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
    parser.add_argument("--semantic_window", type=int, required=False, default=5,
                        help="Window size for semantic link extraction.")

    args = parser.parse_args()
    serialize_dataset(args.trials_dir, args.out_dir, args.time_steps, args.no_overlap, args.semantic_window)
