import argparse
from glob import glob
import numpy as np
import os
import pickle

from tqdm import tqdm

from coordination.model.coordination_blending_latent_vocalics import LatentVocalicsDataset, LatentVocalicsDataSeries
from coordination.common.log import configure_log
from coordination.component.speech.vocalics_component import SegmentationMethod, VocalicsComponent
from coordination.entity.trial import Trial
from coordination.loader.metadata_reader import SurveyMappings


def serialize_dataset(trials_dir: str, out_dir: str, time_steps: int, no_overlap: bool):
    if not os.path.exists(trials_dir):
        raise Exception(f"Directory {trials_dir} does not exist.")

    logs_dir = f"{out_dir}/logs"
    os.makedirs(logs_dir, exist_ok=True)

    trials = glob(f"{trials_dir}/T*")
    num_trials = len(trials)

    mission1_series = []
    mission2_series = []
    all_missions_series = []

    mission1_scores = []
    mission2_scores = []
    all_missions_scores = []

    mission1_process_surveys = []
    mission2_process_surveys = []
    all_missions_process_surveys = []

    mission1_satisfaction_surveys = []
    mission2_satisfaction_surveys = []
    all_missions_satisfaction_surveys = []

    mission1_genders = []
    mission2_genders = []
    all_missions_genders = []

    mission1_ages = []
    mission2_ages = []
    all_missions_ages = []

    avatar_colors = ["red", "blue", "green"]

    gender_map = {"M": 0, "F": 1, "NB": 2, "PNA": 3}

    for i, trial_dir in tqdm(enumerate(trials), desc="Trial:"):
        trial = Trial.from_directory(trial_dir)

        player_per_color = {player.avatar_color: player for player in trial.metadata.subject_id_map.values()}

        configure_log(True, f"{logs_dir}/{trial.metadata.number}.txt")
        segmentation = SegmentationMethod.TRUNCATE_CURRENT if no_overlap else SegmentationMethod.KEEP_ALL
        vocalics_component = VocalicsComponent.from_vocalics(trial.vocalics, segmentation_method=segmentation)

        vocalic_series = vocalics_component.sparse_series(time_steps, trial.metadata.mission_start)
        vocalic_series.normalize_per_subject()

        genders = np.zeros(3)
        ages = np.zeros(3)
        process_surveys = np.zeros((3, len(SurveyMappings.PROCESS_SCALE)))
        satisfaction_surveys = np.zeros((3, len(SurveyMappings.TEAM_SATISFACTION)))
        for j, avatar_color in enumerate(avatar_colors):
            player = player_per_color[avatar_color]
            genders[j] = gender_map[player.gender]
            ages[j] = player.age
            process_surveys[j] = player.team_process_scale_survey_answers
            satisfaction_surveys[j] = player.team_satisfaction_survey_answers

        if trial.order == 1:
            mission1_series.append(
                LatentVocalicsDataSeries(observed_vocalics=vocalic_series, uuid=trial.metadata.number))
            mission1_scores.append(trial.metadata.team_score)
            mission1_genders.append(genders)
            mission1_ages.append(ages)
            mission1_process_surveys.append(process_surveys)
            mission1_satisfaction_surveys.append(satisfaction_surveys)
        else:
            mission2_series.append(
                LatentVocalicsDataSeries(observed_vocalics=vocalic_series, uuid=trial.metadata.number))
            mission2_scores.append(trial.metadata.team_score)
            mission2_genders.append(genders)
            mission2_ages.append(ages)
            mission2_process_surveys.append(process_surveys)
            mission2_satisfaction_surveys.append(satisfaction_surveys)

        all_missions_series.append(
            LatentVocalicsDataSeries(observed_vocalics=vocalic_series, uuid=trial.metadata.number))
        all_missions_scores.append(trial.metadata.team_score)
        all_missions_genders.append(genders)
        all_missions_ages.append(ages)
        all_missions_process_surveys.append(process_surveys)
        all_missions_satisfaction_surveys.append(satisfaction_surveys)

    mission1_scores = np.array(mission1_scores)
    mission1_genders = np.array(mission1_genders)
    mission1_ages = np.array(mission1_ages)
    mission1_process_surveys = np.array(mission1_process_surveys)
    mission1_satisfaction_surveys = np.array(mission1_satisfaction_surveys)

    mission2_scores = np.array(mission2_scores)
    mission2_genders = np.array(mission2_genders)
    mission2_ages = np.array(mission2_ages)
    mission2_process_surveys = np.array(mission2_process_surveys)
    mission2_satisfaction_surveys = np.array(mission2_satisfaction_surveys)

    all_missions_scores = np.array(all_missions_scores)
    all_missions_genders = np.array(all_missions_genders)
    all_missions_ages = np.array(all_missions_ages)
    all_missions_process_surveys = np.array(all_missions_process_surveys)
    all_missions_satisfaction_surveys = np.array(all_missions_satisfaction_surveys)

    mission1_dataset = LatentVocalicsDataset(series=mission1_series, team_scores=mission1_scores,
                                             team_process_surveys=mission1_process_surveys,
                                             team_satisfaction_surveys=mission1_satisfaction_surveys,
                                             genders=mission1_genders, ages=mission1_ages)
    mission2_dataset = LatentVocalicsDataset(series=mission2_series, team_scores=mission2_scores,
                                             team_process_surveys=mission2_process_surveys,
                                             team_satisfaction_surveys=mission2_satisfaction_surveys,
                                             genders=mission2_genders, ages=mission2_ages)
    all_missions_dataset = LatentVocalicsDataset(series=all_missions_series, team_scores=all_missions_scores,
                                                 team_process_surveys=all_missions_process_surveys,
                                                 team_satisfaction_surveys=all_missions_satisfaction_surveys,
                                                 genders=all_missions_genders, ages=all_missions_ages)

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
