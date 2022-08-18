import glob
from datetime import datetime

import json
from src.components.speech.trial import Trial
from src.components.speech.vocalics_aggregator import VocalicsAggregator
from src.components.speech.vocalics_writer import VocalicsWriter

if __name__ == "__main__":

    # Trials for which we have vocalics.
    trials = ["T000745", "T000746", "T000837", "T000838", "T000843", "T000844", "T000847", "T000848"]

    for trial_number in trials:
        for metadata_path in glob.glob(f"../data/asist/study3/metadata/*Trial-{trial_number}*.metadata"):
            if "Terminated" in metadata_path:
                continue

            trial = Trial(metadata_path, database="asist_vocalics")

            vocalics_component = VocalicsAggregator(trial.utterances_per_subject).split(
                VocalicsAggregator.SplitMethod.TRUNCATE_CURRENT)

            time_steps = int((trial.mission_end - trial.mission_start).total_seconds())
            vocalics_writer = VocalicsWriter()

            vocalics_writer.write(f"../data/asist/study3/series/{trial.number}", vocalics_component,
                                  trial.mission_start, time_steps)

            trial_info = {
                "id": trial.id,
                "trial_start": trial.trial_start.strftime("%Y-%m-%d %H:%M:%S.%fZ"),
                "trial_end": trial.trial_end.strftime("%Y-%m-%d %H:%M:%S.%fZ"),
                "mission_start": trial.mission_start.strftime("%Y-%m-%d %H:%M:%S.%fZ"),
                "mission_end": trial.mission_end.strftime("%Y-%m-%d %H:%M:%S.%fZ"),
                "team_score": trial.team_score
            }

            with open(f"../data/asist/study3/series/{trial.number}/trial_info.json", "w") as f:
                json.dump(trial_info, f, indent=4, sort_keys=True)



