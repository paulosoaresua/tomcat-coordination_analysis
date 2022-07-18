import json
from glob import glob

from .trial import Trial
from .vocalics_aggregator import VocalicsAggregator
from .vocalics_writer import VocalicsWriter


def write_vocalics_series(metadata_dir: str, vocalics_database: str, output_dir: str) -> None:
    for metadata_path in glob(metadata_dir + "/*"):
        trial = Trial(metadata_path, database=vocalics_database)
        vocalics_component = VocalicsAggregator(trial.utterances_per_subject).split(
            VocalicsAggregator.SplitMethod.TRUNCATE_CURRENT)

        time_steps = int(
            (trial.mission_end - trial.mission_start).total_seconds())
        vocalics_writer = VocalicsWriter()

        vocalics_writer.write(f"{output_dir}/{trial.number}", vocalics_component,
                              trial.mission_start, time_steps)

        trial_info = {
            "id": trial.id,
            "trial_start": trial.trial_start.strftime("%Y-%m-%d %H:%M:%S.%fZ"),
            "trial_end": trial.trial_end.strftime("%Y-%m-%d %H:%M:%S.%fZ"),
            "mission_start": trial.mission_start.strftime("%Y-%m-%d %H:%M:%S.%fZ"),
            "mission_end": trial.mission_end.strftime("%Y-%m-%d %H:%M:%S.%fZ"),
            "team_score": trial.team_score
        }

        with open(f"{output_dir}/{trial.number}/trial_info.json", "w") as f:
            json.dump(trial_info, f, indent=4, sort_keys=True)
