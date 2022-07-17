import glob

from src.components.speech.trial import Trial
from src.components.speech.vocalics_aggregator import VocalicsAggregator

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

            print("A")

        break
