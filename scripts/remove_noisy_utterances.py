from typing import List

import argparse
import json

from tqdm import tqdm

from coordination.entity.trial import Trial
from coordination.entity.vocalics import Utterance


def remove_noisy_utterances(input_dir: str, out_dir: str, utterances_json_path: str):
    with open(utterances_json_path, "r") as f:
        utterances_json = json.load(f)

        for trial_json in tqdm(utterances_json["trials"], desc="Trials:"):
            trial_number = trial_json["number"]
            trial = Trial.from_directory(f"{input_dir}/{trial_number}")

            utterances: List[Utterance] = []
            for u in trial.vocalics.utterances_per_subject.values():
                utterances.extend(u)
            utterances.sort(key=lambda utterance: utterance.start)

            for i in reversed(sorted(trial_json["utterances"])):
                del utterances[i - 1]

            trial.vocalics.utterances_per_subject = {}

            for utterance in utterances:
                if utterance.subject_id not in trial.vocalics.utterances_per_subject:
                    trial.vocalics.utterances_per_subject[utterance.subject_id] = []

                trial.vocalics.utterances_per_subject[utterance.subject_id].append(utterance)

            trial.save(out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Removes utterances from serialized trials."
    )
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory where serialized trial directories are located.")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Directory where new serialized trial must be saved.")
    parser.add_argument("--utterances_json_path", type=str, required=False, default=False,
                        help="JSON file containing a list of trials and utterances to remove from these trials.")

    args = parser.parse_args()

    remove_noisy_utterances(args.input_dir, args.out_dir, args.utterances_json_path)
