from typing import Dict, List

from .common import Utterance, Vocalics
from .utterance_metadata_parser import utterance_metadata_parser
from .vocalics_reader import VocalicsReader


def extract_utterances(vocalics_reader: VocalicsReader,
                       metadata_path: str,
                       keep_utterance_no_vocalics: bool = True,
                       print_missing: bool = True) -> Dict[str, List[Utterance]]:
    utterances_per_subject = {}

    num_utterances = 0
    num_missing_utterances = 0

    for trial_id, subject_id, text, start_timestamp, end_timestamp in utterance_metadata_parser(metadata_path):
        num_utterances += 1

        # Extract vocalics for the target subject utterance
        vocalics_per_subject = vocalics_reader.read(trial_id=trial_id,
                                                    initial_timestamp=start_timestamp,
                                                    final_timestamp=end_timestamp,
                                                    feature_names=["pitch", "intensity"])

        if subject_id in vocalics_per_subject:
            vocalics = vocalics_per_subject[subject_id]

            # Compute average vocalics
            average_pitch = 0.0
            average_intensity = 0.0
            for vocalic in vocalics:
                average_pitch += vocalic.features["pitch"]
                average_intensity += vocalic.features["intensity"]
            num_vocalics = float(len(vocalics))
            average_pitch /= num_vocalics
            average_intensity /= num_vocalics
        else:
            if print_missing:
                print("Missing vocalic features for subject " + subject_id + " from " +
                      start_timestamp + " to " + end_timestamp + " (only see " + str(list(vocalics_per_subject.keys())) + ") in trial " + trial_id + ". Text: " + text)

            num_missing_utterances += 1

            if not keep_utterance_no_vocalics:
                continue

            vocalics = []
            average_pitch = None
            average_intensity = None

        average_vocalics = {
            "pitch": average_pitch,
            "intensity": average_intensity
        }

        utterance = Utterance(subject_id,
                              start_timestamp,
                              end_timestamp,
                              text, vocalics,
                              average_vocalics)

        if subject_id not in utterances_per_subject:
            utterances_per_subject[subject_id] = []

        utterances_per_subject[subject_id].append(utterance)

    print("Total " + str(num_utterances) + " utterances, missing " +
          str(num_missing_utterances) + " utterances")

    return utterances_per_subject
