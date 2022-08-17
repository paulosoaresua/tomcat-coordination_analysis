from typing import Dict, List
import os
import pickle

from .components.speech.metadata_reader import MetadataReader
from .components.speech.vocalics_reader import VocalicsReader
from .components.speech.pair_speech_vocalics import pair_speech_vocalics, Utterance


class Trial:
    """
    This class encapsulates information about an ASIST trial. It also contains a list of components we will use for
    coordination inference.
    """
    def __init__(self,
                 metadata: MetadataReader,
                 utterances_per_subject: Dict[str, List[Utterance]]) -> None:
        self.metadata = metadata
        self.utterances_per_subject = utterances_per_subject

    @classmethod
    def from_metadata(cls,
                      metadata_file_path: str,
                      vocalics_reader: VocalicsReader,
                      feature_map: Dict[str, str],
                      silent_warnings: bool = False):
        metadata = MetadataReader(MetadataReader.Mode.METADATA, metadata_file_path)

        vocalics_per_subject = vocalics_reader.read(
            trial_id=metadata.id,
            feature_map=feature_map,
            baseline_time=metadata.trial_start,
            subject_id_map=metadata.subject_id_to_color
        )

        utterances_per_subject = pair_speech_vocalics(
            metadata.speeches_per_subject,
            vocalics_per_subject,
            silent_warnings=silent_warnings
        )

        return cls(metadata, utterances_per_subject)

    def save(self, out_dir: str):
        trial_out_dir = f"{out_dir}/{self.metadata.number}"
        os.makedirs(trial_out_dir, exist_ok=True)

        with open(f"{trial_out_dir}/utterances.pkl", "wb") as f:
            pickle.dump(self.utterances_per_subject, f)

        self.metadata.save(trial_out_dir)

    @classmethod
    def load(cls, trial_dir: str):
        metadata = MetadataReader(MetadataReader.Mode.LOAD, trial_dir)

        utterances_path = f"{trial_dir}/utterances.pkl"
        if not os.path.exists(utterances_path):
            raise Exception(f"Could not find the file speeches.pkl in {trial_dir}.")

        with open(utterances_path, "rb") as f:
            utterances_per_subject = pickle.load(f)

        return cls(metadata, utterances_per_subject)
