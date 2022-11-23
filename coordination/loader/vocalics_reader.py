from typing import Any, Dict, List, Optional, Tuple

from datetime import datetime, timedelta
import logging

from tqdm import tqdm
import numpy as np

from coordination.entity.trial_metadata import TrialMetadata
from coordination.entity.vocalics_series import VocalicsSeries

logger = logging.getLogger()


class VocalicFeature:
    """
    List of available vocalic features
    """
    PITCH = "pitch"
    INTENSITY = "intensity"


class VocalicsReader:
    """
    This class is an abstract class to read vocalic features from a specific source
    """

    _FEATURE_MAP = {
        "pitch": "f0final_sma",
        "intensity": "wave_rmsenergy_sma"
    }

    def __init__(self, features: List[str]):
        self.features = features

    def read(self,
             trial_metadata: TrialMetadata,
             baseline_time: datetime,
             time_range: Optional[Tuple[datetime, datetime]] = None) -> Dict[str, VocalicsSeries]:
        """
        Reads vocalic features series from a specific trial.
        """

        vocalics_series_per_subject: Dict[str, VocalicsSeries] = {}

        records = self._read_records(trial_metadata, time_range)

        if len(records) > 0:
            # Records are already sorted by timestamp
            vocalics_per_subject: Dict[str, Tuple[List[float], List[datetime]]] = {}
            pbar = tqdm(total=len(records), desc="Parsing vocalics")
            for subject_id, seconds_offset, *feature_values in records:
                subject_id = trial_metadata.subject_id_map[subject_id].id

                timestamp = baseline_time + timedelta(seconds=seconds_offset)

                values = []  # List with one value per feature
                for feature_value in feature_values:
                    values.append(feature_value)

                if subject_id not in vocalics_per_subject:
                    vocalics_per_subject[subject_id] = ([], [])

                vocalics_per_subject[subject_id][0].append(np.array(values).reshape(-1, 1))
                vocalics_per_subject[subject_id][1].append(timestamp)

                pbar.update()

            # From list to numpy
            for subject_id in vocalics_per_subject.keys():
                vocalics_series_per_subject[subject_id] = VocalicsSeries(
                    np.concatenate(vocalics_per_subject[subject_id][0], axis=1),
                    vocalics_per_subject[subject_id][1])
        else:
            logger.error(f"No vocalic features was found for trial {trial_metadata.number}.")

        return vocalics_series_per_subject

    def _read_records(self, trial_metadata: TrialMetadata, time_range: Optional[Tuple[float, float]]) -> List[
        Tuple[Any]]:
        raise NotImplementedError
