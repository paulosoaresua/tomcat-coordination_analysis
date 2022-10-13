import os.path
from typing import Any, List, Optional, Tuple

from glob import glob
import logging
import os

import pandas as pd

from coordination.entity.trial_metadata import TrialMetadata
from coordination.loader.vocalics_reader import VocalicsReader

logger = logging.getLogger()


class VocalicsReaderCSV(VocalicsReader):
    """
    This class reads vocalic features from a database
    """

    __FEATURE_MAP = {
        "pitch": "F0final_sma",
        "intensity": "pcm_RMSenergy_sma"
    }

    def __init__(self, vocalics_dir: str, features: List[str]):
        """
        @param vocalics_dir: directory containing a list of trials and csv files with the vocalics for each subject.
        """
        super().__init__(features)
        self._vocalics_dir = vocalics_dir

    def _read_records(self, trial_metadata: TrialMetadata, time_range: Optional[Tuple[float, float]]) -> List[
        Tuple[Any]]:

        df_trial = None
        for vocalics_file in glob(f"{self._vocalics_dir}/{trial_metadata.number}/*.csv"):
            df_subject = pd.read_csv(vocalics_file, delimiter=";")

            if time_range is not None:
                df_subject = df_subject[
                    (df_subject["frameTime"] >= time_range[0]) & (df_subject["frameTime"] <= time_range[1])]

            subject_id = os.path.basename(vocalics_file)
            subject_id = subject_id[:subject_id.rfind(".")]
            df_subject["participant"] = subject_id
            cols = ["participant", "frameTime"] + [self.__FEATURE_MAP[feature] for feature in self.features]

            if df_trial is None:
                df_trial = df_subject.loc[:, cols]
            else:
                df_trial = pd.concat([df_trial, df_subject.loc[:, cols]], axis=0)

        return list(df_trial.itertuples(index=False, name=None))
