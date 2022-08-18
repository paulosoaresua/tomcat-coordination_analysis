from typing import Any, Dict, List, Optional, Tuple

from datetime import datetime
from dateutil.parser import parse
import logging

import psycopg2
from tqdm import tqdm
import numpy as np

from coordination.config.database_config import DatabaseConfig
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
    This class reads vocalic features from a database
    """

    __FEATURE_MAP = {
        "pitch": "f0final_sma",
        "intensity": "wave_rmsenergy_sma"
    }

    def __init__(self, database_config: DatabaseConfig, features: List[str]):
        self._database_config = database_config
        self._features = features

    def read(self,
             trial_metadata: TrialMetadata,
             baseline_time: datetime,
             time_range: Optional[Tuple[datetime, datetime]] = None) -> Dict[str, VocalicsSeries]:
        """
        Reads vocalic features series from a specific trial.
        """

        vocalics_series_per_subject: Dict[str, VocalicsSeries] = {}

        with self._connect() as connection:
            records = VocalicsReader._read_records(connection, trial_metadata.id, self._features, time_range)

            if len(records) > 0:
                timestamp_offset = None
                if baseline_time is not None:
                    # When vocalics are generated offline, they take the execution data as timestamp. We might want to
                    # correct that by providing a baseline such that the first vocalics matches with that time.
                    earliest_timestamp = VocalicsReader._read_earliest_timestamp(connection, trial_metadata.id)
                    earliest_vocalics_timestamp = parse(earliest_timestamp)
                    timestamp_offset = baseline_time - earliest_vocalics_timestamp

                # Records are already sorted by timestamp
                vocalics_per_subject: Dict[str, Tuple[List[float], List[datetime]]] = {}
                pbar = tqdm(total=len(records), desc="Parsing vocalics")
                for subject_id, timestamp_str, *feature_values in records:
                    subject_id = trial_metadata.subject_id_map[subject_id]

                    timestamp = parse(timestamp_str)
                    if timestamp_offset is not None:
                        timestamp += timestamp_offset

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
                logger.error(f"No vocalic features were found for trial {trial_metadata.number}.")

        return vocalics_series_per_subject

    def _connect(self) -> Any:
        try:
            connection = psycopg2.connect(host=self._database_config.address,
                                          port=self._database_config.port,
                                          database=self._database_config.database)
        except (Exception, psycopg2.Error) as error:
            raise RuntimeError(
                "Error while connecting to the database: " + str(error))

        return connection

    @staticmethod
    def _read_earliest_timestamp(connection: Any,
                                 trial_id: str) -> str:

        query = f"SELECT min(timestamp) FROM features WHERE trial_id = %s"

        cursor = connection.cursor()
        cursor.execute(
            query, (trial_id,))
        return cursor.fetchall()[0][0]

    @staticmethod
    def _read_records(connection: Any, trial_id: str, features: List[str],
                      time_range: Optional[Tuple[datetime, datetime]]) -> List[Tuple[Any]]:

        db_feature_names = [VocalicsReader.__FEATURE_MAP[feature] for feature in features]
        db_feature_list = ",".join(db_feature_names)

        if time_range is None:
            query = f"SELECT participant, timestamp, {db_feature_list} FROM features WHERE " + \
                    "trial_id = %s ORDER BY participant, timestamp"
            cursor = connection.cursor()
            cursor.execute(query, (trial_id,))
        else:
            query = f"SELECT participant, timestamp, {db_feature_list} FROM features WHERE " + \
                    "trial_id = %s AND timestamp BETWEEN %s AND %s ORDER BY participant, timestamp"
            cursor = connection.cursor()
            cursor.execute(query, (trial_id, time_range[0].isoformat(), time_range[1].isoformat(),))

        return cursor.fetchall()
