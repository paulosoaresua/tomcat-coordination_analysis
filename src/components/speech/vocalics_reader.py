from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from dateutil.parser import parse

import psycopg2
from tqdm import tqdm


class Vocalics:
    def __init__(self, timestamp: datetime, features: Dict[str, float]):
        self.timestamp = timestamp
        self.features = features


class VocalicsReader:
    """
    This class reads vocalic features from a database
    """

    def __init__(self, database_address: str, database_port: int, database_name: str):
        self._database_address = database_address
        self._database_port = database_port
        self._database_name = database_name

    def read(self,
             trial_id: str,
             feature_map: Dict[str, str],
             baseline_time: Optional[datetime],
             subject_id_map: Dict[str, str]) -> Dict[str, List[Vocalics]]:
        """
        Reads vocalic features from a specific trial

        Returns:
            A dictionary with one entry per subject in the trial. For each subject, there will be a tuple in which
            the first element is a matrix of feature values (feature vs timestamp) and the second element is a list
            of timestamps.
        """
        vocalics_per_subject = {}

        with self._connect() as connection:
            # Records already sorted by timestamp
            records = VocalicsReader._read_records(connection, trial_id, list(feature_map.values()))

            timestamp_offset = None
            if baseline_time is not None:
                earliest_vocalics_timestamp = parse(VocalicsReader._read_earliest_timestamp(connection, trial_id))
                timestamp_offset = baseline_time - earliest_vocalics_timestamp

            # create vocalics objects
            pbar = tqdm(total=len(records), desc="Parsing vocalics")
            for subject_id, timestamp_str, *feature_values in records:
                subject_id = subject_id_map[subject_id]

                timestamp = parse(timestamp_str)
                if timestamp_offset is not None:
                    timestamp += timestamp_offset

                values = {}
                for i, feature_name in enumerate(feature_map.keys()):
                    values[feature_name] = feature_values[i]

                if subject_id not in vocalics_per_subject:
                    vocalics_per_subject[subject_id] = []

                vocalics = Vocalics(timestamp, values)
                vocalics_per_subject[subject_id].append(vocalics)

                pbar.update()

        return vocalics_per_subject

    def _connect(self) -> Any:
        try:
            connection = psycopg2.connect(host=self._database_address,
                                          port=self._database_port,
                                          database=self._database_name)
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
    def _read_records(connection: Any, trial_id: str, feature_names: List[str]) -> List[Tuple[Any]]:
        db_feature_list = ",".join(feature_names)

        query = f"SELECT participant, timestamp, {db_feature_list} FROM features WHERE " + \
                "trial_id = %s ORDER BY participant, timestamp"

        cursor = connection.cursor()
        cursor.execute(query, (trial_id,))
        return cursor.fetchall()
