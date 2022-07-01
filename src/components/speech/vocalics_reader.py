from datetime import datetime
from typing import Any, Dict, List, Tuple

import psycopg2
from dateutil.parser import parse

from .common import Vocalics


class VocalicsReader:
    """
    This class extracts vocalic features from a trial.
    """
    FEATURE_MAP = {
        "pitch": "f0final_sma",
        "intensity": "wave_rmsenergy_sma"
    }

    LEN_TIMESTAMP_STRING = 30

    def __init__(self,
                 server: str = 'localhost',
                 port: int = 5432,
                 database: str = 'asist_vocalics') -> None:
        self._server = server
        self._port = port
        self._database = database

    def read(self,
             trial_id: str,
             initial_timestamp: datetime,
             final_timestamp: datetime,
             feature_names: List[str]) -> Dict[str, List[Vocalics]]:
        vocalics_per_subject = {}

        with self._connect() as connection:
            records = VocalicsReader._read_features(connection,
                                                    trial_id,
                                                    initial_timestamp,
                                                    final_timestamp,
                                                    feature_names)
            for player_id, timestamp, *features_record in records:
                feature_map = {}
                for i, feature_name in enumerate(feature_names):
                    feature_map[feature_name] = features_record[i]

                if player_id not in vocalics_per_subject:
                    vocalics_per_subject[player_id] = [
                        Vocalics(parse(timestamp), feature_map)]
                else:
                    vocalics_per_subject[player_id].append(
                        Vocalics(parse(timestamp), feature_map))

        return vocalics_per_subject

    def _connect(self) -> Any:
        connection = None
        try:
            connection = psycopg2.connect(host=self._server,
                                          port=self._port,
                                          database=self._database)
        except (Exception, psycopg2.Error) as error:
            raise RuntimeError(
                "Error while connecting to the database: " + str(error))

        return connection

    @staticmethod
    def _read_features(connection: Any,
                       trial_id: str,
                       initial_timestamp: datetime,
                       final_timestamp: datetime,
                       feature_names: List[str]) -> List[Tuple[Any]]:

        db_feature_list = ",".join(
            [VocalicsReader.FEATURE_MAP[feature_name] for feature_name in feature_names])

        query = f"SELECT participant, timestamp, {db_feature_list} FROM features WHERE trial_id = %s AND " + \
            "timestamp >= %s AND timestamp <= %s ORDER BY participant, timestamp"

        # Transform isoformat of python into timestamp string to match database timestamp
        initial_timestamp_formatted = initial_timestamp.isoformat().ljust(
            VocalicsReader.LEN_TIMESTAMP_STRING - 1, '0') + 'Z'
        final_timestamp_formatted = final_timestamp.isoformat().ljust(
            VocalicsReader.LEN_TIMESTAMP_STRING - 1, '0') + 'Z'

        cursor = connection.cursor()
        cursor.execute(
            query, (trial_id, initial_timestamp_formatted, final_timestamp_formatted))
        return cursor.fetchall()
