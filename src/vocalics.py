from typing import Any, List, Tuple

import psycopg2

from src.trial import Trial


class Vocalics:

    FEATURE_MAP = {
        "pitch": "f0final_sma",
        "intensity": "wave_rmsenergy_sma"
    }

    def __init__(self, server: str = 'localhost', port: int = 5432, database: str = 'asist_vocalics'):
        self._server = server
        self._port = port
        self._database = database
        self.features = {}

    def read_features(self, trial: Trial, feature_names: List[str]):
        with self._connect() as connection:
            records = Vocalics._read_features(connection, trial, feature_names)
            for participant, timestamp, features_record in records:
                if participant not in self.features:
                    self.features[participant] = {"timestamp": [timestamp]}
                    for i, feature_name in enumerate(feature_names):
                        self.features[participant][feature_name] = [features_record[i]]
                else:
                    self.features[participant]["timestamp"].append(timestamp)
                    for i, feature_name in enumerate(feature_names):
                        self.features[participant][feature_name].append(features_record[i])

    def _connect(self) -> Any:
        connection = None
        try:
            connection = psycopg2.connect(host=self._server,
                                          port=self._port,
                                          database=self._database)
        except (Exception, psycopg2.Error) as error:
            print("Error while connecting to the database", error)

        return connection

    @staticmethod
    def _read_features(connection: Any, trial: Trial, feature_names: List[str]) -> List[Tuple[Any]]:
        db_feature_list = ",".join([Vocalics.FEATURE_MAP[feature_name] for feature_name in feature_names])

        query = f"SELECT participant, timestamp, {db_feature_list} FROM features WHERE trial_id = %s AND " \
                "timestamp >= %s AND timestamp <= %s ORDER BY participant, timestamp"

        cursor = connection.cursor()
        cursor.execute(query, (trial.id, trial.start_timestamp, trial.end_timestamp))
        return cursor.fetchall()

    def get_segmented_vocalics(self, player_id: str, segments: List[Tuple[Any, Any]]):
        """
        Keep only features within the timestamps defined by each segment
        """

        segmented_features = {key: [] for key in self.features[player_id].keys()}
        t = 0
        for segment in segments:
            while self.features[player_id]["timestamp"][t] < segment[0]:
                t += 1

            for key in self.features[player_id].keys():
                if key != "timestamp":
                    features_in_segment = []
                    while self.features[player_id]["timestamp"][t] < segment[1]:
                        features_in_segment.append(self.features[player_id][key][t])

                    segmented_features[key].append(features_in_segment)

            # initial and final utterance timestamp
            segmented_features["timestamp"].append(list(segment))

        return segmented_features

