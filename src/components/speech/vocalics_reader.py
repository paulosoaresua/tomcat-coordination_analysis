from typing import Any, List, Tuple, Dict

import psycopg2


class Vocalics:

    def __init__(self, timestamp: Any, features: Dict[str, float]):
        self.timestamp = timestamp
        self.features = features


class VocalicsReader:
    """
    This class extracts vocalic features from a trial.
    """

    FEATURE_MAP = {
        "pitch": "f0final_sma",
        "intensity": "wave_rmsenergy_sma"
    }

    def __init__(self, server: str = 'localhost', port: int = 5432, database: str = 'asist_vocalics'):
        self._server = server
        self._port = port
        self._database = database
        self.vocalics_per_player = {}

    def read(self, trial_id: str, initial_timestamp: Any, final_timestamp: Any, feature_names: List[str]):
        with self._connect() as connection:
            records = VocalicsReader._read_features(connection, trial_id, initial_timestamp, final_timestamp,
                                                    feature_names)
            for player_id, timestamp, features_record in records:
                feature_map = {}
                for i, feature_name in enumerate(feature_names):
                    feature_map[feature_name] = features_record[i]

                if player_id not in self.vocalics_per_player:
                    self.vocalics_per_player[player_id] = [Vocalics(timestamp, feature_map)]
                else:
                    self.vocalics_per_player[player_id].append(Vocalics(timestamp, feature_map))

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
    def _read_features(connection: Any, trial_id: str, initial_timestamp: Any, final_timestamp: Any,
                       feature_names: List[str]) -> List[
        Tuple[Any]]:

        db_feature_list = ",".join([VocalicsReader.FEATURE_MAP[feature_name] for feature_name in feature_names])

        query = f"SELECT participant, timestamp, {db_feature_list} FROM features WHERE trial_id = %s AND " \
                "timestamp >= %s AND timestamp <= %s ORDER BY participant, timestamp"

        cursor = connection.cursor()
        cursor.execute(query, (trial_id, initial_timestamp, final_timestamp))
        return cursor.fetchall()

