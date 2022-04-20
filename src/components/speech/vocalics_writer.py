from typing import Any, List, Tuple, Dict

from src.components.speech.common import SegmentedUtterance, VocalicsComponent


class VocalicsWriter:
    """
    This class writes vocalics component series to a file, converting the timestamps of segments utterances
    appropriately.
    """

    @staticmethod
    def write(out_dir: str, vocalics_component: VocalicsComponent, initial_timestamp: Any, rate_milliseconds: float):
        series_a_out = []
        series_b_out = []
        mask = [] # Keeps track of which time steps have data

        if vocalics_component.size > 0:
            timestamp = initial_timestamp
            while timestamp < vocalics_component.series_a[0].
            for i in range(vocalics_component.size):










    def __init__(self, vocalics_component: VocalicsComponent):
        self._vocalics_component
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
