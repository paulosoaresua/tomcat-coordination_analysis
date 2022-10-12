from typing import Any, List, Optional, Tuple

import logging

import psycopg2

from coordination.config.database_config import DatabaseConfig
from coordination.entity.trial_metadata import TrialMetadata
from coordination.loader.vocalics_reader import VocalicsReader

logger = logging.getLogger()


class VocalicsReaderDB(VocalicsReader):
    """
    This class reads vocalic features from a database
    """

    __FEATURE_MAP = {
        "pitch": "f0final_sma",
        "intensity": "wave_rmsenergy_sma"
    }

    def __init__(self, database_config: DatabaseConfig, features: List[str]):
        super().__init__(features)
        self._database_config = database_config

    def _connect(self) -> Any:
        try:
            connection = psycopg2.connect(host=self._database_config.address,
                                          port=self._database_config.port,
                                          database=self._database_config.database)
        except (Exception, psycopg2.Error) as error:
            raise RuntimeError(
                "Error while connecting to the database: " + str(error))

        return connection

    def _read_records(self, trial_metadata: TrialMetadata, time_range: Optional[Tuple[float, float]]) -> List[
        Tuple[Any]]:

        with self._connect() as connection:
            db_feature_names = [self.__FEATURE_MAP[feature] for feature in self.features]
            db_feature_list = ",".join(db_feature_names)

            if time_range is None:
                query = f"SELECT participant, seconds_offset, {db_feature_list} FROM features WHERE " + \
                        "trial_id = %s ORDER BY participant, seconds_offset"
                cursor = connection.cursor()
                cursor.execute(query, (trial_metadata.id,))
            else:
                query = f"SELECT participant, seconds_offset, {db_feature_list} FROM features WHERE " + \
                        "trial_id = %s AND seconds_offset BETWEEN %f AND %f ORDER BY participant, seconds_offset"
                cursor = connection.cursor()
                cursor.execute(query, (trial_metadata.id, time_range[0], time_range[1],))

            return cursor.fetchall()
