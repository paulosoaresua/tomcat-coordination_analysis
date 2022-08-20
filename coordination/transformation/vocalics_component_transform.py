from typing import List, Tuple

from datetime import datetime
import logging
import numpy as np

from coordination.component.speech.vocalics_component import SegmentedUtterance, VocalicsComponent
from coordination.entity.sparse_series import SparseSeries

logger = logging.getLogger()


def to_seconds_scale(num_time_steps: int, vocalics_component: VocalicsComponent) -> Tuple[SparseSeries, SparseSeries]:
    def series_to_seconds(utterances: List[SegmentedUtterance], initial_timestamp: datetime) -> SparseSeries:
        values = np.zeros((utterances[0].vocalic_series.num_series, num_time_steps))
        mask = np.zeros(num_time_steps)  # 1 for time steps with observation, 0 otherwise

        for i, utterance in enumerate(utterances):
            # We consider that the observation is available at the end of an utterance. We take the average vocalics
            # per feature within the utterance as a measurement at the respective time step.
            time_step = int((utterance.end - initial_timestamp).total_seconds())
            if time_step >= num_time_steps:
                logger.warning(f"""Time step {time_step} exceeds the number of time steps {num_time_steps} at utterance
                               {i} out of {len(utterances)} ending at {utterance.end.isoformat()} considering an 
                               initial timestamp of {initial_timestamp.isoformat()}.""")
                break

            values[:, time_step] = utterance.vocalic_series.values.mean(axis=1)
            mask[time_step] = 1

        return SparseSeries(values, mask)

    # The first utterance always goes in series A
    earliest_timestamp = vocalics_component.series_a[0].start
    sparse_series_a = series_to_seconds(vocalics_component.series_a, earliest_timestamp)
    sparse_series_b = series_to_seconds(vocalics_component.series_b, earliest_timestamp)

    return sparse_series_a, sparse_series_b
