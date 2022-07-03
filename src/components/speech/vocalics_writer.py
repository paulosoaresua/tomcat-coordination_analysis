from typing import Any, List, Dict
import pickle
import os

from src.components.speech.common import VocalicsComponent


class VocalicsWriter:
    """
    This class writes vocalics component series to a file, converting the timestamps of segments utterances
    appropriately.
    """

    @staticmethod
    def write(out_dir: str, vocalics_component: VocalicsComponent, initial_timestamp: Any, freq_milliseconds: float):
        # We consider that vocalics data is available to the model at the end of an utterance.
        series_a_out: Dict[str, List[Any]] = {feature_name: [] for feature_name in vocalics_component.feature_names}
        series_b_out: Dict[str, List[Any]] = {feature_name: [] for feature_name in vocalics_component.feature_names}

        if vocalics_component.size > 0:
            timestamp = initial_timestamp
            for i in range(vocalics_component.size):
                diff = vocalics_component.series_a[i].end - timestamp
                time_steps = (diff.total_seconds() * 1000) * freq_milliseconds
                for feature_name in vocalics_component.feature_names:
                    # No data until the end of an utterance from series A
                    series_a_out[feature_name].extend([None] * time_steps)
                    # Data available in the end of an utterance from series A
                    series_a_out[feature_name].append(vocalics_component.series_a[i].average_vocalics[feature_name])

                timestamp = vocalics_component.series_a[i].end
                diff = vocalics_component.series_b[i].end - timestamp
                time_steps = (diff.total_seconds() * 1000) * freq_milliseconds
                for feature_name in vocalics_component.feature_names:
                    # No data until the end of an utterance from series B
                    series_b_out[feature_name].extend([None] * time_steps)
                    # Data available in the end of an utterance from series B
                    series_b_out[feature_name].append(vocalics_component.series_b[i].average_vocalics[feature_name])

            # At this step, evidence from series A is behind B in the number of time steps. We complement it with None
            # to equalize the series sizes.
            for feature_name in vocalics_component.feature_names:
                series_a_out[feature_name].extend(
                    [None] * (len(series_b_out[feature_name]) - len(series_a_out[feature_name])))

        final_dir = f"{out_dir}/vocalics"
        os.makedirs(final_dir)

        with open(f"{final_dir}/series_a.pkl", "wb") as f:
            pickle.dump(series_a_out, f)

        with open(f"{final_dir}/series_b.pkl", "wb") as f:
            pickle.dump(series_b_out, f)
