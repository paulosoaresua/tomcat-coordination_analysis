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
    def write(out_dir: str, vocalics_component: VocalicsComponent, initial_timestamp: Any, time_steps: int):
        assert len(vocalics_component.series_a) == len(vocalics_component.series_b)

        # TODO check to see if vocalics time step is bigger than time steps

        # We consider that vocalics data is available to the model at the end of an utterance
        series_a_out: Dict[str, List[Any]] = {feature_name: [None] * time_steps for feature_name in
                                              vocalics_component.feature_names}
        series_b_out: Dict[str, List[Any]] = {feature_name: [None] * time_steps for feature_name in
                                              vocalics_component.feature_names}

        num_vocalics = len(vocalics_component.series_a)

        if num_vocalics > 0:
            for i in range(num_vocalics):
                time_step_a = int((vocalics_component.series_a[i].end - initial_timestamp).total_seconds())
                time_step_b = int((vocalics_component.series_b[i].end - initial_timestamp).total_seconds())
                for feature_name in vocalics_component.feature_names:
                    # Data available in the end of an utterance from series A
                    series_a_out[feature_name][time_step_a] = vocalics_component.series_a[i].average_vocalics[
                        feature_name]
                    series_b_out[feature_name][time_step_b] = vocalics_component.series_b[i].average_vocalics[
                        feature_name]

        final_dir = f"{out_dir}/vocalics"
        os.makedirs(final_dir, exist_ok=True)

        with open(f"{final_dir}/series_a.pkl", "wb") as f:
            pickle.dump(series_a_out, f)

        with open(f"{final_dir}/series_b.pkl", "wb") as f:
            pickle.dump(series_b_out, f)
