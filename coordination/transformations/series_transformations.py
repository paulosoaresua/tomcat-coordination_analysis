from typing import Dict, List, Tuple
import numpy as np


def embed_features_across_dimensions(num_time_steps: int, series: Dict[str, List[float]]) -> Tuple[
    np.array, List[int]]:
    multi_dim_series = np.zeros((num_time_steps, len(series)))
    # The mask array keeps track of time steps with observed values
    mask = [1] * num_time_steps
    for t in range(num_time_steps):
        for i, (key, value) in enumerate(series.items()):
            if value[t] is None:
                mask[t] = 0
                break
            else:
                multi_dim_series[t, i] = value[t]

    return multi_dim_series, mask
