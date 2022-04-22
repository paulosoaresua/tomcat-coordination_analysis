from typing import Callable, Dict, List, Tuple
import random


class VocalicsGenerator:
    """
    This class generates synthetic evidence for the vocalics component of a coordination model.
    """

    def __init__(self, coordination_series: List[float], vocalic_features: List[str],
                 time_scale_density: float, prior_a: Callable, prior_b: Callable, pa: Callable, pb: Callable):
        self._coordination_series = coordination_series
        self._vocalic_features = vocalic_features
        self._time_scale_density = time_scale_density
        self._prior_a = prior_a
        self._prior_b = prior_b
        self._pa = pa
        self._pb = pb

    def generate_evidence(self) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
        time_steps_a, time_steps_b = self._get_random_observation_time_steps()
        series_a = {feature_name: [] for feature_name in self._vocalic_features}
        series_b = {feature_name: [] for feature_name in self._vocalic_features}

        for feature_name in self._vocalic_features:
            idx_a = 0
            idx_b = 0
            last_sampled_a = None
            last_sampled_b = None
            for t, c in enumerate(self._coordination_series):

                if idx_a < len(time_steps_a):
                    if time_steps_a[idx_a] == t:
                        if idx_a == 0:
                            last_sampled_a = self._prior_a(feature_name)
                            series_a[feature_name].append(last_sampled_a)
                        else:
                            last_sampled_a = self._pa(feature_name, last_sampled_a, last_sampled_b, c)
                            series_a[feature_name].append(last_sampled_a)
                        idx_a += 1
                    else:
                        series_a[feature_name].append(None)
                else:
                    series_a[feature_name].append(None)

                if idx_b < len(time_steps_b):
                    if time_steps_b[idx_b] == t:
                        if idx_b == 0:
                            last_sampled_b = self._prior_b(feature_name)
                            series_b[feature_name].append(last_sampled_b)
                        else:
                            last_sampled_b = self._pb(feature_name, last_sampled_b, last_sampled_a, c)
                            series_b[feature_name].append(last_sampled_b)
                        idx_b += 1
                    else:
                        series_b[feature_name].append(None)
                else:
                    series_b[feature_name].append(None)

        return series_a, series_b

    def _get_random_observation_time_steps(self):
        """
        Generates random time steps in which series A and B have data available
        """
        time_steps = len(self._coordination_series)
        selected_time_steps = sorted(random.sample(range(time_steps), int(time_steps * self._time_scale_density)))

        # The selected time steps are split between series A and B
        time_steps_a = []
        time_steps_b = []
        for i, t in enumerate(selected_time_steps):
            if i % 2 == 0:
                time_steps_a.append(t)
            else:
                time_steps_b.append(t)

        return time_steps_a, time_steps_b
