from typing import Dict, List, Tuple
from scipy.stats import norm
import random


class VocalicsGenerator:
    """
    This class generates synthetic evidence for the vocalics component of a coordination model.
    """

    def __init__(self, coordination_series: List[float], vocalic_features: List[str], time_scale_density: float):
        self._coordination_series = coordination_series
        self._vocalic_features = vocalic_features
        self._time_scale_density = time_scale_density

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
                        last_sampled_a = self._sample_a(feature_name, last_sampled_a, last_sampled_b, c)
                        series_a[feature_name].append(last_sampled_a)
                        idx_a += 1
                    else:
                        series_a[feature_name].append(None)
                else:
                    series_a[feature_name].append(None)

                if idx_b < len(time_steps_b):
                    if time_steps_b[idx_b] == t:
                        last_sampled_b = self._sample_b(feature_name, last_sampled_b, last_sampled_a, c)
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

    def _sample_a(self, feature_name: str, previous_a: float, previous_b: float, coordination: float):
        raise Exception("Not implemented in this class.")

    def _sample_b(self, feature_name: str, previous_b: float, previous_a: float, coordination: float):
        raise Exception("Not implemented in this class.")


class VocalicsGeneratorForDiscreteCoordination(VocalicsGenerator):

    def __init__(self, coordination_series: List[float], vocalic_features: List[str], time_scale_density: float,
                 mean_prior: float = 0, std_prior: float = 1, mean_shift_coupled: float = 0, var_coupled: float = 1):
        super().__init__(coordination_series, vocalic_features, time_scale_density)
        self._mean_prior = mean_prior
        self._std_prior = std_prior
        self._mean_shift_coupled = mean_shift_coupled
        self._var_coupled = var_coupled

    def _sample_a(self, feature_name: str, previous_a: float, previous_b: float, coordination: float):
        def sample_from_prior():
            return norm.rvs(loc=self._mean_prior, scale=self._std_prior)

        if previous_b is None:
            return sample_from_prior()
        else:
            if int(coordination) == 0:
                return sample_from_prior()
            else:
                return norm.rvs(loc=previous_b + self._mean_shift_coupled, scale=self._var_coupled)

    def _sample_b(self, feature_name: str, previous_b: float, previous_a: float, coordination: float):
        def sample_from_prior():
            return norm.rvs(loc=self._mean_prior, scale=self._std_prior)

        if previous_a is None:
            return sample_from_prior()
        else:
            if int(coordination) == 0:
                return sample_from_prior()
            else:
                return norm.rvs(loc=previous_a + self._mean_shift_coupled, scale=self._var_coupled)


class VocalicsGeneratorForContinuousCoordination(VocalicsGenerator):

    def __init__(self, coordination_series: List[float], vocalic_features: List[str], time_scale_density: float,
                 mean_prior: float = 0, std_prior: float = 1, mean_shift_coupled: float = 0, var_coupled: float = 1):
        super().__init__(coordination_series, vocalic_features, time_scale_density)
        self._mean_prior = mean_prior
        self._std_prior = std_prior
        self._mean_shift_coupled = mean_shift_coupled
        self._var_coupled = var_coupled

    def _sample_a(self, feature_name: str, previous_a: float, previous_b: float, coordination: float):
        def sample_from_prior():
            return norm.rvs(loc=self._mean_prior, scale=self._std_prior)

        if previous_b is None:
            return sample_from_prior()
        else:
            mean = (1 - coordination) * self._mean_prior + coordination * (previous_b + self._mean_shift_coupled)
            return norm.rvs(loc=mean, scale=self._var_coupled)

    def _sample_b(self, feature_name: str, previous_b: float, previous_a: float, coordination: float):
        def sample_from_prior():
            return norm.rvs(loc=self._mean_prior, scale=self._std_prior)

        if previous_a is None:
            return sample_from_prior()
        else:
            mean = (1 - coordination) * self._mean_prior + coordination * (previous_a + self._mean_shift_coupled)
            return norm.rvs(loc=mean, scale=self._var_coupled)
