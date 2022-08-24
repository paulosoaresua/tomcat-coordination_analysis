from typing import Any, Optional, Tuple

import numpy as np
import random

from scipy.stats import norm

from coordination.common.sparse_series import SparseSeries


class VocalicsGenerator:
    """
    This class generates synthetic evidence for the vocalics component of a coordination model.
    """

    def __init__(self, coordination_series: np.ndarray, num_vocalic_features: int, time_scale_density: float):
        self._coordination_series = coordination_series
        self._num_vocalic_features = num_vocalic_features
        self._time_scale_density = time_scale_density

    def generate(self, seed: Optional[int] = None) -> Tuple[SparseSeries, SparseSeries]:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        mask_a, mask_b = self._generate_random_masks()

        num_time_steps = len(self._coordination_series)
        values_a = np.zeros((self._num_vocalic_features, num_time_steps))
        values_b = np.zeros((self._num_vocalic_features, num_time_steps))

        previous_a = None
        previous_b = None
        for t in range(num_time_steps):
            current_coordination = self._coordination_series[t]
            current_a = None
            current_b = None

            if mask_a[t] == 1:
                current_a = self._sample_a(previous_a, previous_b, current_coordination)
                values_a[:, t] = current_a

            if mask_b[t] == 1:
                current_b = self._sample_b(previous_b, previous_a, current_coordination)
                values_b[:, t] = current_b

            previous_a = current_a if current_a is not None else previous_a
            previous_b = current_b if current_b is not None else previous_b

        return SparseSeries(values_a, mask_a), SparseSeries(values_b, mask_b)

    def _generate_random_masks(self):
        """
        Generates random time steps in which series A and B have data available
        """

        num_time_steps = len(self._coordination_series)
        num_selected_time_steps = int(num_time_steps * self._time_scale_density)
        selected_time_steps = sorted(random.sample(range(num_time_steps), num_selected_time_steps))

        # The selected time steps are split between series A and B
        mask_a = np.zeros(num_time_steps)
        mask_b = np.zeros(num_time_steps)

        for i, t in enumerate(selected_time_steps):
            if i % 2 == 0:
                mask_a[t] = 1
            else:
                mask_b[t] = 1

        return mask_a, mask_b

    def _sample_a(self, previous_a: Optional[float], previous_b: Optional[float], coordination: float):
        raise Exception("Not implemented in this class.")

    def _sample_b(self, previous_b: Optional[float], previous_a: Optional[float], coordination: float):
        raise Exception("Not implemented in this class.")


class VocalicsGeneratorForDiscreteCoordination(VocalicsGenerator):

    def __init__(self, mean_prior_a: np.array, mean_prior_b: np.array, std_prior_a: np.array, std_prior_b: np.array,
                 std_uncoordinated_a: np.array, std_uncoordinated_b: np.array, std_coordinated_a: np.array,
                 std_coordinated_b: np.array, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert len(mean_prior_a) == self._num_vocalic_features
        assert len(mean_prior_b) == self._num_vocalic_features
        assert len(std_prior_a) == self._num_vocalic_features
        assert len(std_prior_b) == self._num_vocalic_features
        assert len(std_uncoordinated_a) == self._num_vocalic_features
        assert len(std_uncoordinated_b) == self._num_vocalic_features
        assert len(std_coordinated_a) == self._num_vocalic_features
        assert len(std_coordinated_b) == self._num_vocalic_features

        self._prior_a = norm(loc=mean_prior_a, scale=std_prior_a)
        self._prior_b = norm(loc=mean_prior_b, scale=std_prior_b)

        self._std_uncoordinated_a = std_uncoordinated_a
        self._std_uncoordinated_b = std_uncoordinated_b
        self._std_coordinated_a = std_coordinated_a
        self._std_coordinated_b = std_coordinated_b

    def _sample_a(self, previous_a: Optional[float], previous_b: Optional[float], coordination: float):
        return VocalicsGeneratorForDiscreteCoordination._sample_value(self._prior_a, previous_a, previous_b,
                                                                      int(coordination), self._std_uncoordinated_a,
                                                                      self._std_coordinated_a)

    def _sample_b(self, previous_b: Optional[float], previous_a: Optional[float], coordination: float):
        return VocalicsGeneratorForDiscreteCoordination._sample_value(self._prior_b, previous_b, previous_a,
                                                                      int(coordination), self._std_uncoordinated_b,
                                                                      self._std_coordinated_b)

    @staticmethod
    def _sample_value(prior_distribution: Any, previous_self: Optional[float], previous_other: Optional[float],
                      coordination: int, std_uncoordinated: float, std_coordinated: float):
        if previous_self is None and previous_other is None:
            distribution = prior_distribution
        else:
            if coordination == 0:
                # The current value depends on the previous value of the same series when there's no coordination
                if previous_self is None:
                    distribution = prior_distribution
                else:
                    distribution = norm(loc=previous_self, scale=std_uncoordinated)
            else:
                # The current value depends on the previous value of the other series when there's coordination
                if previous_other is None:
                    distribution = prior_distribution
                else:
                    distribution = norm(loc=previous_other, scale=std_coordinated)

        return distribution.rvs()


class VocalicsGeneratorForContinuousCoordination(VocalicsGenerator):

    def __init__(self, mean_prior_a: np.array, mean_prior_b: np.array, std_prior_a: np.array, std_prior_b: np.array,
                 std_coupling_a: np.array, std_coupling_b: np.array, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert len(mean_prior_a) == self._num_vocalic_features
        assert len(mean_prior_b) == self._num_vocalic_features
        assert len(std_prior_a) == self._num_vocalic_features
        assert len(std_prior_b) == self._num_vocalic_features
        assert len(std_coupling_a) == self._num_vocalic_features
        assert len(std_coupling_b) == self._num_vocalic_features

        self._prior_a = norm(loc=mean_prior_a, scale=std_prior_a)
        self._prior_b = norm(loc=mean_prior_b, scale=std_prior_b)

        self._std_coupling_a = std_coupling_a
        self._std_coupling_b = std_coupling_b

    def _sample_a(self, previous_a: Optional[float], previous_b: Optional[float], coordination: float):
        return VocalicsGeneratorForContinuousCoordination._sample_value(self._prior_a, previous_a, previous_b,
                                                                        coordination, self._std_coupling_a)

    def _sample_b(self, previous_b: Optional[float], previous_a: Optional[float], coordination: float):
        return VocalicsGeneratorForContinuousCoordination._sample_value(self._prior_b, previous_b, previous_a,
                                                                        coordination, self._std_coupling_b)

    @staticmethod
    def _sample_value(prior_distribution: Any, previous_self: Optional[float], previous_other: Optional[float],
                      coordination: float, std_coupling: float):
        if previous_self is not None and previous_other is not None:
            D = previous_other - previous_self
            distribution = norm(loc=D * coordination + previous_self, scale=std_coupling)
        else:
            distribution = prior_distribution

        return distribution.rvs()
