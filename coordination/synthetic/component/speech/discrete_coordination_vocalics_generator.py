from typing import Optional

import numpy as np

from scipy.stats import norm

from coordination.synthetic.component.speech.vocalics_generator import VocalicsGenerator


class DiscreteCoordinationVocalicsGenerator(VocalicsGenerator):

    def __init__(self, mean_prior_vocalics: np.array, std_prior_vocalics: np.array,
                 std_uncoordinated_vocalics: np.array, std_coordinated_vocalics: np.array, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert len(mean_prior_vocalics) == self._num_vocalic_features
        assert len(std_prior_vocalics) == self._num_vocalic_features
        assert len(std_uncoordinated_vocalics) == self._num_vocalic_features
        assert len(std_coordinated_vocalics) == self._num_vocalic_features

        self._prior_vocalics = norm(loc=mean_prior_vocalics, scale=std_prior_vocalics)

        self._std_uncoordinated_vocalics = std_uncoordinated_vocalics
        self._std_coordinated_vocalics = std_coordinated_vocalics

    def _sample(self, previous_self: Optional[float], previous_other: Optional[float],
                coordination: float) -> np.ndarray:
        if previous_other is None:
            distribution = self._prior_vocalics
        else:
            if coordination == 0:
                # The current value depends on the previous value of the same series when there's no coordination
                if previous_self is None:
                    distribution = self._prior_vocalics
                else:
                    distribution = norm(loc=previous_self, scale=self._std_uncoordinated_vocalics)
            else:
                # The current value depends on the previous value of the other series when there's coordination
                if previous_other is None:
                    distribution = self._prior_vocalics
                else:
                    distribution = norm(loc=previous_other, scale=self._std_coordinated_vocalics)

        return distribution.rvs()
