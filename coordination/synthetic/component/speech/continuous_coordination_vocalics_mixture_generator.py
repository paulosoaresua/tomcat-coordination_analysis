from typing import Optional

import numpy as np

from scipy.stats import norm

from coordination.synthetic.component.speech.vocalics_generator import VocalicsGenerator


class ContinuousCoordinationVocalicsMixtureGenerator(VocalicsGenerator):

    def __init__(self, mean_prior_vocalics: np.array, std_prior_vocalics: np.array,
                 std_uncoordinated_vocalics: np.array, std_coordinated_vocalics: np.array,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert len(mean_prior_vocalics) == self._num_vocalic_features
        assert len(std_prior_vocalics) == self._num_vocalic_features
        assert len(std_uncoordinated_vocalics) == self._num_vocalic_features
        assert len(std_coordinated_vocalics) == self._num_vocalic_features

        self._prior_vocalics = norm(loc=mean_prior_vocalics, scale=std_prior_vocalics)

        self._std_prior_vocalics = std_prior_vocalics
        self._mean_prior_vocalics = mean_prior_vocalics
        self._std_uncoordinated_vocalics = std_uncoordinated_vocalics
        self._std_coordinated_vocalics = std_coordinated_vocalics

    def _sample(self, previous_self: Optional[float], previous_other: Optional[float], coordination: float):
        if previous_other is None:
            distribution = self._prior_vocalics
        else:
            if previous_self is None:
                if np.random.rand() <= coordination:
                    distribution = norm(loc=previous_other,
                                        scale=self._std_coordinated_vocalics)
                else:
                    distribution = norm(loc=self._mean_prior_vocalics,
                                        scale=self._std_prior_vocalics)
            else:
                if np.random.rand() <= coordination:
                    distribution = norm(loc=previous_other,
                                        scale=self._std_coordinated_vocalics)
                else:
                    distribution = norm(loc=previous_self,
                                        scale=self._std_uncoordinated_vocalics)

        return distribution.rvs()
