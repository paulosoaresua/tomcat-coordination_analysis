from typing import Optional

import numpy as np

from scipy.stats import norm

from coordination.synthetic.component.speech.latent_vocalics_generator import LatentVocalicsGenerator


class ContinuousCoordinationLatentVocalicsBlendingGenerator(LatentVocalicsGenerator):

    def __init__(self, mean_prior_latent_vocalics: np.array, std_prior_latent_vocalics: np.array,
                 std_coordinated_vocalics: np.array, std_observed_vocalics: np.array,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert len(mean_prior_latent_vocalics) == self._num_vocalic_features
        assert len(std_prior_latent_vocalics) == self._num_vocalic_features
        assert len(std_coordinated_vocalics) == self._num_vocalic_features
        assert len(std_observed_vocalics) == self._num_vocalic_features

        self._prior_latent_vocalics = norm(loc=mean_prior_latent_vocalics, scale=std_prior_latent_vocalics)

        self._mean_prior_vocalics = mean_prior_latent_vocalics
        self._std_coordinated_vocalics = std_coordinated_vocalics
        self._std_observed_vocalics = std_observed_vocalics

    def _sample_latent(self, previous_self: Optional[float], previous_other: Optional[float],
                       coordination: float) -> np.ndarray:
        if previous_other is None:
            distribution = self._prior_latent_vocalics
        else:
            if previous_self is None:
                D = previous_other - self._mean_prior_vocalics
                distribution = norm(loc=D * coordination + self._mean_prior_vocalics,
                                    scale=self._std_coordinated_vocalics)
            else:
                D = previous_other - previous_self
                distribution = norm(loc=D * coordination + previous_self,
                                    scale=self._std_coordinated_vocalics)

        return distribution.rvs()

    def _sample_observed(self, latent_vocalics: np.array) -> np.ndarray:
        return norm(loc=latent_vocalics, scale=self._std_observed_vocalics).rvs()
