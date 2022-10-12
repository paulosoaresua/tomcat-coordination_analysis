from typing import Dict, Optional

import numpy as np
from scipy.special import expit
from scipy.stats import norm

from coordination.inference.gaussian_coordination_blending_latent_vocalics import \
    GaussianCoordinationBlendingInferenceLatentVocalics


class LogisticCoordinationBlendingInferenceLatentVocalics(GaussianCoordinationBlendingInferenceLatentVocalics):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def estimate_means_and_variances(self) -> np.ndarray:
        M = int(self._time_steps / 2)
        num_time_steps = M + 1 if self._fix_coordination_on_second_half else self._time_steps

        params = np.zeros((2, num_time_steps))
        for t in range(0, self._time_steps):
            self.next()

            # We keep generating latent vocalics after M but not coordination. The fixed coordination is given by
            # the set of particles after the last latent vocalics was generated
            real_time = min(t, M) if self._fix_coordination_on_second_half else t
            mean = expit(self.states[-1].coordination).mean()
            variance = expit(self.states[-1].coordination).var()
            params[:, real_time] = [mean, variance]

        return params

    def _transform_coordination(self, coordination_particles: np.ndarray) -> np.ndarray:
        return expit(coordination_particles)
