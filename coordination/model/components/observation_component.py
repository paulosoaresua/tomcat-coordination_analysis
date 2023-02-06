from typing import Any, List, Optional

import numpy as np
import pymc as pm
from scipy.stats import norm

from coordination.common.utils import set_random_seed
from coordination.model.parametrization import Parameter, HalfNormalParameterPrior


class ObservationComponentParameters:

    def __init__(self, sd_sd_o: np.ndarray):
        self.sd_o = Parameter(HalfNormalParameterPrior(sd_sd_o))

    def clear_values(self):
        self.sd_o.value = None


class ObservationComponentSamples:

    def __init__(self):
        self.values = np.array([])

        # 1 for time steps in which there are observations, 0 otherwise.
        self.mask = np.array([])


class ObservationComponent:

    def __init__(self, uuid: str, num_subjects: int, dim_value: int, sd_sd_o: np.ndarray):
        assert (num_subjects, dim_value) == sd_sd_o.shape

        self.uuid = uuid
        self.num_subjects = num_subjects
        self.dim_value = dim_value

        self.parameters = ObservationComponentParameters(sd_sd_o)

    def draw_samples(self, seed: Optional[int], latent_component: np.ndarray,
                     latent_mask: np.ndarray) -> ObservationComponentSamples:
        assert latent_component.shape[1:3] == self.parameters.sd_o.value.shape

        set_random_seed(seed)

        samples = ObservationComponentSamples()

        M = latent_mask[:, None, None, :]

        samples.values = norm(loc=latent_component, scale=self.parameters.sd_o.value[None, :, :, None]).rvs(
            size=latent_component.shape) * M
        samples.mask = latent_mask

        return samples

    def update_pymc_model(self, latent_component: Any, observed_values: Any) -> Any:
        sd_o = pm.HalfNormal(name=f"sd_o_{self.uuid}", sigma=self.parameters.sd_o.prior.sd,
                             size=(self.num_subjects, self.dim_value), observed=self.parameters.sd_o.value)

        observation_component = pm.Normal(name=self.uuid, mu=latent_component, sigma=sd_o[:, :, None],
                                          observed=observed_values)

        return observation_component
