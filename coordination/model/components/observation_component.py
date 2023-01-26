from typing import Any, List, Optional

import numpy as np
import pymc as pm
from scipy.stats import norm

from coordination.common.utils import set_random_seed


class ObservationComponentParameters:

    def __init__(self):
        self.sd_o = None

    def reset(self):
        self.sd_o = None


class ObservationComponentSamples:

    def __init__(self):
        self.values = np.array([])

        # 1 for time steps in which there are observations, 0 otherwise.
        self.mask = np.array([])


class ObservationComponent:

    def __init__(self, uuid: str):
        self.uuid = uuid

        self.parameters = ObservationComponentParameters()

    def draw_samples(self, seed: Optional[int], latent_component: np.ndarray,
                     latent_mask: np.ndarray) -> ObservationComponentSamples:
        # assert latent_component.shape[1] == self.parameters.sd_o.shape[0]
        # assert latent_component.shape[2] == self.parameters.sd_o.shape[1]

        set_random_seed(seed)

        samples = ObservationComponentSamples()

        M = latent_mask[:, None, None, :]

        # samples.values = norm(loc=latent_component, scale=self.parameters.sd_o[None, :, :, None]).rvs(
        #     size=latent_component.shape) * M
        samples.values = norm(loc=latent_component, scale=self.parameters.sd_o).rvs(
            size=latent_component.shape) * M
        samples.mask = latent_mask

        return samples

    def update_pymc_model(self, latent_component: Any, parameter_size: List[int], observed_values: Any) -> Any:
        # sd_o = pm.HalfNormal(name=f"sd_o_{self.uuid}", sigma=1, size=parameter_size, observed=self.parameters.sd_o)

        # observation_component = pm.Normal(name=self.uuid, mu=latent_component, sigma=sd_o[:, :, None],
        #                                   observed=observed_values)

        sd_o = pm.HalfNormal(name=f"sd_o_{self.uuid}", sigma=0.5, size=1, observed=self.parameters.sd_o)
        # sd_o = pm.Flat(name=f"sd_o_{self.uuid}", size=1, observed=self.parameters.sd_o, initval=np.array([1]))

        observation_component = pm.Normal(name=self.uuid, mu=latent_component, sigma=sd_o,
                                          observed=observed_values)

        return observation_component
