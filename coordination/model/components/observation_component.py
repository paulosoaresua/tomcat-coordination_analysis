from typing import Any, List, Optional

import numpy as np
import pymc as pm
from scipy.stats import norm
import pytensor.tensor as ptt

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


class ObservationComponent:

    def __init__(self, uuid: str, num_subjects: int, dim_value: int, sd_sd_o: np.ndarray):
        assert (num_subjects, dim_value) == sd_sd_o.shape

        self.uuid = uuid
        self.num_subjects = num_subjects
        self.dim_value = dim_value

        self.parameters = ObservationComponentParameters(sd_sd_o)

    def draw_samples(self, latent_component: np.ndarray, seed: Optional[int] = None) -> ObservationComponentSamples:
        assert (self.num_subjects, self.dim_value) == self.parameters.sd_o.value.shape

        set_random_seed(seed)

        samples = ObservationComponentSamples()

        samples.values = norm(loc=latent_component, scale=self.parameters.sd_o.value[None, :, :, None]).rvs(
            size=latent_component.shape)

        return samples

    def update_pymc_model(self, latent_component: Any, observed_values: Any) -> Any:
        sd_o = pm.HalfNormal(name=f"sd_o_{self.uuid}", sigma=self.parameters.sd_o.prior.sd,
                             size=(self.num_subjects, self.dim_value), observed=self.parameters.sd_o.value)

        observation_component = pm.Normal(name=self.uuid, mu=latent_component, sigma=sd_o[:, :, None],
                                          observed=observed_values)

        return observation_component, sd_o


class SerializedObservationComponentSamples:

    def __init__(self):
        self.values: List[np.ndarray] = []


class SerializedObservationComponent:

    def __init__(self, uuid: str, num_subjects: int, dim_value: int, sd_sd_o: np.ndarray):
        assert (num_subjects, dim_value) == sd_sd_o.shape

        self.uuid = uuid
        self.num_subjects = num_subjects
        self.dim_value = dim_value

        self.parameters = ObservationComponentParameters(sd_sd_o)

    def draw_samples(self, latent_component: List[np.ndarray],
                     subjects: List[np.ndarray], seed: Optional[int] = None) -> SerializedObservationComponentSamples:
        assert (self.num_subjects, self.dim_value) == self.parameters.sd_o.value.shape

        set_random_seed(seed)

        samples = SerializedObservationComponentSamples()

        for s in range(len(latent_component)):
            samples.values.append(norm(loc=latent_component[s], scale=self.parameters.sd_o.value[subjects[s]].T).rvs(
                size=latent_component[s].shape))

        return samples

    def update_pymc_model(self, latent_component: Any, subjects: np.ndarray, observed_values: Any) -> Any:
        sd_o = pm.HalfNormal(name=f"sd_o_{self.uuid}", sigma=self.parameters.sd_o.prior.sd,
                             size=(self.num_subjects, self.dim_value), observed=self.parameters.sd_o.value)

        observation_component = pm.Normal(name=self.uuid, mu=latent_component,
                                          sigma=sd_o[ptt.constant(subjects)].transpose(),
                                          observed=observed_values)

        return observation_component
