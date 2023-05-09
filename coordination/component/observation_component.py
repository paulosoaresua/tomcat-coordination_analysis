from typing import Any, Dict, List, Optional

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


class ObservationComponent:

    def __init__(self, uuid: str, num_subjects: int, dim_value: int, sd_sd_o: np.ndarray,
                 share_sd_o_across_subjects: bool, share_sd_o_across_features: bool):

        # Check dimensionality of the parameters priors
        if share_sd_o_across_features:
            dim_sd_o_features = 1
        else:
            dim_sd_o_features = dim_value

        if share_sd_o_across_subjects:
            assert (dim_sd_o_features,) == sd_sd_o.shape
        else:
            assert (num_subjects, dim_sd_o_features) == sd_sd_o.shape

        self.uuid = uuid
        self.num_subjects = num_subjects
        self.dim_value = dim_value
        self.share_sd_o_across_subjects = share_sd_o_across_subjects
        self.share_sd_o_across_features = share_sd_o_across_features

        self.parameters = ObservationComponentParameters(sd_sd_o)

    @property
    def parameter_names(self) -> List[str]:
        return [
            self.sd_o_name
        ]

    @property
    def sd_o_name(self) -> str:
        return f"sd_o_{self.uuid}"

    def draw_samples(self, latent_component: np.ndarray, seed: Optional[int] = None) -> ObservationComponentSamples:
        # Check dimensionality of the parameters
        if self.share_sd_o_across_features:
            dim_sd_o_features = 1
        else:
            dim_sd_o_features = self.dim_value

        if self.share_sd_o_across_subjects:
            assert (dim_sd_o_features,) == self.parameters.sd_o.value.shape
        else:
            assert (self.num_subjects, dim_sd_o_features) == self.parameters.sd_o.value.shape

        set_random_seed(seed)

        samples = ObservationComponentSamples()

        # Dimension: (sample, subject, feature, time)
        if self.share_sd_o_across_subjects:
            # Broadcast across samples, subjects and time
            sd = self.parameters.sd_o.value[None, None, :, None]
        else:
            # Broadcast across samples and time
            sd = self.parameters.sd_o.value[None, :, :, None]

        samples.values = norm(loc=latent_component, scale=sd).rvs(size=latent_component.shape)

        return samples

    def _create_random_parameters(self, sd_o: Optional[Any] = None):
        """
        This function creates the standard deviation of the observation component distribution as a random variable.
        """

        # Adjust feature dimensionality according to sharing options
        if self.share_sd_o_across_features:
            dim_sd_o_features = 1
        else:
            dim_sd_o_features = self.dim_value

        # Initialize sd_o parameter if it hasn't been defined previously
        if sd_o is None:
            if self.share_sd_o_across_subjects:
                sd_o = pm.HalfNormal(name=self.sd_o_name,
                                     sigma=self.parameters.sd_o.prior.sd,
                                     size=dim_sd_o_features,
                                     observed=self.parameters.sd_o.value)
                sd_o = sd_o[None, :, None]  # subject x feature x time (broadcast across subject and time)
            else:
                sd_o = pm.HalfNormal(name=self.sd_o_name,
                                     sigma=self.parameters.sd_o.prior.sd,
                                     size=(self.num_subjects, dim_sd_o_features),
                                     observed=self.parameters.sd_o.value)
                sd_o = sd_o[:, :, None]  # subject x feature x time (broadcast across time)

        return sd_o

    def update_pymc_model(self, latent_component: Any, subject_dimension: str, feature_dimension: str,
                          time_dimension: str, observed_values: Any, sd_o: Optional[Any] = None) -> Any:

        sd_o = self._create_random_parameters(sd_o)

        observation_component = pm.Normal(name=self.uuid,
                                          mu=latent_component,
                                          sigma=sd_o,
                                          dims=[subject_dimension, feature_dimension, time_dimension],
                                          observed=observed_values)

        return observation_component, sd_o


class SerializedObservationComponentSamples:

    def __init__(self):
        self.values: List[np.ndarray] = []


class SerializedObservationComponent:

    def __init__(self, uuid: str, num_subjects: int, dim_value: int, sd_sd_o: np.ndarray,
                 share_sd_o_across_subjects: bool, share_sd_o_across_features: bool):

        # Check dimensionality of the parameters priors
        if share_sd_o_across_features:
            dim_sd_o_features = 1
        else:
            dim_sd_o_features = dim_value

        if share_sd_o_across_subjects:
            assert (dim_sd_o_features,) == sd_sd_o.shape
        else:
            assert (num_subjects, dim_sd_o_features) == sd_sd_o.shape

        self.uuid = uuid
        self.num_subjects = num_subjects
        self.dim_value = dim_value
        self.share_sd_o_across_subjects = share_sd_o_across_subjects
        self.share_sd_o_across_features = share_sd_o_across_features

        self.parameters = ObservationComponentParameters(sd_sd_o)

    @property
    def parameter_names(self) -> List[str]:
        return [
            self.sd_o_name
        ]

    @property
    def sd_o_name(self) -> str:
        return f"sd_o_{self.uuid}"

    def draw_samples(self, latent_component: List[np.ndarray], subjects: List[np.ndarray],
                     seed: Optional[int] = None) -> SerializedObservationComponentSamples:

        # Check dimensionality of the parameters
        if self.share_sd_o_across_features:
            dim_sd_o_features = 1
        else:
            dim_sd_o_features = self.dim_value

        if self.share_sd_o_across_subjects:
            assert (dim_sd_o_features,) == self.parameters.sd_o.value.shape
        else:
            assert (self.num_subjects, dim_sd_o_features) == self.parameters.sd_o.value.shape

        set_random_seed(seed)

        samples = SerializedObservationComponentSamples()

        for i in range(len(latent_component)):
            if self.share_sd_o_across_subjects:
                # Broadcasted across time
                sd = self.parameters.sd_o.value[:, None]
            else:
                sd = self.parameters.sd_o.value[subjects[i]].T

            samples.values.append(norm(loc=latent_component[i], scale=sd).rvs(size=latent_component[i].shape))

        return samples

    def _create_random_parameters(self, subjects: np.ndarray, sd_o: Optional[Any] = None):
        """
        This function creates the standard deviation of the serialized observation component distribution as
        a random variable.
        """

        # Adjust feature dimensionality according to sharing options
        if self.share_sd_o_across_features:
            dim_sd_o_features = 1
        else:
            dim_sd_o_features = self.dim_value

        # Initialize sd_o parameter if it hasn't been defined previously
        if sd_o is None:
            if self.share_sd_o_across_subjects:
                sd_o = pm.HalfNormal(name=self.sd_o_name,
                                     sigma=self.parameters.sd_o.prior.sd,
                                     size=dim_sd_o_features,
                                     observed=self.parameters.sd_o.value)
                sd_o = sd_o[:, None]  # feature x time = 1 (broadcast across time)
            else:
                sd_o = pm.HalfNormal(name=self.sd_o_name,
                                     sigma=self.parameters.sd_o.prior.sd,
                                     size=(self.num_subjects, dim_sd_o_features),
                                     observed=self.parameters.sd_o.value)
                sd_o = sd_o[subjects].transpose()  # feature x time

            if self.share_sd_o_across_features:
                sd_o = sd_o.repeat(self.dim_value, axis=0)

        return sd_o

    def update_pymc_model(self, latent_component: Any, subjects: np.ndarray, feature_dimension: str,
                          time_dimension: str, observed_values: Any, sd_o: Optional[Any] = None) -> Any:

        sd_o = self._create_random_parameters(subjects=subjects, sd_o=sd_o)

        observation_component = pm.Normal(name=self.uuid,
                                          mu=latent_component,
                                          sigma=sd_o,
                                          dims=[feature_dimension, time_dimension],
                                          observed=observed_values)

        return observation_component
