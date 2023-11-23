from typing import Any, List, Optional

import numpy as np
import pymc as pm
from scipy.stats import norm

from coordination.common.utils import set_random_seed
from coordination.module.observation import Observation


class SerialObservationSamples:
    def __init__(self):
        self.values: List[np.ndarray] = []


class SerialObservation(Observation):
    def __init__(
        self,
        uuid: str,
        num_subjects: int,
        dim_value: int,
        sd_sd_o: np.ndarray,
        share_sd_o_across_subjects: bool,
        share_sd_o_across_features: bool,
    ):
        super().__init__(
            uuid=uuid,
            num_subjects=num_subjects,
            dim_value=dim_value,
            sd_sd_o=sd_sd_o,
            share_sd_o_across_subjects=share_sd_o_across_subjects,
            share_sd_o_across_features=share_sd_o_across_features,
        )

    def draw_samples(
        self,
        latent_component: List[np.ndarray],
        subjects: List[np.ndarray],
        seed: Optional[int] = None,
    ) -> SerialObservationSamples:
        # Check dimensionality of the parameters
        if self.share_sd_o_across_features:
            dim_sd_o_features = 1
        else:
            dim_sd_o_features = self.dim_value

        if self.share_sd_o_across_subjects:
            assert (dim_sd_o_features,) == self.parameters.sd_o.value.shape
        else:
            assert (
                self.num_subjects,
                dim_sd_o_features,
            ) == self.parameters.sd_o.value.shape

        # Generate samples
        set_random_seed(seed)

        samples = SerialObservationSamples()

        for i in range(len(latent_component)):
            # Adjust dimensions according to parameter sharing specification
            if self.share_sd_o_across_subjects:
                # Broadcast across time
                sd = self.parameters.sd_o.value[:, None]
            else:
                sd = self.parameters.sd_o.value[subjects[i]].T

            samples.values.append(
                norm(loc=latent_component[i], scale=sd).rvs(
                    size=latent_component[i].shape
                )
            )

        return samples

    def _create_random_parameters(
        self, subjects: np.ndarray, sd_o: Optional[Any] = None
    ):
        """
        This function creates the standard deviation of the serialized observation component
        distribution as a random variable.
        """

        # Adjust feature dimensionality according to sharing options
        if self.share_sd_o_across_features:
            dim_sd_o_features = 1
        else:
            dim_sd_o_features = self.dim_value

        # Initialize sd_o parameter if it hasn't been defined previously
        if sd_o is None:
            if self.share_sd_o_across_subjects:
                sd_o = pm.HalfNormal(
                    name=self.sd_o_name,
                    sigma=self.parameters.sd_o.prior.sd,
                    size=dim_sd_o_features,
                    observed=self.parameters.sd_o.value,
                )
                sd_o = sd_o[:, None]  # feature x time = 1 (broadcast across time)
            else:
                sd_o = pm.HalfNormal(
                    name=self.sd_o_name,
                    sigma=self.parameters.sd_o.prior.sd,
                    size=(self.num_subjects, dim_sd_o_features),
                    observed=self.parameters.sd_o.value,
                )
                sd_o = sd_o[subjects].transpose()  # feature x time

            if self.share_sd_o_across_features:
                sd_o = sd_o.repeat(self.dim_value, axis=0)

        return sd_o

    def update_pymc_model(
        self,
        latent_component: Any,
        subjects: np.ndarray,
        feature_dimension: str,
        time_dimension: str,
        observed_values: Any,
        sd_o: Optional[Any] = None,
    ) -> Any:
        sd_o = self._create_random_parameters(subjects=subjects, sd_o=sd_o)

        observation_component = pm.Normal(
            name=self.uuid,
            mu=latent_component,
            sigma=sd_o,
            dims=[feature_dimension, time_dimension],
            observed=observed_values,
        )

        return observation_component
