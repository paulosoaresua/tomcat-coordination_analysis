from __future__ import annotations
from typing import Optional, Tuple

import arviz as az
import numpy as np
import pymc as pm

from coordination.model.components.coordination_component import BetaGaussianCoordinationComponent, \
    BetaGaussianCoordinationComponentSamples
from coordination.model.components.mixture_component import MixtureComponent, MixtureComponentSamples
from coordination.model.components.observation_component import ObservationComponent, ObservationComponentSamples


class BrainBodySamples:

    def __init__(self, coordination: BetaGaussianCoordinationComponentSamples, latent_brain: MixtureComponentSamples,
                 latent_body: MixtureComponentSamples, obs_brain: ObservationComponentSamples,
                 obs_body: ObservationComponentSamples):
        self.coordination = coordination
        self.latent_brain = latent_brain
        self.latent_body = latent_body
        self.obs_brain = obs_brain
        self.obs_body = obs_body


class BrainBodyModel2:

    def __init__(self, initial_coordination: float, num_subjects: int, num_brain_channels: int, self_dependent: bool = True):
        self.num_subjects = num_subjects
        self.num_brain_channels = num_brain_channels

        self.coordination_cpn = BetaGaussianCoordinationComponent(initial_coordination)
        self.latent_brain_cpn = MixtureComponent("latent_brain", num_subjects, num_brain_channels, self_dependent)
        self.latent_body_cpn = MixtureComponent("latent_body", num_subjects, 1, self_dependent)
        self.obs_brain_cpn = ObservationComponent("obs_brain")
        self.obs_body_cpn = ObservationComponent("obs_body")

    def draw_samples(self, num_series: int, num_time_steps: int,
                     seed: Optional[int], brain_relative_frequency: float,
                     body_relative_frequency: float) -> BrainBodySamples:
        coordination_samples = self.coordination_cpn.draw_samples(num_series, num_time_steps, seed)
        latent_brain_samples = self.latent_brain_cpn.draw_samples(num_series, num_time_steps,
                                                                  seed=None,  # Do not restart seed
                                                                  relative_frequency=brain_relative_frequency,
                                                                  coordination=coordination_samples.coordination)
        latent_body_samples = self.latent_body_cpn.draw_samples(num_series, num_time_steps,
                                                                seed=None,  # Do not restart seed
                                                                relative_frequency=body_relative_frequency,
                                                                coordination=coordination_samples.coordination)

        obs_brain_samples = self.obs_brain_cpn.draw_samples(seed=None,  # Do not restart seed
                                                            latent_component=latent_brain_samples.values,
                                                            latent_mask=latent_brain_samples.mask)

        obs_body_samples = self.obs_body_cpn.draw_samples(seed=None,  # Do not restart seed
                                                          latent_component=latent_body_samples.values,
                                                          latent_mask=latent_body_samples.mask)

        samples = BrainBodySamples(coordination_samples, latent_brain_samples, latent_body_samples, obs_brain_samples,
                                   obs_body_samples)

        return samples

    def fit(self, evidence: BrainBodyDataSeries, burn_in: int, num_samples: int, num_chains: int,
            seed: Optional[int], retain_every: int = 1, num_jobs: int = 1) -> Tuple[pm.Model, az.InferenceData]:
        coords = {"subject": np.arange(self.num_subjects),
                  "brain_channel": np.arange(self.num_brain_channels), "body_feature": np.arange(1),
                  "time": np.arange(evidence.num_time_steps)}

        model = pm.Model(coords=coords)
        with model:
            coordination = self.coordination_cpn.update_pymc_model(time_dimension="time")
            latent_brain = self.latent_brain_cpn.update_pymc_model(coordination, ...)
            latent_body = self.latent_body_cpn.update_pymc_model(coordination, ...)
            self.obs_brain_cpn.update_pymc_model(latent_brain, ["subject", "brain_channel"], ...)
            self.obs_body_cpn.update_pymc_model(latent_body, ["subject", "brain_channel"], ...)

            idata = pm.sample(num_samples, init="adapt_diag", tune=burn_in, chains=num_chains, random_seed=seed,
                              cores=num_jobs)

        return model, idata
