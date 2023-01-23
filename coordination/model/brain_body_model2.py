from __future__ import annotations
from typing import Any, Optional, Tuple

import arviz as az
import numpy as np
import pymc as pm
import pytensor.tensor as pt

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


class BrainBodySeries:

    def __init__(self, obs_brain: np.ndarray, brain_mask: np.ndarray, brain_prev_time: np.ndarray,
                 obs_body: np.ndarray, body_mask: np.ndarray, body_prev_time: np.ndarray):
        self.obs_brain = obs_brain
        self.brain_mask = brain_mask
        self.brain_prev_time = brain_prev_time
        self.obs_body = obs_body
        self.body_mask = body_mask
        self.body_prev_time = body_prev_time

    @property
    def num_time_steps(self) -> int:
        return self.obs_brain.shape[-1]

    @property
    def num_brain_channels(self) -> int:
        return self.obs_brain.shape[-2]

    @property
    def num_subjects(self) -> int:
        return self.obs_brain.shape[-3]

    @property
    def brain_prev_time_mask(self) -> np.ndarray:
        return np.where(self.brain_prev_time == -1, 0, 1)

    @property
    def body_prev_time_mask(self) -> np.ndarray:
        return np.where(self.body_prev_time == -1, 0, 1)


class BrainBodyInferenceSummary:

    def __init__(self):
        self.unbounded_coordination_means = np.array([])
        self.coordination_means = np.array([])
        self.latent_brain_means = np.array([])
        self.latent_body_means = np.array([])

        self.unbounded_coordination_sds = np.array([])
        self.coordination_sds = np.array([])
        self.latent_brain_sds = np.array([])
        self.latent_body_sds = np.array([])

    @classmethod
    def from_inference_data(cls, idata: Any, retain_every: int = 1) -> BrainBodyInferenceSummary:
        summary = cls()

        if "unbounded_coordination" in idata.posterior:
            summary.unbounded_coordination_means = idata.posterior["unbounded_coordination"][::retain_every].mean(
                dim=["chain", "draw"]).to_numpy()
            summary.unbounded_coordination_sds = idata.posterior["unbounded_coordination"][::retain_every].std(
                dim=["chain", "draw"]).to_numpy()

        if "coordination" in idata.posterior:
            summary.coordination_means = idata.posterior["coordination"][::retain_every].mean(
                dim=["chain", "draw"]).to_numpy()
            summary.coordination_sds = idata.posterior["coordination"][::retain_every].std(
                dim=["chain", "draw"]).to_numpy()

        if "latent_brain" in idata.posterior:
            summary.latent_brain_means = idata.posterior["latent_brain"][::retain_every].mean(
                dim=["chain", "draw"]).to_numpy()
            summary.latent_brain_sds = idata.posterior["latent_brain"][::retain_every].std(
                dim=["chain", "draw"]).to_numpy()

        if "latent_body" in idata.posterior:
            summary.latent_body_means = idata.posterior["latent_body"][::retain_every].mean(
                dim=["chain", "draw"]).to_numpy()
            summary.latent_body_sds = idata.posterior["latent_body"][::retain_every].std(
                dim=["chain", "draw"]).to_numpy()

        return summary


class BrainBodyModel:

    def __init__(self, initial_coordination: float, num_subjects: int, num_brain_channels: int,
                 self_dependent: bool = True):
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

    def fit(self, evidence: BrainBodySeries, burn_in: int, num_samples: int, num_chains: int,
            seed: Optional[int], num_jobs: int = 1) -> Tuple[pm.Model, az.InferenceData]:
        assert evidence.num_subjects == self.num_subjects
        assert evidence.num_brain_channels == self.num_brain_channels

        coords = {"subject": np.arange(self.num_subjects),
                  "brain_channel": np.arange(self.num_brain_channels), "body_feature": np.arange(1),
                  "time": np.arange(evidence.num_time_steps)}

        model = pm.Model(coords=coords)
        with model:
            _, coordination = self.coordination_cpn.update_pymc_model(time_dimension="time")
            latent_brain = self.latent_brain_cpn.update_pymc_model(coordination, pt.constant(evidence.brain_prev_time),
                                                                   pt.constant(evidence.brain_prev_time_mask),
                                                                   pt.constant(evidence.brain_mask),
                                                                   "subject", "brain_channel", "time")
            latent_body = self.latent_body_cpn.update_pymc_model(coordination, pt.constant(evidence.body_prev_time),
                                                                 pt.constant(evidence.body_prev_time_mask),
                                                                 pt.constant(evidence.body_mask),
                                                                 "subject", "body_feature", "time")
            self.obs_brain_cpn.update_pymc_model(latent_brain, [self.num_subjects, self.num_brain_channels], evidence.obs_brain)
            self.obs_body_cpn.update_pymc_model(latent_body, [self.num_subjects, 1], evidence.obs_body)

            idata = pm.sample(num_samples, init="adapt_diag", tune=burn_in, chains=num_chains, random_seed=seed,
                              cores=num_jobs)

        return model, idata
