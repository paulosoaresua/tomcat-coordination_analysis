from __future__ import annotations
from typing import Any, Optional, Tuple

import arviz as az
import numpy as np
import pymc as pm

from coordination.model.components.coordination_component import SigmoidGaussianCoordinationComponent, \
    SigmoidGaussianCoordinationComponentSamples
from coordination.model.components.mixture_component import MixtureComponent, MixtureComponentSamples
from coordination.model.components.observation_component import ObservationComponent, ObservationComponentSamples

from coordination.common.functions import sigmoid


class BrainBodySamples:

    def __init__(self, coordination: SigmoidGaussianCoordinationComponentSamples, latent_brain: MixtureComponentSamples,
                 latent_body: MixtureComponentSamples, obs_brain: ObservationComponentSamples,
                 obs_body: ObservationComponentSamples):
        self.coordination = coordination
        self.latent_brain = latent_brain
        self.latent_body = latent_body
        self.obs_brain = obs_brain
        self.obs_body = obs_body


class BrainBodySeries:

    def __init__(self, num_time_steps_in_coordination_scale: int, obs_brain: np.ndarray,
                 brain_time_steps_in_coordination_scale: np.ndarray, obs_body: np.ndarray,
                 body_time_steps_in_coordination_scale: np.ndarray):
        self.num_time_steps_in_coordination_scale = num_time_steps_in_coordination_scale
        self.obs_brain = obs_brain
        self.brain_time_steps_in_coordination_scale = brain_time_steps_in_coordination_scale
        self.obs_body = obs_body
        self.body_time_steps_in_coordination_scale = body_time_steps_in_coordination_scale

    @property
    def num_time_steps_in_brain_scale(self) -> int:
        return self.obs_brain.shape[-1]

    @property
    def num_time_steps_in_body_scale(self) -> int:
        return self.obs_body.shape[-1]

    @property
    def num_brain_channels(self) -> int:
        return self.obs_brain.shape[-2]

    @property
    def num_subjects(self) -> int:
        return self.obs_brain.shape[-3]


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

            summary.coordination_means = sigmoid(idata.posterior["unbounded_coordination"][::retain_every]).mean(
                dim=["chain", "draw"]).to_numpy()
            summary.coordination_sds = sigmoid(idata.posterior["unbounded_coordination"][::retain_every]).std(
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
                 self_dependent: bool, sd_uc: float, sd_mean_a0_brain: np.ndarray, sd_sd_aa_brain: np.ndarray,
                 sd_sd_o_brain: np.ndarray, sd_mean_a0_body: np.ndarray, sd_sd_aa_body: np.ndarray,
                 sd_sd_o_body: np.ndarray, a_mixture_weights: np.ndarray):
        self.num_subjects = num_subjects
        self.num_brain_channels = num_brain_channels

        self.coordination_cpn = SigmoidGaussianCoordinationComponent(initial_coordination, sd_uc=sd_uc)
        self.latent_brain_cpn = MixtureComponent("latent_brain", num_subjects, num_brain_channels, self_dependent,
                                                 sd_mean_a0=sd_mean_a0_brain, sd_sd_aa=sd_sd_aa_brain,
                                                 a_mixture_weights=a_mixture_weights)
        self.latent_body_cpn = MixtureComponent("latent_body", num_subjects, 1, self_dependent,
                                                sd_mean_a0=sd_mean_a0_body, sd_sd_aa=sd_sd_aa_body,
                                                a_mixture_weights=a_mixture_weights)
        self.obs_brain_cpn = ObservationComponent("obs_brain", num_subjects, num_brain_channels, sd_sd_o=sd_sd_o_brain)
        self.obs_body_cpn = ObservationComponent("obs_body", num_subjects, 1, sd_sd_o=sd_sd_o_body)

    def draw_samples(self, num_series: int, num_time_steps: int,
                     seed: Optional[int], brain_relative_frequency: float,
                     body_relative_frequency: float) -> BrainBodySamples:
        coordination_samples = self.coordination_cpn.draw_samples(num_series, num_time_steps, seed)
        latent_brain_samples = self.latent_brain_cpn.draw_samples(num_series,
                                                                  relative_frequency=brain_relative_frequency,
                                                                  coordination=coordination_samples.coordination)
        latent_body_samples = self.latent_body_cpn.draw_samples(num_series,
                                                                relative_frequency=body_relative_frequency,
                                                                coordination=coordination_samples.coordination)
        obs_brain_samples = self.obs_brain_cpn.draw_samples(latent_component=latent_brain_samples.values)
        obs_body_samples = self.obs_body_cpn.draw_samples(latent_component=latent_body_samples.values)

        samples = BrainBodySamples(coordination_samples, latent_brain_samples, latent_body_samples, obs_brain_samples,
                                   obs_body_samples)

        return samples

    def fit(self, evidence: BrainBodySeries, burn_in: int, num_samples: int, num_chains: int,
            seed: Optional[int] = None, num_jobs: int = 1) -> Tuple[pm.Model, az.InferenceData]:
        assert evidence.num_subjects == self.num_subjects
        assert evidence.num_brain_channels == self.num_brain_channels

        pymc_model = self._define_pymc_model(evidence)
        with pymc_model:
            idata = pm.sample(num_samples, init="jitter+adapt_diag", tune=burn_in, chains=num_chains, random_seed=seed,
                              cores=num_jobs)

        return pymc_model, idata

    def _define_pymc_model(self, evidence: BrainBodySeries):
        coords = {"subject": np.arange(self.num_subjects),
                  "brain_channel": np.arange(self.num_brain_channels),
                  "body_feature": np.arange(1),
                  "coordination_time": np.arange(evidence.num_time_steps_in_coordination_scale),
                  "brain_time": np.arange(evidence.num_time_steps_in_brain_scale),
                  "body_time": np.arange(evidence.num_time_steps_in_body_scale)}

        pymc_model = pm.Model(coords=coords)
        with pymc_model:
            _, coordination, _ = self.coordination_cpn.update_pymc_model(time_dimension="coordination_time")
            latent_brain, _, _, mixture_weights = self.latent_brain_cpn.update_pymc_model(
                coordination=coordination[evidence.brain_time_steps_in_coordination_scale],
                subject_dimension="subject",
                time_dimension="brain_time",
                feature_dimension="brain_channel")
            # We share the mixture weights between the brain and body components as we assume they should reflect
            # degrees of influences across components.
            latent_body, _, _, _ = self.latent_body_cpn.update_pymc_model(
                coordination=coordination[evidence.body_time_steps_in_coordination_scale],
                subject_dimension="subject",
                time_dimension="body_time",
                feature_dimension="body_feature",
                mixture_weights=mixture_weights)
            self.obs_brain_cpn.update_pymc_model(latent_component=latent_brain, observed_values=evidence.obs_brain)
            self.obs_body_cpn.update_pymc_model(latent_component=latent_body, observed_values=evidence.obs_body)

        return pymc_model

    def prior_predictive(self, evidence: BrainBodySeries, seed: Optional[int] = None):
        pymc_model = self._define_pymc_model(evidence)
        with pymc_model:
            idata = pm.sample_prior_predictive(random_seed=seed)

        return pymc_model, idata

    def clear_parameter_values(self):
        self.coordination_cpn.parameters.clear_values()
        self.latent_brain_cpn.parameters.clear_values()
        self.latent_body_cpn.parameters.clear_values()
        self.obs_brain_cpn.parameters.clear_values()
        self.obs_body_cpn.parameters.clear_values()
