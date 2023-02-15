from __future__ import annotations
from typing import Any, List, Optional, Tuple

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import xarray

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

    @classmethod
    def from_data_frame(cls, experiment_id: str, evidence_df: pd.DataFrame, brain_channels: List[str]):
        #TODO - Fix this
        row_df = evidence_df[evidence_df["experiment_id"] == experiment_id]

        return cls(
            num_time_steps_in_coordination_scale=row_df["num_total_time_steps"],
            obs_brain=row_df["num_total_time_steps"],
            brain_time_steps_in_coordination_scale=row_df["num_total_time_steps"],
            obs_body=row_df["num_total_time_steps"],
            body_time_steps_in_coordination_scale=row_df["num_total_time_steps"],
        )



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


class BrainBodyPosteriorSamples:

    def __init__(self, unbounded_coordination: xarray.Dataset, coordination: xarray.Dataset,
                 latent_brain: xarray.Dataset, latent_body: xarray.Dataset):
        self.unbounded_coordination = unbounded_coordination
        self.coordination = coordination
        self.latent_brain = latent_brain
        self.latent_body = latent_body

    @classmethod
    def from_inference_data(cls, idata: Any) -> BrainBodyPosteriorSamples:
        unbounded_coordination = idata.posterior["unbounded_coordination"]
        coordination = sigmoid(unbounded_coordination)
        latent_brain = idata.posterior["latent_brain"]
        latent_body = idata.posterior["latent_body"]

        return cls(unbounded_coordination, coordination, latent_brain, latent_body)


class BrainBodyModel:

    def __init__(self, initial_coordination: float, num_subjects: int, brain_channels: List[str],
                 self_dependent: bool, sd_uc: float, sd_mean_a0_brain: np.ndarray, sd_sd_aa_brain: np.ndarray,
                 sd_sd_o_brain: np.ndarray, sd_mean_a0_body: np.ndarray, sd_sd_aa_body: np.ndarray,
                 sd_sd_o_body: np.ndarray, a_mixture_weights: np.ndarray):
        self.num_subjects = num_subjects
        self.brain_channels = brain_channels

        self.coordination_cpn = SigmoidGaussianCoordinationComponent(initial_coordination, sd_uc=sd_uc)
        self.latent_brain_cpn = MixtureComponent("latent_brain", num_subjects, len(brain_channels), self_dependent,
                                                 sd_mean_a0=sd_mean_a0_brain, sd_sd_aa=sd_sd_aa_brain,
                                                 a_mixture_weights=a_mixture_weights)
        self.latent_body_cpn = MixtureComponent("latent_body", num_subjects, 1, self_dependent,
                                                sd_mean_a0=sd_mean_a0_body, sd_sd_aa=sd_sd_aa_body,
                                                a_mixture_weights=a_mixture_weights)
        self.obs_brain_cpn = ObservationComponent("obs_brain", num_subjects, len(brain_channels), sd_sd_o=sd_sd_o_brain)
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
        assert evidence.num_brain_channels == len(self.brain_channels)

        pymc_model = self._define_pymc_model(evidence)
        with pymc_model:
            idata = pm.sample(num_samples, init="jitter+adapt_diag", tune=burn_in, chains=num_chains, random_seed=seed,
                              cores=num_jobs)

        return pymc_model, idata

    def _define_pymc_model(self, evidence: BrainBodySeries):
        coords = {"subject": np.arange(self.num_subjects),
                  "brain_channel": self.brain_channels,
                  "body_feature": ["total_energy"],
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

    def prior_predictive(self, evidence: BrainBodySeries, num_samples: int, seed: Optional[int] = None):
        pymc_model = self._define_pymc_model(evidence)
        with pymc_model:
            idata = pm.sample_prior_predictive(samples=num_samples, random_seed=seed)

        return pymc_model, idata

    def clear_parameter_values(self):
        self.coordination_cpn.parameters.clear_values()
        self.latent_brain_cpn.parameters.clear_values()
        self.latent_body_cpn.parameters.clear_values()
        self.obs_brain_cpn.parameters.clear_values()
        self.obs_body_cpn.parameters.clear_values()
