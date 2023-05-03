from __future__ import annotations
from typing import Any, List, Optional, Tuple

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import xarray

from coordination.common.functions import logit
from coordination.component.coordination_component import SigmoidGaussianCoordinationComponent, \
    CoordinationComponentSamples
from coordination.component.mixture_component import MixtureComponent, MixtureComponentSamples
from coordination.component.observation_component import ObservationComponent, ObservationComponentSamples
from coordination.model.body_model import BodyPosteriorSamples, BodySeries
from coordination.model.brain_model import BrainPosteriorSamples, BrainSeries


class BrainBodySamples:

    def __init__(self, coordination: CoordinationComponentSamples, latent_brain: MixtureComponentSamples,
                 latent_body: MixtureComponentSamples, obs_brain: ObservationComponentSamples,
                 obs_body: ObservationComponentSamples):
        self.coordination = coordination
        self.latent_brain = latent_brain
        self.latent_body = latent_body
        self.obs_brain = obs_brain
        self.obs_body = obs_body


class BrainBodySeries:

    def __init__(self, uuid: str, brain_series: BrainSeries, body_series: BodySeries):
        self.uuid = uuid
        self.brain = brain_series
        self.body = body_series

    @classmethod
    def from_data_frame(cls, evidence_df: pd.DataFrame, brain_channels: List[str], ignore_bad_brain_channels: bool):
        brain_series = BrainSeries.from_data_frame(evidence_df=evidence_df, brain_channels=brain_channels,
                                                   ignore_bad_channels=ignore_bad_brain_channels)
        body_series = BodySeries.from_data_frame(evidence_df=evidence_df)

        return cls(
            uuid=brain_series.uuid,
            brain_series=brain_series,
            body_series=body_series
        )

    def chop(self, min_time_step: int, max_time_step: int):
        self.body.chop(min_time_step, max_time_step)
        self.brain.chop(min_time_step, max_time_step)

    def standardize(self):
        """
        Make sure measurements are between 0 and 1 and per feature. Don't normalize per subject otherwise we lose
        proximity relativity (how close measurements from different subjects are) which is important for the
        coordination model.
        """
        self.brain.standardize()
        self.body.standardize()

    def normalize_per_subject(self):
        self.brain.normalize_per_subject()
        self.body.normalize_per_subject()

    def normalize_across_subject(self):
        self.brain.normalize_across_subject()
        self.body.normalize_across_subject()


class BrainBodyPosteriorSamples:

    def __init__(self, unbounded_coordination: xarray.Dataset, coordination: xarray.Dataset,
                 latent_brain: xarray.Dataset, latent_body: xarray.Dataset):
        self.unbounded_coordination = unbounded_coordination
        self.coordination = coordination
        self.latent_brain = latent_brain
        self.latent_body = latent_body

    @classmethod
    def from_inference_data(cls, idata: Any) -> BrainBodyPosteriorSamples:
        brain_posterior_samples = BrainPosteriorSamples.from_inference_data(idata)
        body_posterior_samples = BodyPosteriorSamples.from_inference_data(idata)

        unbounded_coordination = brain_posterior_samples.unbounded_coordination
        coordination = brain_posterior_samples.coordination
        latent_brain = brain_posterior_samples.latent_brain
        latent_body = body_posterior_samples.latent_body

        return cls(unbounded_coordination, coordination, latent_brain, latent_body)


class BrainBodyModel:

    def __init__(self, subjects: List[str], brain_channels: List[str], self_dependent: bool, sd_mean_uc0: float,
                 sd_sd_uc: float, mean_mean_a0_brain: np.ndarray, sd_mean_a0_brain: np.ndarray,
                 sd_sd_aa_brain: np.ndarray, sd_sd_o_brain: np.ndarray, mean_mean_a0_body: np.ndarray,
                 sd_mean_a0_body: np.ndarray, sd_sd_aa_body: np.ndarray, sd_sd_o_body: np.ndarray,
                 a_mixture_weights: np.ndarray, share_mean_a0_brain_across_subjects: bool,
                 share_mean_a0_brain_across_features: bool, share_sd_aa_brain_across_subjects: bool,
                 share_sd_aa_brain_across_features: bool, share_sd_o_brain_across_subjects: bool,
                 share_sd_o_brain_across_features: bool, share_mean_a0_body_across_subjects: bool,
                 share_mean_a0_body_across_features: bool, share_sd_aa_body_across_subjects: bool,
                 share_sd_aa_body_across_features: bool, share_sd_o_body_across_subjects: bool,
                 share_sd_o_body_across_features: bool, initial_coordination: Optional[float] = None):
        self.subjects = subjects
        self.brain_channels = brain_channels
        self.num_body_features = 1

        self.coordination_cpn = SigmoidGaussianCoordinationComponent(sd_mean_uc0=sd_mean_uc0,
                                                                     sd_sd_uc=sd_sd_uc)
        if initial_coordination is not None:
            self.coordination_cpn.parameters.mean_uc0.value = np.array([logit(initial_coordination)])

        self.latent_brain_cpn = MixtureComponent(uuid="latent_brain",
                                                 num_subjects=len(subjects),
                                                 dim_value=len(brain_channels),
                                                 self_dependent=self_dependent,
                                                 mean_mean_a0=mean_mean_a0_brain,
                                                 sd_mean_a0=sd_mean_a0_brain,
                                                 sd_sd_aa=sd_sd_aa_brain,
                                                 a_mixture_weights=a_mixture_weights,
                                                 share_mean_a0_across_subjects=share_mean_a0_brain_across_subjects,
                                                 share_mean_a0_across_features=share_mean_a0_brain_across_features,
                                                 share_sd_aa_across_subjects=share_sd_aa_brain_across_subjects,
                                                 share_sd_aa_across_features=share_sd_aa_brain_across_features)
        self.latent_body_cpn = MixtureComponent(uuid="latent_body",
                                                num_subjects=len(subjects),
                                                dim_value=self.num_body_features,
                                                self_dependent=self_dependent,
                                                mean_mean_a0=mean_mean_a0_body,
                                                sd_mean_a0=sd_mean_a0_body,
                                                sd_sd_aa=sd_sd_aa_body,
                                                a_mixture_weights=a_mixture_weights,
                                                share_mean_a0_across_subjects=share_mean_a0_body_across_subjects,
                                                share_mean_a0_across_features=share_mean_a0_body_across_features,
                                                share_sd_aa_across_subjects=share_sd_aa_body_across_subjects,
                                                share_sd_aa_across_features=share_sd_aa_body_across_features)
        self.obs_brain_cpn = ObservationComponent(uuid="obs_brain",
                                                  num_subjects=len(subjects),
                                                  dim_value=len(brain_channels),
                                                  sd_sd_o=sd_sd_o_brain,
                                                  share_sd_o_across_subjects=share_sd_o_brain_across_subjects,
                                                  share_sd_o_across_features=share_sd_o_brain_across_features)
        self.obs_body_cpn = ObservationComponent(uuid="obs_body",
                                                 num_subjects=len(subjects),
                                                 dim_value=self.num_body_features,
                                                 sd_sd_o=sd_sd_o_body,
                                                 share_sd_o_across_subjects=share_sd_o_body_across_subjects,
                                                 share_sd_o_across_features=share_sd_o_body_across_features)

    @property
    def parameter_names(self) -> List[str]:
        names = self.coordination_cpn.parameter_names
        names.extend(self.latent_brain_cpn.parameter_names)
        names.extend(self.latent_body_cpn.parameter_names)
        names.extend(self.obs_brain_cpn.parameter_names)
        names.extend(self.obs_body_cpn.parameter_names)

        # The mixture weight component is shared between brain and body data.
        names.remove(self.latent_body_cpn.mixture_weights_name)

        return names

    @property
    def obs_brain_variable_name(self) -> str:
        return self.obs_brain_cpn.uuid

    @property
    def obs_body_variable_name(self) -> str:
        return self.obs_body_cpn.uuid

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
        assert evidence.brain.num_subjects == len(self.subjects)
        assert evidence.brain.num_channels == len(self.brain_channels)

        pymc_model = self._define_pymc_model(evidence)
        with pymc_model:
            idata = pm.sample(num_samples, init="jitter+adapt_diag", tune=burn_in, chains=num_chains, random_seed=seed,
                              cores=num_jobs)

        return pymc_model, idata

    def _define_pymc_model(self, evidence: BrainBodySeries):
        coords = {"subject": self.subjects,
                  "brain_channel": self.brain_channels,
                  "body_feature": ["total_energy"],
                  "coordination_time": np.arange(evidence.brain.num_time_steps_in_coordination_scale),
                  "brain_time": np.arange(evidence.brain.num_time_steps_in_brain_scale),
                  "body_time": np.arange(evidence.body.num_time_steps_in_body_scale)}

        pymc_model = pm.Model(coords=coords)
        with pymc_model:
            _, coordination, _ = self.coordination_cpn.update_pymc_model(time_dimension="coordination_time")
            latent_brain, _, _, mixture_weights = self.latent_brain_cpn.update_pymc_model(
                coordination=coordination[evidence.brain.time_steps_in_coordination_scale],
                subject_dimension="subject",
                time_dimension="brain_time",
                feature_dimension="brain_channel",
                num_time_steps=evidence.brain.num_time_steps_in_brain_scale)
            # We share the mixture weights between the brain and body components as we assume they should reflect
            # degrees of influences across components.
            latent_body, _, _, _ = self.latent_body_cpn.update_pymc_model(
                coordination=coordination[evidence.body.time_steps_in_coordination_scale],
                subject_dimension="subject",
                time_dimension="body_time",
                feature_dimension="body_feature",
                mixture_weights=mixture_weights,
                num_time_steps=evidence.body.num_time_steps_in_body_scale)
            self.obs_brain_cpn.update_pymc_model(latent_component=latent_brain,
                                                 subject_dimension="subject",
                                                 feature_dimension="brain_channel",
                                                 time_dimension="brain_time",
                                                 observed_values=evidence.brain.observation)
            self.obs_body_cpn.update_pymc_model(latent_component=latent_body,
                                                subject_dimension="subject",
                                                feature_dimension="body_feature",
                                                time_dimension="body_time",
                                                observed_values=evidence.body.observation)

        return pymc_model

    def prior_predictive(self, evidence: BrainBodySeries, num_samples: int, seed: Optional[int] = None):
        pymc_model = self._define_pymc_model(evidence)
        with pymc_model:
            idata = pm.sample_prior_predictive(samples=num_samples, random_seed=seed)

        return pymc_model, idata

    def posterior_predictive(self, evidence: BrainBodySeries, trace: az.InferenceData, seed: Optional[int] = None):
        pymc_model = self._define_pymc_model(evidence)
        with pymc_model:
            idata = pm.sample_posterior_predictive(trace=trace, random_seed=seed)

        return pymc_model, idata

    def clear_parameter_values(self):
        self.coordination_cpn.parameters.clear_values()
        self.latent_brain_cpn.parameters.clear_values()
        self.latent_body_cpn.parameters.clear_values()
        self.obs_brain_cpn.parameters.clear_values()
        self.obs_body_cpn.parameters.clear_values()

    @staticmethod
    def inference_data_to_posterior_samples(idata: az.InferenceData) -> BrainBodyPosteriorSamples:
        return BrainBodyPosteriorSamples.from_inference_data(idata)
