from __future__ import annotations

from typing import Any, List, Optional, Tuple

import arviz as az
from ast import literal_eval
import numpy as np
import pandas as pd
import pymc as pm
import xarray

from coordination.common.functions import logit
from coordination.component.coordination_component import SigmoidGaussianCoordinationComponent, \
    SigmoidGaussianCoordinationComponentSamples
from coordination.component.mixture_component import MixtureComponent, MixtureComponentSamples
from coordination.component.observation_component import ObservationComponent, ObservationComponentSamples
from coordination.model.coordination_model import CoordinationPosteriorSamples


class BodySamples:

    def __init__(self, coordination: SigmoidGaussianCoordinationComponentSamples, latent_body: MixtureComponentSamples,
                 obs_body: ObservationComponentSamples):
        self.coordination = coordination
        self.latent_body = latent_body
        self.obs_body = obs_body


class BodySeries:

    def __init__(self, uuid: str, subjects: List[str], num_time_steps_in_coordination_scale: int,
                 observation: np.ndarray,
                 time_steps_in_coordination_scale: np.ndarray):
        self.uuid = uuid
        self.subjects = subjects
        self.num_time_steps_in_coordination_scale = num_time_steps_in_coordination_scale
        self.observation = observation
        self.time_steps_in_coordination_scale = time_steps_in_coordination_scale

    def chop(self, min_time_step: int, max_time_step: int):
        """
        Chops the series into a pre-defined range.
        """
        self.num_time_steps_in_coordination_scale = max_time_step - min_time_step
        t_min_component = 0
        t_max_component = 0
        for t in range(self.num_time_steps_in_coordination_scale):
            if self.time_steps_in_coordination_scale[t] < min_time_step:
                t_min_component = t + 1

            if self.time_steps_in_coordination_scale[t] < max_time_step:
                t_max_component = t + 1

        self.observation = self.observation[..., t_min_component:t_max_component]
        self.time_steps_in_coordination_scale = self.time_steps_in_coordination_scale[
                                                t_min_component:t_max_component] - min_time_step

    def plot_observations(self, ax: Any):
        # One plot per channel
        xs = self.time_steps_in_coordination_scale[:, None].repeat(self.num_subjects, axis=1)
        ys = self.observation[:, 0, :].T
        ax.plot(xs, ys, label=self.subjects)
        ax.set_title("Body Motion Energy")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Observed Value")
        ax.set_xlim([-0.5, self.num_time_steps_in_coordination_scale + 0.5])
        ax.legend()

    def plot_observation_differences(self, ax: Any, self_dependent: bool):
        # Plot the difference between the current subject's energy and their previous signal and a different
        # subject's previous energy

        labels = []
        for subject_target in self.subjects:
            for subject_source in self.subjects:
                labels.append(f"{subject_target} - {subject_source}")

        xs = self.time_steps_in_coordination_scale[1:, None].repeat(self.num_subjects ** 2, axis=1)
        Obs = self.observation[:, 0, :]
        A = np.concatenate([Obs] * self.num_subjects, axis=0).T

        if not self_dependent:
            # We approximate the fixed mean as the first observation
            for i in range(self.num_subjects):
                A[:, i * self.num_subjects] = Obs[i, 0]

        B = np.repeat(Obs, self.num_subjects, axis=0).T
        ys = np.abs(B[1:] - A[:-1])
        ax.plot(xs, ys, label=labels)
        ax.set_title("Body Motion Energy")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Observed Value")
        ax.set_xlim([-0.5, self.num_time_steps_in_coordination_scale + 0.5])
        ax.legend()

    @classmethod
    def from_data_frame(cls, evidence_df: pd.DataFrame):
        obs_body = np.array(literal_eval(evidence_df["body_motion_energy"].values[0]))

        return cls(
            uuid=evidence_df["experiment_id"].values[0],
            subjects=literal_eval(evidence_df["subjects"].values[0]),
            num_time_steps_in_coordination_scale=evidence_df["num_time_steps_in_coordination_scale"].values[0],
            observation=obs_body,
            time_steps_in_coordination_scale=np.array(
                literal_eval(evidence_df["body_motion_energy_time_steps_in_coordination_scale"].values[0]), dtype=int))

    def standardize(self):
        """
        Make sure measurements are between 0 and 1 and per feature. Don't normalize per subject otherwise we lose
        proximity relativity (how close measurements from different subjects are) which is important for the
        coordination model.
        """
        max_value = self.observation.max(axis=(0, 2), initial=0)[None, :, None]
        min_value = self.observation.min(axis=(0, 2), initial=0)[None, :, None]
        self.observation = (self.observation - min_value) / (max_value - min_value)

    def normalize_per_subject(self):
        """
        Make sure measurements have mean 0 and standard deviation 1 per subject and feature.
        """
        mean = self.observation.mean(axis=-1)[..., None]
        std = self.observation.std(axis=-1)[..., None]
        self.observation = (self.observation - mean) / std

    def normalize_across_subject(self):
        """
        Make sure measurements have mean 0 and standard deviation 1 per feature.
        """

        mean = self.observation.mean(axis=(0, 2))[None, :, None]
        std = self.observation.std(axis=(0, 2))[None, :, None]
        self.observation = (self.observation - mean) / std

    @property
    def num_time_steps_in_body_scale(self) -> int:
        return self.observation.shape[-1]

    @property
    def num_body_features(self) -> int:
        return self.observation.shape[-2]

    @property
    def num_subjects(self) -> int:
        return self.observation.shape[-3]


class BodyPosteriorSamples:

    def __init__(self, unbounded_coordination: xarray.Dataset, coordination: xarray.Dataset,
                 latent_body: xarray.Dataset):
        self.unbounded_coordination = unbounded_coordination
        self.coordination = coordination
        self.latent_body = latent_body

    @classmethod
    def from_inference_data(cls, idata: Any) -> BodyPosteriorSamples:
        coordination_posterior_samples = CoordinationPosteriorSamples.from_inference_data(idata)
        unbounded_coordination = coordination_posterior_samples.unbounded_coordination
        coordination = coordination_posterior_samples.coordination
        latent_body = idata.posterior["latent_body"]

        return cls(unbounded_coordination, coordination, latent_body)


class BodyModel:

    def __init__(self, subjects: List[str], self_dependent: bool, sd_mean_uc0: float,
                 sd_sd_uc: float, sd_mean_a0: np.ndarray, sd_sd_aa: np.ndarray, sd_sd_o: np.ndarray,
                 a_mixture_weights: np.ndarray, share_params_across_subjects: bool,
                 initial_coordination: Optional[float] = None):
        self.subjects = subjects
        self.share_params_across_subjects = share_params_across_subjects

        # Single number representing quantity of movement per time step.
        self.num_body_features = 1

        self.coordination_cpn = SigmoidGaussianCoordinationComponent(sd_mean_uc0=sd_mean_uc0,
                                                                     sd_sd_uc=sd_sd_uc)
        if initial_coordination is not None:
            self.coordination_cpn.parameters.mean_uc0.value = np.array([logit(initial_coordination)])

        self.latent_body_cpn = MixtureComponent(uuid="latent_body",
                                                num_subjects=len(subjects),
                                                dim_value=self.num_body_features,
                                                self_dependent=self_dependent,
                                                sd_mean_a0=sd_mean_a0,
                                                sd_sd_aa=sd_sd_aa,
                                                a_mixture_weights=a_mixture_weights,
                                                share_params_across_subjects=share_params_across_subjects)
        self.obs_body_cpn = ObservationComponent(uuid="obs_body",
                                                 num_subjects=len(subjects),
                                                 dim_value=self.num_body_features,
                                                 sd_sd_o=sd_sd_o,
                                                 share_params_across_subjects=share_params_across_subjects)

    @property
    def parameter_names(self) -> List[str]:
        names = self.coordination_cpn.parameter_names
        names.extend(self.latent_body_cpn.parameter_names)
        names.extend(self.obs_body_cpn.parameter_names)

        return names

    @property
    def obs_body_variable_name(self) -> str:
        return self.obs_body_cpn.uuid

    def draw_samples(self, num_series: int, num_time_steps: int, seed: Optional[int],
                     body_relative_frequency: float) -> BodySamples:
        coordination_samples = self.coordination_cpn.draw_samples(num_series, num_time_steps, seed)
        latent_body_samples = self.latent_body_cpn.draw_samples(num_series,
                                                                relative_frequency=body_relative_frequency,
                                                                coordination=coordination_samples.coordination)
        obs_body_samples = self.obs_body_cpn.draw_samples(latent_component=latent_body_samples.values)

        samples = BodySamples(coordination_samples, latent_body_samples, obs_body_samples)

        return samples

    def fit(self, evidence: BodySeries, burn_in: int, num_samples: int, num_chains: int,
            seed: Optional[int] = None, num_jobs: int = 1) -> Tuple[pm.Model, az.InferenceData]:
        assert evidence.num_subjects == len(self.subjects)
        assert evidence.num_body_features == self.num_body_features

        pymc_model = self._define_pymc_model(evidence)
        with pymc_model:
            idata = pm.sample(num_samples, init="jitter+adapt_diag", tune=burn_in, chains=num_chains, random_seed=seed,
                              cores=num_jobs)

        return pymc_model, idata

    def _define_pymc_model(self, evidence: BodySeries):
        coords = {"subject": self.subjects,
                  "body_feature": ["total_energy"],
                  "coordination_time": np.arange(evidence.num_time_steps_in_coordination_scale),
                  "body_time": np.arange(evidence.num_time_steps_in_body_scale)}

        pymc_model = pm.Model(coords=coords)
        with pymc_model:
            _, coordination, _ = self.coordination_cpn.update_pymc_model(time_dimension="coordination_time")
            latent_body, _, _, _ = self.latent_body_cpn.update_pymc_model(
                coordination=coordination[evidence.time_steps_in_coordination_scale],
                subject_dimension="subject",
                time_dimension="body_time",
                feature_dimension="body_feature")
            self.obs_body_cpn.update_pymc_model(latent_component=latent_body,
                                                subject_dimension="subject",
                                                feature_dimension="body_feature",
                                                time_dimension="body_time",
                                                observed_values=evidence.observation)

        return pymc_model

    def prior_predictive(self, evidence: BodySeries, num_samples: int, seed: Optional[int] = None):
        pymc_model = self._define_pymc_model(evidence)
        with pymc_model:
            idata = pm.sample_prior_predictive(samples=num_samples, random_seed=seed)

        return pymc_model, idata

    def posterior_predictive(self, evidence: BodySeries, trace: az.InferenceData, seed: Optional[int] = None):
        pymc_model = self._define_pymc_model(evidence)
        with pymc_model:
            idata = pm.sample_posterior_predictive(trace=trace, random_seed=seed)

        return pymc_model, idata

    def clear_parameter_values(self):
        self.coordination_cpn.parameters.clear_values()
        self.latent_body_cpn.parameters.clear_values()
        self.obs_body_cpn.parameters.clear_values()

    @staticmethod
    def inference_data_to_posterior_samples(idata: az.InferenceData) -> BodyPosteriorSamples:
        return BodyPosteriorSamples.from_inference_data(idata)
