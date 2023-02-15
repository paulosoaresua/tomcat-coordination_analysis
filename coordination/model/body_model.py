from __future__ import annotations
from typing import Any, Optional, Tuple

import arviz as az
import numpy as np
import pymc as pm
import xarray

from coordination.model.components.coordination_component import SigmoidGaussianCoordinationComponent, \
    SigmoidGaussianCoordinationComponentSamples
from coordination.model.components.mixture_component import MixtureComponent, MixtureComponentSamples
from coordination.model.components.observation_component import ObservationComponent, ObservationComponentSamples

from coordination.common.functions import sigmoid


class BodySamples:

    def __init__(self, coordination: SigmoidGaussianCoordinationComponentSamples, latent_body: MixtureComponentSamples,
                 obs_body: ObservationComponentSamples):
        self.coordination = coordination
        self.latent_body = latent_body
        self.obs_body = obs_body


class BodySeries:

    def __init__(self, num_time_steps_in_coordination_scale: int, obs_body: np.ndarray,
                 body_time_steps_in_coordination_scale: np.ndarray):
        self.num_time_steps_in_coordination_scale = num_time_steps_in_coordination_scale
        self.obs_body = obs_body
        self.body_time_steps_in_coordination_scale = body_time_steps_in_coordination_scale

    @property
    def num_time_steps_in_body_scale(self) -> int:
        return self.obs_body.shape[-1]

    @property
    def num_body_features(self) -> int:
        return self.obs_body.shape[-2]

    @property
    def num_subjects(self) -> int:
        return self.obs_body.shape[-3]


class BodyPosteriorSamples:

    def __init__(self, unbounded_coordination: xarray.Dataset, coordination: xarray.Dataset,
                 latent_body: xarray.Dataset):
        self.unbounded_coordination = unbounded_coordination
        self.coordination = coordination
        self.latent_body = latent_body

    @classmethod
    def from_inference_data(cls, idata: Any) -> BodyPosteriorSamples:
        unbounded_coordination = idata.posterior["unbounded_coordination"]
        coordination = sigmoid(unbounded_coordination)
        latent_body = idata.posterior["latent_body"]

        return cls(unbounded_coordination, coordination, latent_body)


class BodyModel:

    def __init__(self, initial_coordination: float, num_subjects: int, self_dependent: bool, sd_uc: float,
                 sd_mean_a0: np.ndarray, sd_sd_aa: np.ndarray, sd_sd_o: np.ndarray, a_mixture_weights: np.ndarray):
        self.num_subjects = num_subjects

        # Single number representing quantity of movement per time step.
        self.num_body_features = 1

        self.hyper_parameters = {
            "num_subjects": num_subjects,
            "self_dependent": self_dependent,
            "sd_uc": sd_uc,
            "sd_mean_a0": sd_mean_a0.tolist(),
            "sd_sd_aa": sd_sd_aa.tolist(),
            "sd_sd_o": sd_sd_o.tolist(),
            "a_mixture_weights": a_mixture_weights.tolist()
        }

        self.coordination_cpn = SigmoidGaussianCoordinationComponent(initial_coordination, sd_uc=sd_uc)
        self.latent_body_cpn = MixtureComponent("latent_body", num_subjects, self.num_body_features, self_dependent,
                                                sd_mean_a0=sd_mean_a0, sd_sd_aa=sd_sd_aa,
                                                a_mixture_weights=a_mixture_weights)
        self.obs_body_cpn = ObservationComponent("obs_body", num_subjects, self.num_body_features, sd_sd_o=sd_sd_o)

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
        assert evidence.num_subjects == self.num_subjects
        assert evidence.num_body_features == self.num_body_features

        pymc_model = self._define_pymc_model(evidence)
        with pymc_model:
            idata = pm.sample(num_samples, init="jitter+adapt_diag", tune=burn_in, chains=num_chains, random_seed=seed,
                              cores=num_jobs)

        return pymc_model, idata

    def _define_pymc_model(self, evidence: BodySeries):
        coords = {"subject": np.arange(self.num_subjects),
                  "body_channel": np.arange(self.num_body_features),
                  "coordination_time": np.arange(evidence.num_time_steps_in_coordination_scale),
                  "body_time": np.arange(evidence.num_time_steps_in_body_scale)}

        pymc_model = pm.Model(coords=coords)
        with pymc_model:
            _, coordination, _ = self.coordination_cpn.update_pymc_model(time_dimension="coordination_time")
            latent_body, _, _, _ = self.latent_body_cpn.update_pymc_model(
                coordination=coordination[evidence.body_time_steps_in_coordination_scale],
                subject_dimension="subject",
                time_dimension="body_time",
                feature_dimension="body_channel")
            self.obs_body_cpn.update_pymc_model(latent_component=latent_body, observed_values=evidence.obs_body)

        return pymc_model

    def prior_predictive(self, evidence: BodySeries, seed: Optional[int] = None):
        pymc_model = self._define_pymc_model(evidence)
        with pymc_model:
            idata = pm.sample_prior_predictive(random_seed=seed)

        return pymc_model, idata

    def clear_parameter_values(self):
        self.coordination_cpn.parameters.clear_values()
        self.latent_body_cpn.parameters.clear_values()
        self.obs_body_cpn.parameters.clear_values()
