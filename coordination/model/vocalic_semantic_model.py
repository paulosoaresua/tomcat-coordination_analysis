from __future__ import annotations
from typing import Any, Optional, Tuple

import arviz as az
import numpy as np
import pymc as pm

from coordination.model.components.coordination_component import SigmoidGaussianCoordinationComponent, \
    SigmoidGaussianCoordinationComponentSamples
from coordination.model.components.serialized_component import SerializedComponent, SerializedComponentSamples
from coordination.model.components.link_component import LinkComponent, LinkComponentSamples
from coordination.model.components.observation_component import SerializedObservationComponent, \
    SerializedObservationComponentSamples

from coordination.common.functions import sigmoid


class VocalicSemanticSamples:

    def __init__(self, coordination: SigmoidGaussianCoordinationComponentSamples,
                 latent_vocalic: SerializedComponentSamples,
                 semantic_link: LinkComponentSamples, obs_vocalic: SerializedObservationComponentSamples):
        self.coordination = coordination
        self.latent_vocalic = latent_vocalic
        self.semantic_link = semantic_link
        self.obs_vocalic = obs_vocalic


class VocalicSemanticSeries:

    def __init__(self, num_time_steps_in_coordination_scale: int, vocalic_subjects: np.ndarray, obs_vocalic: np.ndarray,
                 vocalic_prev_time_same_subject: np.ndarray, vocalic_prev_time_diff_subject: np.ndarray,
                 vocalic_time_steps_in_coordination_scale: np.ndarray,
                 semantic_link_time_steps_in_coordination_scale: np.ndarray):
        self.num_time_steps_in_coordination_scale = num_time_steps_in_coordination_scale
        self.vocalic_subjects = vocalic_subjects
        self.obs_vocalic = obs_vocalic
        self.vocalic_prev_time_same_subject = vocalic_prev_time_same_subject
        self.vocalic_prev_time_diff_subject = vocalic_prev_time_diff_subject
        self.vocalic_time_steps_in_coordination_scale = vocalic_time_steps_in_coordination_scale
        self.semantic_link_time_steps_in_coordination_scale = semantic_link_time_steps_in_coordination_scale

    @property
    def num_time_steps_in_vocalic_scale(self) -> int:
        return self.obs_vocalic.shape[-1]

    @property
    def num_time_steps_in_semantic_link_scale(self) -> int:
        return self.semantic_link_time_steps_in_coordination_scale.shape[-1]

    @property
    def num_vocalic_features(self) -> int:
        return self.obs_vocalic.shape[-2]

    @property
    def vocalic_prev_same_subject_mask(self) -> np.ndarray:
        return np.where(self.vocalic_prev_time_same_subject >= 0, 1, 0)

    @property
    def vocalic_prev_diff_subject_mask(self) -> np.ndarray:
        return np.where(self.vocalic_prev_time_diff_subject >= 0, 1, 0)


class VocalicSemanticInferenceSummary:

    def __init__(self):
        self.unbounded_coordination_means = np.array([])
        self.coordination_means = np.array([])
        self.latent_vocalic_means = np.array([])

        self.unbounded_coordination_sds = np.array([])
        self.coordination_sds = np.array([])
        self.latent_vocalic_sds = np.array([])

    @classmethod
    def from_inference_data(cls, idata: Any, retain_every: int = 1) -> VocalicSemanticInferenceSummary:
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

        if "latent_vocalic" in idata.posterior:
            summary.latent_brain_means = idata.posterior["latent_vocalic"][::retain_every].mean(
                dim=["chain", "draw"]).to_numpy()
            summary.latent_brain_sds = idata.posterior["latent_vocalic"][::retain_every].std(
                dim=["chain", "draw"]).to_numpy()

        return summary


class VocalicSemanticModel:

    def __init__(self, initial_coordination: float, num_subjects: int, num_vocalic_features: int,
                 self_dependent: bool, sd_uc: float, sd_mean_a0_vocalic: np.ndarray, sd_sd_aa_vocalic: np.ndarray,
                 sd_sd_o_vocalic: np.ndarray, a_p_semantic_link: float, b_p_semantic_link: float):
        self.num_subjects = num_subjects
        self.num_vocalic_features = num_vocalic_features

        self.coordination_cpn = SigmoidGaussianCoordinationComponent(initial_coordination, sd_uc=sd_uc)
        self.latent_vocalic_cpn = SerializedComponent("latent_vocalic", num_subjects, num_vocalic_features,
                                                      self_dependent,
                                                      sd_mean_a0=sd_mean_a0_vocalic, sd_sd_aa=sd_sd_aa_vocalic)
        self.semantic_link_cpn = LinkComponent("semantic_link", a_p=a_p_semantic_link, b_p=b_p_semantic_link)
        self.obs_vocalic_cpn = SerializedObservationComponent("obs_vocalic", num_subjects, num_vocalic_features,
                                                              sd_sd_o=sd_sd_o_vocalic)

    def draw_samples(self, num_series: int, num_time_steps: int, vocalic_time_scale_density: float,
                     semantic_link_time_Scale_density: float, can_repeat_subject: bool,
                     seed: Optional[int] = None) -> VocalicSemanticSamples:
        coordination_samples = self.coordination_cpn.draw_samples(num_series, num_time_steps, seed)
        latent_vocalic_samples = self.latent_vocalic_cpn.draw_samples(num_series=num_series,
                                                                      time_scale_density=vocalic_time_scale_density,
                                                                      coordination=coordination_samples.coordination,
                                                                      can_repeat_subject=can_repeat_subject)
        semantic_link_samples = self.semantic_link_cpn.draw_samples(num_series=num_series,
                                                                    time_scale_density=semantic_link_time_Scale_density,
                                                                    coordination=coordination_samples.coordination)
        obs_vocalic_samples = self.obs_vocalic_cpn.draw_samples(latent_component=latent_vocalic_samples.values,
                                                                subjects=latent_vocalic_samples.subjects)

        samples = VocalicSemanticSamples(coordination=coordination_samples, latent_vocalic=latent_vocalic_samples,
                                         semantic_link=semantic_link_samples,
                                         obs_vocalic=obs_vocalic_samples)

        return samples

    def fit(self, evidence: VocalicSemanticSeries, burn_in: int, num_samples: int, num_chains: int,
            seed: Optional[int] = None, num_jobs: int = 1) -> Tuple[pm.Model, az.InferenceData]:
        assert evidence.num_vocalic_features == self.num_vocalic_features

        pymc_model = self._define_pymc_model(evidence)
        with pymc_model:
            idata = pm.sample(num_samples, init="jitter+adapt_diag", tune=burn_in, chains=num_chains, random_seed=seed,
                              cores=num_jobs)

        return pymc_model, idata

    def _define_pymc_model(self, evidence: VocalicSemanticSeries):
        coords = {"subject": np.arange(self.num_subjects),
                  "vocalic_feature": np.arange(self.num_vocalic_features),
                  "coordination_time": np.arange(evidence.num_time_steps_in_coordination_scale),
                  "vocalic_time": np.arange(evidence.num_time_steps_in_vocalic_scale),
                  "link_time": np.arange(evidence.num_time_steps_in_semantic_link_scale)}

        pymc_model = pm.Model(coords=coords)
        with pymc_model:
            _, coordination, _ = self.coordination_cpn.update_pymc_model(time_dimension="coordination_time")
            latent_vocalic, _, _ = self.latent_vocalic_cpn.update_pymc_model(
                coordination=coordination[evidence.vocalic_time_steps_in_coordination_scale],
                prev_time_same_subject=evidence.vocalic_prev_time_same_subject,
                prev_time_diff_subject=evidence.vocalic_prev_time_diff_subject,
                prev_same_subject_mask=evidence.vocalic_prev_same_subject_mask,
                prev_diff_subject_mask=evidence.vocalic_prev_diff_subject_mask,
                subjects=evidence.vocalic_subjects,
                time_dimension="vocalic_time",
                feature_dimension="vocalic_feature")

            self.semantic_link_cpn.update_pymc_model(
                coordination=coordination[evidence.num_time_steps_in_semantic_link_scale],
                observed_values=np.ones(evidence.num_time_steps_in_semantic_link_scale))

            self.obs_vocalic_cpn.update_pymc_model(latent_component=latent_vocalic,
                                                   subjects=evidence.vocalic_subjects,
                                                   observed_values=evidence.obs_vocalic)

        return pymc_model

    def prior_predictive(self, evidence: VocalicSemanticSeries, seed: Optional[int] = None):
        pymc_model = self._define_pymc_model(evidence)
        with pymc_model:
            idata = pm.sample_prior_predictive(random_seed=seed)

        return pymc_model, idata

    def clear_parameter_values(self):
        self.coordination_cpn.parameters.clear_values()
        self.latent_vocalic_cpn.parameters.clear_values()
        self.semantic_link_cpn.parameters.clear_values()
        self.obs_vocalic_cpn.parameters.clear_values()
