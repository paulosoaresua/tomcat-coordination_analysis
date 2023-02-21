from __future__ import annotations
from typing import Any, List, Optional, Tuple

import arviz as az
from ast import literal_eval
import numpy as np
import pandas as pd
import pymc as pm
import xarray

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

    def __init__(self, uuid: str, vocalic_features: List[str], num_time_steps_in_coordination_scale: int,
                 vocalic_subjects: np.ndarray, obs_vocalic: np.ndarray, vocalic_prev_time_same_subject: np.ndarray,
                 vocalic_prev_time_diff_subject: np.ndarray, vocalic_time_steps_in_coordination_scale: np.ndarray,
                 semantic_link_time_steps_in_coordination_scale: np.ndarray):
        self.uuid = uuid
        self.vocalic_features = vocalic_features
        self.num_time_steps_in_coordination_scale = num_time_steps_in_coordination_scale
        self.vocalic_subjects = vocalic_subjects
        self.obs_vocalic = obs_vocalic
        self.vocalic_prev_time_same_subject = vocalic_prev_time_same_subject
        self.vocalic_prev_time_diff_subject = vocalic_prev_time_diff_subject
        self.vocalic_time_steps_in_coordination_scale = vocalic_time_steps_in_coordination_scale
        self.semantic_link_time_steps_in_coordination_scale = semantic_link_time_steps_in_coordination_scale

    @classmethod
    def from_data_frame(cls, experiment_id: str, evidence_df: pd.DataFrame, vocalic_features: List[str]):
        row_df = evidence_df[evidence_df["experiment_id"] == experiment_id]

        obs_vocalic = []
        for vocalic_feature in vocalic_features:
            obs_vocalic.append(np.array(literal_eval(row_df[f"{vocalic_feature}"].values[0])))
        # Swap axes such that the first dimension represents the different subjects and the second the vocalic features
        obs_vocalic = np.array(obs_vocalic).swapaxes(0, 1)

        return cls(
            uuid=row_df["experiment_id"].values[0],
            vocalic_features=vocalic_features,
            num_time_steps_in_coordination_scale=row_df["num_time_steps_in_coordination_scale"],
            vocalic_subjects=np.array(literal_eval(row_df["subjects"].values[0])),
            obs_vocalic=obs_vocalic,
            vocalic_prev_time_same_subject=np.array(literal_eval(row_df["vocalic_prev_time_same_subject"].values[0])),
            vocalic_prev_time_diff_subject=np.array(literal_eval(row_df["vocalic_prev_time_diff_subject"].values[0])),
            vocalic_time_steps_in_coordination_scale=np.array(
                literal_eval(row_df["vocalic_time_steps_in_coordination_scale"].values[0])),
            semantic_link_time_steps_in_coordination_scale=np.array(
                literal_eval(row_df["semantic_link_time_steps_in_coordination_scale"].values[0]))
        )

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


class VocalicSemanticPosteriorSamples:

    def __init__(self, unbounded_coordination: xarray.Dataset, coordination: xarray.Dataset,
                 latent_vocalic: xarray.Dataset):
        self.unbounded_coordination = unbounded_coordination
        self.coordination = coordination
        self.latent_vocalic = latent_vocalic

    @classmethod
    def from_inference_data(cls, idata: Any) -> VocalicSemanticPosteriorSamples:
        unbounded_coordination = idata.posterior["unbounded_coordination"]
        coordination = sigmoid(unbounded_coordination)
        latent_vocalic = idata.posterior["latent_vocalic"]

        return cls(unbounded_coordination, coordination, latent_vocalic)


class VocalicSemanticModel:

    def __init__(self, initial_coordination: float, num_subjects: int, vocalic_features: List[str],
                 self_dependent: bool, sd_uc: float, sd_mean_a0_vocalic: np.ndarray, sd_sd_aa_vocalic: np.ndarray,
                 sd_sd_o_vocalic: np.ndarray, a_p_semantic_link: float, b_p_semantic_link: float):
        self.num_subjects = num_subjects
        self.vocalic_features = vocalic_features

        self.coordination_cpn = SigmoidGaussianCoordinationComponent(initial_coordination, sd_uc=sd_uc)
        self.latent_vocalic_cpn = SerializedComponent("latent_vocalic", num_subjects, len(vocalic_features),
                                                      self_dependent,
                                                      sd_mean_a0=sd_mean_a0_vocalic, sd_sd_aa=sd_sd_aa_vocalic)
        self.semantic_link_cpn = LinkComponent("semantic_link", a_p=a_p_semantic_link, b_p=b_p_semantic_link)
        self.obs_vocalic_cpn = SerializedObservationComponent("obs_vocalic", num_subjects, len(vocalic_features),
                                                              sd_sd_o=sd_sd_o_vocalic)

    @property
    def parameter_names(self) -> List[str]:
        names = self.coordination_cpn.parameter_names
        names.extend(self.latent_vocalic_cpn.parameter_names)
        names.extend(self.obs_vocalic_cpn.parameter_names)
        names.extend(self.semantic_link_cpn.parameter_names)

        return names

    @property
    def obs_vocalic_variable_name(self) -> str:
        return self.obs_vocalic_cpn.uuid

    def draw_samples(self, num_series: int, num_time_steps: int, vocalic_time_scale_density: float,
                     semantic_link_time_scale_density: float, can_repeat_subject: bool,
                     seed: Optional[int] = None) -> VocalicSemanticSamples:
        coordination_samples = self.coordination_cpn.draw_samples(num_series, num_time_steps, seed)
        latent_vocalic_samples = self.latent_vocalic_cpn.draw_samples(num_series=num_series,
                                                                      time_scale_density=vocalic_time_scale_density,
                                                                      coordination=coordination_samples.coordination,
                                                                      can_repeat_subject=can_repeat_subject)
        semantic_link_samples = self.semantic_link_cpn.draw_samples(num_series=num_series,
                                                                    time_scale_density=semantic_link_time_scale_density,
                                                                    coordination=coordination_samples.coordination)
        obs_vocalic_samples = self.obs_vocalic_cpn.draw_samples(latent_component=latent_vocalic_samples.values,
                                                                subjects=latent_vocalic_samples.subjects)

        samples = VocalicSemanticSamples(coordination=coordination_samples, latent_vocalic=latent_vocalic_samples,
                                         semantic_link=semantic_link_samples,
                                         obs_vocalic=obs_vocalic_samples)

        return samples

    def fit(self, evidence: VocalicSemanticSeries, burn_in: int, num_samples: int, num_chains: int,
            seed: Optional[int] = None, num_jobs: int = 1) -> Tuple[pm.Model, az.InferenceData]:
        assert evidence.num_vocalic_features == len(self.vocalic_features)

        pymc_model = self._define_pymc_model(evidence)
        with pymc_model:
            idata = pm.sample(num_samples, init="jitter+adapt_diag", tune=burn_in, chains=num_chains, random_seed=seed,
                              cores=num_jobs)

        return pymc_model, idata

    def _define_pymc_model(self, evidence: VocalicSemanticSeries):
        coords = {"vocalic_feature": self.vocalic_features,
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
                                                   feature_dimension="vocalic_feature",
                                                   time_dimension="vocalic_time",
                                                   observed_values=evidence.obs_vocalic)

        return pymc_model

    def prior_predictive(self, evidence: VocalicSemanticSeries, num_samples: int, seed: Optional[int] = None):
        pymc_model = self._define_pymc_model(evidence)
        with pymc_model:
            idata = pm.sample_prior_predictive(samples=num_samples, random_seed=seed)

        return pymc_model, idata

    def clear_parameter_values(self):
        self.coordination_cpn.parameters.clear_values()
        self.latent_vocalic_cpn.parameters.clear_values()
        self.semantic_link_cpn.parameters.clear_values()
        self.obs_vocalic_cpn.parameters.clear_values()

    @staticmethod
    def inference_data_to_posterior_samples(idata: az.InferenceData) -> VocalicSemanticPosteriorSamples:
        return VocalicSemanticPosteriorSamples.from_inference_data(idata)
