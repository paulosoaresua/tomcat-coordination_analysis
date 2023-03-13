from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

import arviz as az
from ast import literal_eval
import numpy as np
import pandas as pd
import pymc as pm
import xarray

from coordination.common.functions import logit
from coordination.component.coordination_component import SigmoidGaussianCoordinationComponent, \
    SigmoidGaussianCoordinationComponentSamples
from coordination.component.serialized_component import SerializedComponent, SerializedComponentSamples
from coordination.component.link_component import LinkComponent, LinkComponentSamples
from coordination.component.observation_component import SerializedObservationComponent, \
    SerializedObservationComponentSamples
from coordination.model.vocalic_model import VocalicSeries, VocalicPosteriorSamples


class VocalicSemanticSamples:

    def __init__(self, coordination: SigmoidGaussianCoordinationComponentSamples,
                 latent_vocalic: SerializedComponentSamples,
                 semantic_link: LinkComponentSamples, obs_vocalic: SerializedObservationComponentSamples):
        self.coordination = coordination
        self.latent_vocalic = latent_vocalic
        self.semantic_link = semantic_link
        self.obs_vocalic = obs_vocalic


class VocalicSemanticSeries:

    def __init__(self, uuid: str, vocalic_series: VocalicSeries,
                 semantic_link_time_steps_in_coordination_scale: np.ndarray):
        self.uuid = uuid
        self.vocalic = vocalic_series
        self.semantic_link_time_steps_in_coordination_scale = semantic_link_time_steps_in_coordination_scale

    @property
    def num_genders(self) -> int:
        return self.vocalic.num_genders

    @property
    def gender_map(self) -> Dict[int, int]:
        return self.vocalic.gender_map

    @classmethod
    def from_data_frame(cls, evidence_df: pd.DataFrame, vocalic_features: List[str]):
        vocalic_series = VocalicSeries.from_data_frame(evidence_df=evidence_df, vocalic_features=vocalic_features)

        return cls(
            uuid=vocalic_series.uuid,
            vocalic_series=vocalic_series,
            semantic_link_time_steps_in_coordination_scale=np.array(
                literal_eval(evidence_df["conversational_semantic_link_time_steps_in_coordination_scale"].values[0]),
                dtype=int)
        )

    def standardize(self):
        self.vocalic.standardize()

    def normalize_per_subject(self):
        self.vocalic.normalize_per_subject()

    def normalize_across_subject(self):
        self.vocalic.normalize_across_subject()

    @property
    def num_time_steps_in_semantic_link_scale(self) -> int:
        return self.semantic_link_time_steps_in_coordination_scale.shape[-1]


class VocalicSemanticPosteriorSamples:

    def __init__(self, unbounded_coordination: xarray.Dataset, coordination: xarray.Dataset,
                 latent_vocalic: xarray.Dataset):
        self.unbounded_coordination = unbounded_coordination
        self.coordination = coordination
        self.latent_vocalic = latent_vocalic

    @classmethod
    def from_inference_data(cls, idata: Any) -> VocalicSemanticPosteriorSamples:
        vocalic_posterior_samples = VocalicPosteriorSamples.from_inference_data(idata)

        return cls(vocalic_posterior_samples.unbounded_coordination, vocalic_posterior_samples.coordination,
                   vocalic_posterior_samples.latent_vocalic)


class VocalicSemanticModel:

    def __init__(self, num_subjects: int, vocalic_features: List[str], self_dependent: bool, sd_mean_uc0: float,
                 sd_sd_uc: float, sd_mean_a0_vocalic: np.ndarray, sd_sd_aa_vocalic: np.ndarray,
                 sd_sd_o_vocalic: np.ndarray, a_p_semantic_link: float, b_p_semantic_link: float,
                 share_params_across_subjects: bool, share_params_across_genders: bool,
                 initial_coordination: Optional[float] = None):
        self.num_subjects = num_subjects
        self.vocalic_features = vocalic_features
        self.share_params_across_subjects = share_params_across_subjects
        self.share_params_across_genders = share_params_across_genders

        self.coordination_cpn = SigmoidGaussianCoordinationComponent(sd_mean_uc0=sd_mean_uc0,
                                                                     sd_sd_uc=sd_sd_uc)
        if initial_coordination is not None:
            self.coordination_cpn.parameters.mean_uc0.value = np.array([logit(initial_coordination)])

        self.latent_vocalic_cpn = SerializedComponent(uuid="latent_vocalic",
                                                      num_subjects=num_subjects,
                                                      dim_value=len(vocalic_features),
                                                      self_dependent=self_dependent,
                                                      sd_mean_a0=sd_mean_a0_vocalic,
                                                      sd_sd_aa=sd_sd_aa_vocalic,
                                                      share_params_across_subjects=share_params_across_subjects,
                                                      share_params_across_genders=share_params_across_genders)
        self.semantic_link_cpn = LinkComponent("obs_semantic_link",
                                               a_p=a_p_semantic_link,
                                               b_p=b_p_semantic_link)
        self.obs_vocalic_cpn = SerializedObservationComponent(uuid="obs_vocalic",
                                                              num_subjects=num_subjects,
                                                              dim_value=len(vocalic_features),
                                                              sd_sd_o=sd_sd_o_vocalic,
                                                              share_params_across_subjects=share_params_across_subjects,
                                                              share_params_across_genders=share_params_across_genders)

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

    @property
    def obs_semantic_link_variable_name(self) -> str:
        return self.semantic_link_cpn.uuid

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
                                                                subjects=latent_vocalic_samples.subjects,
                                                                gender_map=latent_vocalic_samples.gender_map)

        samples = VocalicSemanticSamples(coordination=coordination_samples, latent_vocalic=latent_vocalic_samples,
                                         semantic_link=semantic_link_samples,
                                         obs_vocalic=obs_vocalic_samples)

        return samples

    def fit(self, evidence: VocalicSemanticSeries, burn_in: int, num_samples: int, num_chains: int,
            seed: Optional[int] = None, num_jobs: int = 1) -> Tuple[pm.Model, az.InferenceData]:
        assert evidence.vocalic.num_vocalic_features == len(self.vocalic_features)

        pymc_model = self._define_pymc_model(evidence)
        with pymc_model:
            idata = pm.sample(num_samples, init="jitter+adapt_diag", tune=burn_in, chains=num_chains, random_seed=seed,
                              cores=num_jobs)

        return pymc_model, idata

    def _define_pymc_model(self, evidence: VocalicSemanticSeries):
        coords = {"vocalic_feature": self.vocalic_features,
                  "coordination_time": np.arange(evidence.vocalic.num_time_steps_in_coordination_scale),
                  "vocalic_time": np.arange(evidence.vocalic.num_time_steps_in_vocalic_scale),
                  "link_time": np.arange(evidence.num_time_steps_in_semantic_link_scale)}

        pymc_model = pm.Model(coords=coords)
        with pymc_model:
            _, coordination, _ = self.coordination_cpn.update_pymc_model(time_dimension="coordination_time")
            latent_vocalic, _, _ = self.latent_vocalic_cpn.update_pymc_model(
                coordination=coordination[evidence.vocalic.time_steps_in_coordination_scale],
                prev_time_same_subject=evidence.vocalic.previous_time_same_subject,
                prev_time_diff_subject=evidence.vocalic.previous_time_diff_subject,
                prev_same_subject_mask=evidence.vocalic.vocalic_prev_same_subject_mask,
                prev_diff_subject_mask=evidence.vocalic.vocalic_prev_diff_subject_mask,
                subjects=evidence.vocalic.subjects_in_time,
                gender_map=evidence.vocalic.gender_map,
                time_dimension="vocalic_time",
                feature_dimension="vocalic_feature")

            self.semantic_link_cpn.update_pymc_model(
                coordination=coordination[evidence.num_time_steps_in_semantic_link_scale],
                time_dimension="link_time",
                observed_values=np.ones(evidence.num_time_steps_in_semantic_link_scale))

            self.obs_vocalic_cpn.update_pymc_model(latent_component=latent_vocalic,
                                                   subjects=evidence.vocalic.subjects_in_time,
                                                   gender_map=evidence.vocalic.gender_map,
                                                   feature_dimension="vocalic_feature",
                                                   time_dimension="vocalic_time",
                                                   observed_values=evidence.vocalic.observation)

        return pymc_model

    def prior_predictive(self, evidence: VocalicSemanticSeries, num_samples: int, seed: Optional[int] = None):
        pymc_model = self._define_pymc_model(evidence)
        with pymc_model:
            idata = pm.sample_prior_predictive(samples=num_samples, random_seed=seed)

        return pymc_model, idata

    def posterior_predictive(self, evidence: VocalicSemanticSeries, trace: az.InferenceData,
                             seed: Optional[int] = None):
        pymc_model = self._define_pymc_model(evidence)
        with pymc_model:
            idata = pm.sample_posterior_predictive(trace=trace, random_seed=seed)

        return pymc_model, idata

    def clear_parameter_values(self):
        self.coordination_cpn.parameters.clear_values()
        self.latent_vocalic_cpn.parameters.clear_values()
        self.semantic_link_cpn.parameters.clear_values()
        self.obs_vocalic_cpn.parameters.clear_values()

    @staticmethod
    def inference_data_to_posterior_samples(idata: az.InferenceData) -> VocalicSemanticPosteriorSamples:
        return VocalicSemanticPosteriorSamples.from_inference_data(idata)
