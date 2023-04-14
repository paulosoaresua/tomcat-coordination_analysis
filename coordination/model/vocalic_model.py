from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Tuple

import arviz as az
from ast import literal_eval
import numpy as np
import pandas as pd
import pymc as pm
import xarray

from coordination.common.functions import logit
from coordination.component.coordination_component import SigmoidGaussianCoordinationComponent, \
    CoordinationComponentSamples, BetaGaussianCoordinationComponent
from coordination.component.serialized_component import SerializedComponent, SerializedComponentSamples, Mode
from coordination.component.observation_component import SerializedObservationComponent, \
    SerializedObservationComponentSamples
from coordination.model.coordination_model import CoordinationPosteriorSamples

VOCALIC_FEATURES = [
    "pitch",
    "intensity",
    "jitter",
    "shimmer"
]


class VocalicSamples:

    def __init__(self, coordination: CoordinationComponentSamples,
                 latent_vocalic: SerializedComponentSamples, obs_vocalic: SerializedObservationComponentSamples):
        self.coordination = coordination
        self.latent_vocalic = latent_vocalic
        self.obs_vocalic = obs_vocalic


class VocalicSeries:

    def __init__(self, uuid: str, features: List[str], num_time_steps_in_coordination_scale: int,
                 subjects_in_time: np.ndarray, observation: np.ndarray, previous_time_same_subject: np.ndarray,
                 previous_time_diff_subject: np.ndarray, time_steps_in_coordination_scale: np.ndarray,
                 gender_map: Dict[int, int]):
        self.uuid = uuid
        self.features = features
        self.num_time_steps_in_coordination_scale = num_time_steps_in_coordination_scale
        self.subjects_in_time = subjects_in_time
        self.observation = observation
        self.previous_time_same_subject = previous_time_same_subject
        self.previous_time_diff_subject = previous_time_diff_subject
        self.time_steps_in_coordination_scale = time_steps_in_coordination_scale
        self.gender_map = gender_map

    @property
    def num_genders(self) -> int:
        return len(set(list(self.gender_map.values())))

    def chop(self, min_time_step: int, max_time_step: int):
        """
        Chops the series into a pre-defined range.
        """
        self.num_time_steps_in_coordination_scale = max_time_step - min_time_step
        t_min_vocalic = 0
        t_max_vocalic = 0
        for t in range(self.num_time_steps_in_vocalic_scale):
            if self.time_steps_in_coordination_scale[t] < min_time_step:
                t_min_vocalic = t + 1

            if self.time_steps_in_coordination_scale[t] < max_time_step:
                t_max_vocalic = t + 1

        self.subjects_in_time = self.subjects_in_time[t_min_vocalic:t_max_vocalic]
        self.observation = self.observation[:, t_min_vocalic:t_max_vocalic]
        self.previous_time_same_subject = np.maximum(
            self.previous_time_same_subject[t_min_vocalic:t_max_vocalic] - t_min_vocalic, -1)
        self.previous_time_diff_subject = np.maximum(
            self.previous_time_diff_subject[t_min_vocalic:t_max_vocalic] - t_min_vocalic, -1)
        self.time_steps_in_coordination_scale = self.time_steps_in_coordination_scale[
                                                t_min_vocalic:t_max_vocalic] - min_time_step

    def standardize(self):
        """
        Make sure measurements are between 0 and 1 and per feature. Don't normalize per subject otherwise we lose
        proximity relativity (how close measurements from different subjects are) which is important for the
        coordination model.
        """

        max_value = self.observation.max(axis=-1, initial=0)[:, None]
        min_value = self.observation.min(axis=-1, initial=0)[:, None]
        self.observation = (self.observation - min_value) / (max_value - min_value)

    def normalize_per_subject(self):
        """
        Make sure measurements have mean 0 and standard deviation 1 per subject and feature.
        """
        all_subjects = set(self.subjects_in_time)

        for subject in all_subjects:
            obs_per_subject = self.observation[:, self.subjects_in_time == subject]
            mean = obs_per_subject.mean(axis=1)[:, None]
            std = obs_per_subject.std(axis=1)[:, None]
            self.observation[:, self.subjects_in_time == subject] = (obs_per_subject - mean) / std

    def normalize_per_gender(self):
        """
        Make sure measurements have mean 0 and standard deviation 1 per gender and feature.
        """

        genders_in_time = np.array([self.gender_map[s] for s in self.subjects_in_time])

        for gender in [0, 1]:  # Male and Female
            obs_per_gender = self.observation[:, genders_in_time == gender]
            mean = obs_per_gender.mean(axis=1)[:, None]
            std = obs_per_gender.std(axis=1)[:, None]
            self.observation[:, genders_in_time == gender] = (obs_per_gender - mean) / std

    def normalize_across_subject(self):
        """
        Make sure measurements have mean 0 and standard deviation 1 per feature.
        """

        mean = self.observation.mean(axis=1)[:, None]
        std = self.observation.std(axis=1)[:, None]
        self.observation = (self.observation - mean) / std

    def plot_observations(self, axs: List[Any]):
        # One plot per channel
        all_subjects = set(self.subjects_in_time)

        for vocalic_feature_idx in range(min(self.num_vocalic_features, len(axs))):
            ax = axs[vocalic_feature_idx]
            for subject in all_subjects:
                subject_mask = self.subjects_in_time == subject
                xs = self.time_steps_in_coordination_scale[subject_mask]
                ys = self.observation[vocalic_feature_idx, subject_mask]
                if len(xs) == 1:
                    ax.scatter(xs, ys, label=subject)
                else:
                    ax.plot(xs, ys, label=subject, marker="o")

            ax.set_title(self.features[vocalic_feature_idx])
            ax.set_xlabel("Time Step")
            ax.set_ylabel("Observed Value")
            ax.set_xlim([-0.5, self.num_time_steps_in_coordination_scale + 0.5])
            ax.legend()

    def plot_observation_differences(self, axs: List[Any], self_dependent: bool):
        # Plot the difference between the current subject's vocalic and their previous vocalic and a different
        # subject's previous vocalic

        for vocalic_feature_idx in range(min(self.num_vocalic_features, len(axs))):
            ax = axs[vocalic_feature_idx]

            xs_same = []
            ys_same = []
            xs_diff = []
            ys_diff = []

            fixed_means = {}
            if not self_dependent:
                # Approximated fixed mean is the first observation of a subject
                all_subjects = set(self.subjects_in_time)
                for subject in all_subjects:
                    t0 = np.where(self.subjects_in_time == subject)[0][0]
                    fixed_means[subject] = self.observation[vocalic_feature_idx, t0]

            for t in range(self.num_time_steps_in_vocalic_scale):
                if self_dependent:
                    if self.previous_time_same_subject[t] >= 0:
                        t_p = self.previous_time_same_subject[t]
                        xs_same.append(self.time_steps_in_coordination_scale[t])
                        ys_same.append(
                            np.abs(
                                self.observation[vocalic_feature_idx, t] - self.observation[vocalic_feature_idx, t_p]))
                else:
                    subject = self.subjects_in_time[t]
                    xs_same.append(self.time_steps_in_coordination_scale[t])
                    ys_same.append(np.abs(self.observation[vocalic_feature_idx, t] - fixed_means[subject]))

                if self.previous_time_diff_subject[t] >= 0:
                    t_p = self.previous_time_diff_subject[t]
                    xs_diff.append(self.time_steps_in_coordination_scale[t])
                    ys_diff.append(
                        np.abs(self.observation[vocalic_feature_idx, t] - self.observation[vocalic_feature_idx, t_p]))

            if len(xs_same) == 1:
                ax.scatter(xs_same, ys_same, label="Same Subject")
            else:
                ax.plot(xs_same, ys_same, label="Same Subject", marker="o")

            if len(xs_same) == 1:
                ax.scatter(xs_diff, ys_diff, label="Different Subject")
            else:
                ax.plot(xs_diff, ys_diff, label="Different Subject", marker="o")

            ax.set_title(self.features[vocalic_feature_idx])
            ax.set_xlabel("Time Step")
            ax.set_ylabel("Difference btw Observed Values")
            ax.set_xlim([-0.5, self.num_time_steps_in_coordination_scale + 0.5])
            ax.legend()

    @classmethod
    def from_data_frame(cls, evidence_df: pd.DataFrame, vocalic_features: List[str]):
        obs_vocalic = []
        for vocalic_feature in vocalic_features:
            obs_vocalic.append(np.array(literal_eval(evidence_df[f"{vocalic_feature}"].values[0])))
        # Swap axes such that the first dimension represents the different subjects and the second the vocalic features
        obs_vocalic = np.array(obs_vocalic)

        gender_map = {}
        gender_cols = ["red_gender", "green_gender", "blue_gender"]
        for i, gender in enumerate(evidence_df[gender_cols].values[0]):
            if gender == "M":
                gender_map[i] = 0
            elif gender == "F":
                gender_map[i] = 1
            else:
                gender_map[i] = np.random.choice([0, 1])

        return cls(
            uuid=evidence_df["experiment_id"].values[0],
            features=vocalic_features,
            num_time_steps_in_coordination_scale=evidence_df["num_time_steps_in_coordination_scale"].values[0],
            subjects_in_time=np.array(literal_eval(evidence_df["vocalic_subjects"].values[0]), dtype=int),
            observation=obs_vocalic,
            previous_time_same_subject=np.array(
                literal_eval(evidence_df["vocalic_previous_time_same_subject"].values[0]), dtype=int),
            previous_time_diff_subject=np.array(
                literal_eval(evidence_df["vocalic_previous_time_diff_subject"].values[0]), dtype=int),
            time_steps_in_coordination_scale=np.array(
                literal_eval(evidence_df["vocalic_time_steps_in_coordination_scale"].values[0]), dtype=int),
            gender_map=gender_map
        )

    @property
    def num_time_steps_in_vocalic_scale(self) -> int:
        return self.observation.shape[-1]

    @property
    def num_vocalic_features(self) -> int:
        return self.observation.shape[-2]

    @property
    def vocalic_prev_same_subject_mask(self) -> np.ndarray:
        return np.where(self.previous_time_same_subject >= 0, 1, 0)

    @property
    def vocalic_prev_diff_subject_mask(self) -> np.ndarray:
        return np.where(self.previous_time_diff_subject >= 0, 1, 0)


class VocalicPosteriorSamples:

    def __init__(self, unbounded_coordination: xarray.Dataset, coordination: xarray.Dataset,
                 latent_vocalic: xarray.Dataset):
        self.unbounded_coordination = unbounded_coordination
        self.coordination = coordination
        self.latent_vocalic = latent_vocalic

    @classmethod
    def from_inference_data(cls, idata: Any) -> VocalicPosteriorSamples:
        coordination_posterior_samples = CoordinationPosteriorSamples.from_inference_data(idata)
        unbounded_coordination = coordination_posterior_samples.unbounded_coordination
        coordination = coordination_posterior_samples.coordination
        latent_vocalic = idata.posterior["latent_vocalic"]

        return cls(unbounded_coordination, coordination, latent_vocalic)


class VocalicModel:

    def __init__(self, num_subjects: int, vocalic_features: List[str],
                 self_dependent: bool, sd_mean_uc0: float, sd_sd_uc: float, mean_mean_a0_vocalic: np.ndarray,
                 sd_mean_a0_vocalic: np.ndarray, sd_sd_aa_vocalic: np.ndarray, sd_sd_o_vocalic: np.ndarray,
                 share_params_across_subjects: bool, share_params_across_genders: bool,
                 share_params_across_features_latent: bool, share_params_across_features_observation: bool,
                 initial_coordination: Optional[float] = None, sd_sd_c: Optional[float] = None,
                 mode: Mode = Mode.BLENDING, f: Optional[Callable] = None, num_hidden_layers_f: int = 0,
                 activation_function_f: str = "linear"):

        # Either one or the other
        assert not (share_params_across_genders and share_params_across_subjects)

        self.num_subjects = num_subjects
        self.vocalic_features = vocalic_features
        self.share_params_across_subjects = share_params_across_subjects
        self.share_params_across_genders = share_params_across_genders
        self.num_hidden_layers_f = num_hidden_layers_f
        self.activation_function_f = activation_function_f

        if sd_sd_c is None:
            # Coordination is a deterministic transformation of its unbounded estimate
            self.coordination_cpn = SigmoidGaussianCoordinationComponent(sd_mean_uc0=sd_mean_uc0,
                                                                         sd_sd_uc=sd_sd_uc)
        else:
            # Coordination is a latent variable centered around its unbounded estimate
            self.coordination_cpn = BetaGaussianCoordinationComponent(sd_mean_uc0=sd_mean_uc0,
                                                                      sd_sd_uc=sd_sd_uc,
                                                                      sd_sd_c=sd_sd_c)

        if initial_coordination is not None:
            self.coordination_cpn.parameters.mean_uc0.value = np.array([logit(initial_coordination)])

        self.latent_vocalic_cpn = SerializedComponent(uuid="latent_vocalic",
                                                      num_subjects=num_subjects,
                                                      dim_value=len(vocalic_features),
                                                      self_dependent=self_dependent,
                                                      mean_mean_a0=mean_mean_a0_vocalic,
                                                      sd_mean_a0=sd_mean_a0_vocalic,
                                                      sd_sd_aa=sd_sd_aa_vocalic,
                                                      share_params_across_subjects=share_params_across_subjects,
                                                      share_params_across_genders=share_params_across_genders,
                                                      share_params_across_features=share_params_across_features_latent,
                                                      mode=mode,
                                                      f=f)
        self.obs_vocalic_cpn = SerializedObservationComponent(uuid="obs_vocalic",
                                                              num_subjects=num_subjects,
                                                              dim_value=len(vocalic_features),
                                                              sd_sd_o=sd_sd_o_vocalic,
                                                              share_params_across_subjects=share_params_across_subjects,
                                                              share_params_across_genders=share_params_across_genders,
                                                              share_params_across_features=share_params_across_features_observation)

    @property
    def parameter_names(self) -> List[str]:
        names = self.coordination_cpn.parameter_names
        names.extend(self.latent_vocalic_cpn.parameter_names)
        names.extend(self.obs_vocalic_cpn.parameter_names)

        return names

    @property
    def obs_vocalic_variable_name(self) -> str:
        return self.obs_vocalic_cpn.uuid

    def draw_samples(self, num_series: int, num_time_steps: int, vocalic_time_scale_density: float,
                     can_repeat_subject: bool, seed: Optional[int] = None) -> VocalicSamples:
        coordination_samples = self.coordination_cpn.draw_samples(num_series, num_time_steps, seed)
        latent_vocalic_samples = self.latent_vocalic_cpn.draw_samples(num_series=num_series,
                                                                      time_scale_density=vocalic_time_scale_density,
                                                                      coordination=coordination_samples.coordination,
                                                                      can_repeat_subject=can_repeat_subject)
        obs_vocalic_samples = self.obs_vocalic_cpn.draw_samples(latent_component=latent_vocalic_samples.values,
                                                                subjects=latent_vocalic_samples.subjects,
                                                                gender_map=latent_vocalic_samples.gender_map)

        samples = VocalicSamples(coordination=coordination_samples, latent_vocalic=latent_vocalic_samples,
                                 obs_vocalic=obs_vocalic_samples)

        return samples

    def fit(self, evidence: VocalicSeries, burn_in: int, num_samples: int, num_chains: int,
            seed: Optional[int] = None, num_jobs: int = 1) -> Tuple[pm.Model, az.InferenceData]:
        assert evidence.num_vocalic_features == len(self.vocalic_features)

        pymc_model = self._define_pymc_model(evidence)
        with pymc_model:
            idata = pm.sample(num_samples, init="jitter+adapt_diag", tune=burn_in, chains=num_chains, random_seed=seed,
                              cores=num_jobs)

        return pymc_model, idata

    def _define_pymc_model(self, evidence: VocalicSeries):
        coords = {"vocalic_feature": self.vocalic_features,
                  "coordination_time": np.arange(evidence.num_time_steps_in_coordination_scale),
                  "vocalic_time": np.arange(evidence.num_time_steps_in_vocalic_scale)}

        if self.share_params_across_genders:
            subjects_in_time = np.array([evidence.gender_map[subject] for subject in evidence.subjects_in_time])
        else:
            subjects_in_time = evidence.subjects_in_time

        pymc_model = pm.Model(coords=coords)
        with pymc_model:
            coordination = self.coordination_cpn.update_pymc_model(time_dimension="coordination_time")[1]
            latent_vocalic = self.latent_vocalic_cpn.update_pymc_model(
                coordination=coordination[evidence.time_steps_in_coordination_scale],
                prev_time_same_subject=evidence.previous_time_same_subject,
                prev_time_diff_subject=evidence.previous_time_diff_subject,
                prev_same_subject_mask=evidence.vocalic_prev_same_subject_mask,
                prev_diff_subject_mask=evidence.vocalic_prev_diff_subject_mask,
                subjects=subjects_in_time,
                gender_map=evidence.gender_map,
                time_dimension="vocalic_time",
                feature_dimension="vocalic_feature",
                num_hidden_layers_f=self.num_hidden_layers_f,
                activation_function_f=self.activation_function_f)[0]

            self.obs_vocalic_cpn.update_pymc_model(latent_component=latent_vocalic,
                                                   feature_dimension="vocalic_feature",
                                                   time_dimension="vocalic_time",
                                                   subjects=subjects_in_time,
                                                   gender_map=evidence.gender_map,
                                                   observed_values=evidence.observation)

        return pymc_model

    def prior_predictive(self, evidence: VocalicSeries, num_samples: int, seed: Optional[int] = None):
        pymc_model = self._define_pymc_model(evidence)
        with pymc_model:
            idata = pm.sample_prior_predictive(samples=num_samples, random_seed=seed)

        return pymc_model, idata

    def posterior_predictive(self, evidence: VocalicSeries, trace: az.InferenceData, seed: Optional[int] = None):
        pymc_model = self._define_pymc_model(evidence)
        with pymc_model:
            idata = pm.sample_posterior_predictive(trace=trace, random_seed=seed)

        return pymc_model, idata

    def clear_parameter_values(self):
        self.coordination_cpn.parameters.clear_values()
        self.latent_vocalic_cpn.parameters.clear_values()
        self.obs_vocalic_cpn.parameters.clear_values()

    @staticmethod
    def inference_data_to_posterior_samples(idata: az.InferenceData) -> VocalicPosteriorSamples:
        return VocalicPosteriorSamples.from_inference_data(idata)