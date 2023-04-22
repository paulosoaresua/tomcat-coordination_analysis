from __future__ import annotations
from typing import Any, Callable, List, Optional, Tuple

from copy import deepcopy
import sys

import arviz as az
import matplotlib.pyplot as plt
import math
import numpy as np
import random
import pandas as pd
import pymc as pm

from coordination.common.functions import logit, one_hot_encode
from coordination.component.coordination_component import SigmoidGaussianCoordinationComponent
from coordination.component.neural_network import NeuralNetwork
from coordination.component.mixture_component import MixtureComponent
from coordination.component.observation_component import ObservationComponent
from coordination.component.lag import Lag
from coordination.model.coordination_model import CoordinationPosteriorSamples


class SyntheticSeriesMixture:

    def __init__(self,
                 values: np.ndarray,
                 num_time_steps_in_coordination_scale: int,
                 time_steps_in_coordination_scale: np.ndarray):

        self.values = values
        self.num_time_steps_in_coordination_scale = num_time_steps_in_coordination_scale
        self.time_steps_in_coordination_scale = time_steps_in_coordination_scale

    @classmethod
    def from_function(cls,
                      fn: Callable,
                      num_subjects: int,
                      time_steps: np.ndarray,  # It must be a matrix with dimensions: num_subjects x time_steps
                      noise_scale: Optional[float] = None,
                      vertical_offset_per_subject: Optional[np.ndarray] = None,
                      horizontal_offset_per_subject: Optional[np.ndarray] = None) -> SyntheticSeriesMixture:

        assert vertical_offset_per_subject is None or len(vertical_offset_per_subject) == num_subjects
        assert horizontal_offset_per_subject is None or len(horizontal_offset_per_subject) == num_subjects

        num_time_steps = time_steps.shape[-1]

        noise = np.random.normal(size=(num_subjects, num_time_steps),
                                 scale=noise_scale) if noise_scale is not None else 0

        if horizontal_offset_per_subject is not None:
            time_steps += horizontal_offset_per_subject[:, None]

        values = fn(time_steps) + noise

        if vertical_offset_per_subject is not None:
            values += vertical_offset_per_subject[:, None]

        return cls(values=values[:, None, :],  # Subject x feature x time
                   num_time_steps_in_coordination_scale=num_time_steps,
                   time_steps_in_coordination_scale=np.arange(num_time_steps))

    @property
    def num_time_steps_in_component_scale(self) -> int:
        return self.values.shape[-1]

    def normalize_per_subject(self, inplace: bool):
        mean = self.values.mean(axis=-1)[..., None]
        std = self.values.std(axis=-1)[..., None]
        new_values = (self.values - mean) / std

        if inplace:
            self.values = new_values
            return self
        else:
            return SyntheticSeriesMixture(values=new_values,
                                          num_time_steps_in_coordination_scale=self.num_time_steps_in_component_scale,
                                          time_steps_in_coordination_scale=self.time_steps_in_coordination_scale)

    def plot(self, ax: Any, marker_size: int):
        for series in self.values:
            ax.scatter(self.time_steps_in_coordination_scale, series[0], s=marker_size)
            ax.set_xlabel("Time Step")
            ax.set_ylabel("Value")


class MixtureModel:

    def __init__(self,
                 num_subjects: int,
                 self_dependent: bool,
                 sd_mean_uc0: float,
                 sd_sd_uc: float,
                 mean_mean_a0: np.ndarray,
                 sd_mean_a0: np.ndarray,
                 sd_sd_aa: np.ndarray,
                 a_mixture_weights: np.ndarray,
                 sd_sd_o: np.ndarray,
                 share_params_across_subjects: bool,
                 share_params_across_features_latent: bool,
                 share_params_across_features_observation: bool,
                 initial_coordination: Optional[float] = None,
                 num_hidden_layers_f: int = 0,
                 dim_hidden_layer_f: int = 0,
                 activation_function_name_f="linear",
                 num_hidden_layers_g: int = 0,
                 dim_hidden_layer_g: int = 0,
                 activation_function_name_g="linear",
                 max_lag: int = 0):

        self.num_subjects = num_subjects
        self.num_hidden_layers_f = num_hidden_layers_f
        self.dim_hidden_layer_f = dim_hidden_layer_f
        self.activation_function_name_f = activation_function_name_f

        self.coordination_cpn = SigmoidGaussianCoordinationComponent(sd_mean_uc0=sd_mean_uc0,
                                                                     sd_sd_uc=sd_sd_uc)

        if initial_coordination is not None:
            self.coordination_cpn.parameters.mean_uc0.value = np.array([logit(initial_coordination)])

        self.lag_cpn = Lag("lag", max_lag=max_lag)

        self.latent_cpn = MixtureComponent(uuid="mixture_component",
                                           num_subjects=num_subjects,
                                           dim_value=1,
                                           self_dependent=self_dependent,
                                           mean_mean_a0=mean_mean_a0,
                                           sd_mean_a0=sd_mean_a0,
                                           sd_sd_aa=sd_sd_aa,
                                           a_mixture_weights=a_mixture_weights,
                                           share_params_across_subjects=share_params_across_subjects,
                                           share_params_across_features=share_params_across_features_latent)

        if num_hidden_layers_g > 0:
            self.g_nn = NeuralNetwork(uuid="g",
                                      num_hidden_layers=num_hidden_layers_g,
                                      dim_hidden_layer=dim_hidden_layer_g,
                                      activation_function_name=activation_function_name_g)
        else:
            # No transformation between latent and observation
            self.g_nn = None

        self.observation_cpn = ObservationComponent(uuid="observation_component",
                                                    num_subjects=num_subjects,
                                                    dim_value=1,
                                                    sd_sd_o=sd_sd_o,
                                                    share_params_across_subjects=share_params_across_subjects,
                                                    share_params_across_features=share_params_across_features_observation)

    @property
    def parameter_names(self) -> List[str]:
        names = self.coordination_cpn.parameter_names
        names.extend(self.latent_cpn.parameter_names)
        names.extend(self.observation_cpn.parameter_names)
        names.extend(self.lag_cpn.parameter_names)

        return names

    def fit(self, evidence: SyntheticSeriesMixture, burn_in: int, num_samples: int, num_chains: int,
            seed: Optional[int] = None, num_jobs: int = 1, init_method: str = "jitter+adapt_diag") -> Tuple[
        pm.Model, az.InferenceData]:

        pymc_model = self._define_pymc_model(evidence)
        with pymc_model:
            idata = pm.sample(num_samples,
                              init=init_method,
                              tune=burn_in,
                              chains=num_chains,
                              random_seed=seed,
                              cores=num_jobs)

        return pymc_model, idata

    def _define_pymc_model(self, evidence: SyntheticSeriesMixture):
        coords = {"component_subject": np.arange(self.num_subjects),
                  "component_feature": np.array([1]),
                  "coordination_time": np.arange(evidence.num_time_steps_in_coordination_scale),
                  "component_time": np.arange(evidence.num_time_steps_in_component_scale)}

        pymc_model = pm.Model(coords=coords)
        with pymc_model:
            coordination = self.coordination_cpn.update_pymc_model(time_dimension="coordination_time")[1]
            # lag = self.lag_cpn.update_pymc_model(math.comb(self.num_subjects, 2))
            latent_component = self.latent_cpn.update_pymc_model(
                coordination=coordination[evidence.time_steps_in_coordination_scale],
                # lag=lag,
                subject_dimension="component_subject",
                feature_dimension="component_feature",
                time_dimension="component_time",
                num_hidden_layers_f=self.num_hidden_layers_f,
                dim_hidden_layer_f=self.dim_hidden_layer_f,
                activation_function_name_f=self.activation_function_name_f)[0]

            obs_input = latent_component
            if self.g_nn is not None:
                # features + subject id (one hot encode) + time step + bias term
                X = pm.Deterministic("augmented_latent_component",
                                     pm.math.concatenate(
                                         [latent_component,
                                          one_hot_encode(evidence.subjects_in_time, self.num_subjects),
                                          evidence.time_steps_in_coordination_scale[None, :]]))
                outputs = self.g_nn.update_pymc_model(input_data=X.transpose(),
                                                      output_dim=latent_component.shape[0])[0]

                obs_input = outputs.transpose()

            self.observation_cpn.update_pymc_model(latent_component=obs_input,
                                                   subject_dimension="component_subject",
                                                   feature_dimension="component_feature",
                                                   time_dimension="component_time",
                                                   observed_values=evidence.values)

        return pymc_model


def train(model: Any,
          evidence: Any,
          init_method: str = "jitter+adapt_diag",
          burn_in: int = 1000,
          num_samples: int = 1000,
          num_chains: int = 2,
          seed: int = 0):
    # Ignore PyMC warnings
    if not sys.warnoptions:
        import warnings
        warnings.simplefilter("ignore")

    _, idata = model.fit(evidence=evidence,
                         init_method=init_method,
                         burn_in=burn_in,
                         num_samples=num_samples,
                         num_chains=num_chains,
                         seed=seed,
                         num_jobs=num_chains)

    posterior_samples = CoordinationPosteriorSamples.from_inference_data(idata)

    return posterior_samples, idata


def plot_parameter_trace(model: Any, idata: Any):
    sampled_vars = set(idata.posterior.data_vars)
    var_names = sorted(list(set(model.parameter_names).intersection(sampled_vars)))
    az.plot_trace(idata, var_names=var_names)
    plt.tight_layout()


if __name__ == "__main__":
    SEED = 0
    random.seed(SEED)
    np.random.seed(SEED)

    # Synthetic data
    evidence_vertical_shift = SyntheticSeriesMixture.from_function(fn=np.cos,
                                                                   num_subjects=3,
                                                                   time_steps=np.linspace(0, 4 * np.pi, 60).reshape(20,
                                                                                                                    3).T,
                                                                   noise_scale=None,
                                                                   vertical_offset_per_subject=np.array([0, 1, 2]))
    evidence_vertical_shift_normalized = evidence_vertical_shift.normalize_per_subject(inplace=False)

    evidence_vertical_shift_noise = SyntheticSeriesMixture.from_function(fn=np.sin,
                                                                         num_subjects=3,
                                                                         time_steps=np.linspace(0, 4 * np.pi,
                                                                                                60).reshape(20, 3).T,
                                                                         noise_scale=0.5,
                                                                         vertical_offset_per_subject=np.array(
                                                                             [0, 1, 2]))
    evidence_vertical_shift_noise_normalized = evidence_vertical_shift_noise.normalize_per_subject(inplace=False)

    # The second person is anti-symmetric with respect to the first and third one
    evidence_vertical_shift_anti_symmetry = SyntheticSeriesMixture.from_function(fn=np.sin,
                                                                                 num_subjects=3,
                                                                                 time_steps=np.linspace(0, 4 * np.pi,
                                                                                                        60).reshape(20,
                                                                                                                    3).T,
                                                                                 noise_scale=None,
                                                                                 vertical_offset_per_subject=np.array(
                                                                                     [0, 1, 2]),
                                                                                 horizontal_offset_per_subject=np.array(
                                                                                     [0, np.pi, 0]))
    evidence_vertical_shift_anti_symmetry_normalized = evidence_vertical_shift_anti_symmetry.normalize_per_subject(
        inplace=False)

    evidence_random = SyntheticSeriesMixture.from_function(fn=lambda x: np.random.randn(*x.shape),
                                                           num_subjects=3,
                                                           time_steps=np.linspace(0, 4 * np.pi, 60).reshape(20, 3).T,
                                                           noise_scale=None)
    evidence_random_normalized = evidence_random.normalize_per_subject(inplace=False)

    evidence_vertical_shift_lag = SyntheticSeriesMixture.from_function(fn=np.sin,
                                                                       num_subjects=3,
                                                                       time_steps=np.linspace(0, 4 * np.pi, 60).reshape(
                                                                           20, 3).T,
                                                                       noise_scale=None,
                                                                       vertical_offset_per_subject=np.array([0, 1, 2]),
                                                                       horizontal_offset_per_subject=np.array(
                                                                           [0, np.pi, np.pi / 2]))
    evidence_vertical_shift_lag_normalized = evidence_vertical_shift_lag.normalize_per_subject(inplace=False)

    # Model to test
    evidence = evidence_vertical_shift_anti_symmetry_normalized

    model = MixtureModel(num_subjects=3,
                         self_dependent=True,
                         sd_mean_uc0=5,
                         sd_sd_uc=1,
                         mean_mean_a0=np.zeros(1),
                         sd_mean_a0=np.ones(1),
                         sd_sd_aa=np.ones(1),
                         a_mixture_weights=np.ones((3, 2)),
                         sd_sd_o=np.ones(1),
                         share_params_across_subjects=True,
                         share_params_across_features_latent=False,
                         share_params_across_features_observation=False,
                         initial_coordination=None,
                         num_hidden_layers_f=2,
                         dim_hidden_layer_f=8,
                         activation_function_name_f="tanh")

    posterior_samples, idata_vertical_shift_anti_symmetry_normalized_fit_f = train(model, evidence, burn_in=1500)

    # Plot parameter trace
    plot_parameter_trace(model, idata_vertical_shift_anti_symmetry_normalized_fit_f)

    # Plot evidence and coordination side by side
    fig, axs = plt.subplots(1, 2)

    evidence.plot(axs[0], marker_size=8)
    posterior_samples.plot(axs[1], show_samples=False, line_width=1)
