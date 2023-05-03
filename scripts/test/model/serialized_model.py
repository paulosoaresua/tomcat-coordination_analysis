from __future__ import annotations
from typing import Any, Callable, List, Optional, Tuple

import sys

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
import pymc as pm

from coordination.common.functions import logit, one_hot_encode
from coordination.component.coordination_component import SigmoidGaussianCoordinationComponent
from coordination.component.neural_network import NeuralNetwork
from coordination.component.serialized_component import SerializedComponent, Mode
from coordination.component.observation_component import SerializedObservationComponent
from coordination.model.coordination_model import CoordinationPosteriorSamples


class SyntheticSeriesSerialized:

    def __init__(self,
                 subjects_in_time: np.ndarray,
                 values: np.ndarray,
                 num_time_steps_in_coordination_scale: int,
                 time_steps_in_coordination_scale: np.ndarray,
                 previous_time_same_subject: np.ndarray,
                 previous_time_diff_subject: np.ndarray):

        self.subjects_in_time = subjects_in_time
        self.values = values
        self.num_time_steps_in_coordination_scale = num_time_steps_in_coordination_scale
        self.time_steps_in_coordination_scale = time_steps_in_coordination_scale
        self.previous_time_same_subject = previous_time_same_subject
        self.previous_time_diff_subject = previous_time_diff_subject

    @classmethod
    def from_function(cls,
                      fn: Callable,
                      num_subjects: int,
                      num_time_steps: int,
                      time_step_size: float,
                      max_lag: int,
                      noise_scale: Optional[float] = None,
                      vertical_offset_per_subject: Optional[np.ndarray] = None,
                      horizontal_offset_per_subject: Optional[np.ndarray] = None) -> SyntheticSeriesSerialized:

        assert vertical_offset_per_subject is None or len(vertical_offset_per_subject) == num_subjects
        assert horizontal_offset_per_subject is None or len(horizontal_offset_per_subject) == num_subjects
        assert horizontal_offset_per_subject is None or (horizontal_offset_per_subject <= max_lag).all()

        max_lag *= num_subjects

        if horizontal_offset_per_subject is None:
            horizontal_offset_per_subject = np.zeros(num_subjects, dtype=int)

        extra_time_steps = 2 * max_lag

        values = np.zeros((num_subjects, num_time_steps + extra_time_steps))
        for t in range(num_time_steps + extra_time_steps):
            values[:, t] = fn((t - max_lag) * time_step_size) + np.arange(num_subjects)

        # Apply lags
        for s in range(num_subjects):
            values[s] = np.roll(values[s], horizontal_offset_per_subject[s] * num_subjects)

        # Remove extra time steps
        values = values[:, max_lag:(values.shape[-1] - max_lag)]

        # Serialize speakers
        subjects_in_time = np.tile(np.arange(num_subjects), int(np.ceil(num_time_steps / num_subjects)))
        subjects_in_time = subjects_in_time[:num_time_steps]
        previous_time_same_subject = np.arange(num_time_steps) - num_subjects
        previous_time_diff_subject = np.arange(num_time_steps) - 1

        for t, s in enumerate(subjects_in_time):
            values[0, t] = values[s, t]

        values = values[0]

        noise = np.random.normal(size=num_time_steps, scale=noise_scale) if noise_scale is not None else 0

        values += noise

        #
        #
        #
        #
        # num_time_steps = len(time_steps)
        #
        # noise = np.random.normal(size=num_time_steps, scale=noise_scale) if noise_scale is not None else 0
        #
        # subjects_in_time = np.tile(np.arange(num_subjects), int(np.ceil(num_time_steps / num_subjects)))
        # subjects_in_time = subjects_in_time[:num_time_steps]
        #
        # if horizontal_offset_per_subject is not None:
        #     offsets = np.tile(horizontal_offset_per_subject, int(np.ceil(num_time_steps / num_subjects)))
        #     offsets = offsets[:num_time_steps]
        #     time_steps += offsets
        #
        # values = fn(time_steps) + noise

        if vertical_offset_per_subject is not None:
            offsets = np.tile(vertical_offset_per_subject, int(np.ceil(num_time_steps / num_subjects)))
            offsets = offsets[:num_time_steps]
            values += offsets

        return cls(subjects_in_time=subjects_in_time,
                   values=values[None, :],
                   num_time_steps_in_coordination_scale=num_time_steps,
                   time_steps_in_coordination_scale=np.arange(num_time_steps),
                   previous_time_same_subject=previous_time_same_subject,
                   previous_time_diff_subject=previous_time_diff_subject)

    @property
    def num_time_steps_in_component_scale(self) -> int:
        return self.values.shape[-1]

    @property
    def prev_same_subject_mask(self) -> np.ndarray:
        return np.where(self.previous_time_same_subject >= 0, 1, 0)

    @property
    def prev_diff_subject_mask(self) -> np.ndarray:
        return np.where(self.previous_time_diff_subject >= 0, 1, 0)

    def values_per_subject(self, subject: int) -> Tuple[np.ndarray, np.ndarray]:
        x, = np.where(self.subjects_in_time == subject)
        y = self.values[:, x]

        return x, y

    def normalize_per_subject(self, inplace: bool):
        all_subjects = set(self.subjects_in_time)

        new_values = self.values.copy()
        for subject in all_subjects:
            values_per_subject = self.values[:, self.subjects_in_time == subject]
            mean = values_per_subject.mean(axis=1)[:, None]
            std = values_per_subject.std(axis=1)[:, None]
            new_values[:, self.subjects_in_time == subject] = (values_per_subject - mean) / std

        if inplace:
            self.values = new_values
            return self
        else:
            return SyntheticSeriesSerialized(subjects_in_time=self.subjects_in_time,
                                             values=new_values,
                                             num_time_steps_in_coordination_scale=self.num_time_steps_in_coordination_scale,
                                             time_steps_in_coordination_scale=self.time_steps_in_coordination_scale,
                                             previous_time_same_subject=self.previous_time_same_subject,
                                             previous_time_diff_subject=self.previous_time_diff_subject)

    def plot(self, ax: Any, marker_size: int, hide_x_label: bool = False, hide_y_label: bool = False):
        all_subjects = list(set(self.subjects_in_time))
        all_subjects.sort()

        colors = ["tab:blue", "tab:orange", "tab:green"]
        for i, subject in enumerate(all_subjects):
            ax.scatter(*self.values_per_subject(subject), s=marker_size, label=f"Subject {i + 1}", c=colors[i])
            if not hide_x_label:
                ax.set_xlabel("Time Step")
            if not hide_y_label:
                ax.set_ylabel("Value")


class SerializedModel:

    def __init__(self,
                 num_subjects: int,
                 self_dependent: bool,
                 sd_mean_uc0: float,
                 sd_sd_uc: float,
                 mean_mean_a0: np.ndarray,
                 sd_mean_a0: np.ndarray,
                 sd_sd_aa: np.ndarray,
                 sd_sd_o: np.ndarray,
                 share_mean_a0_across_subjects: bool,
                 share_mean_a0_across_features: bool,
                 share_sd_aa_across_subjects: bool,
                 share_sd_aa_across_features: bool,
                 share_sd_o_across_subjects: bool,
                 share_sd_o_across_features: bool,
                 initial_coordination: Optional[float] = None,
                 mode: Mode = Mode.BLENDING,
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

        self.latent_cpn = SerializedComponent(uuid="latent_component",
                                              num_subjects=num_subjects,
                                              dim_value=1,
                                              self_dependent=self_dependent,
                                              mean_mean_a0=mean_mean_a0,
                                              sd_mean_a0=sd_mean_a0,
                                              sd_sd_aa=sd_sd_aa,
                                              share_mean_a0_across_subjects=share_mean_a0_across_subjects,
                                              share_mean_a0_across_features=share_mean_a0_across_features,
                                              share_sd_aa_across_subjects=share_sd_aa_across_subjects,
                                              share_sd_aa_across_features=share_sd_aa_across_features,
                                              mode=mode,
                                              max_lag=max_lag)

        if num_hidden_layers_g > 0:
            self.g_nn = NeuralNetwork(uuid="g",
                                      num_hidden_layers=num_hidden_layers_g,
                                      dim_hidden_layer=dim_hidden_layer_g,
                                      activation_function_name=activation_function_name_g)
        else:
            # No transformation between latent and observation
            self.g_nn = None

        self.observation_cpn = SerializedObservationComponent(uuid="observation_component",
                                                              num_subjects=num_subjects,
                                                              dim_value=1,
                                                              sd_sd_o=sd_sd_o,
                                                              share_sd_o_across_subjects=share_sd_o_across_subjects,
                                                              share_sd_o_across_features=share_sd_o_across_features)

    @property
    def parameter_names(self) -> List[str]:
        names = self.coordination_cpn.parameter_names
        names.extend(self.latent_cpn.parameter_names)
        names.extend(self.observation_cpn.parameter_names)

        return names

    def fit(self, evidence: SyntheticSeriesSerialized, burn_in: int, num_samples: int, num_chains: int,
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

    def prior_predictive(self, evidence: SyntheticSeriesSerialized, num_samples: int, seed: Optional[int] = None):
        pymc_model = self._define_pymc_model(evidence)
        with pymc_model:
            idata = pm.sample_prior_predictive(samples=num_samples, random_seed=seed)

        return pymc_model, idata

    def posterior_predictive(self, evidence: SyntheticSeriesSerialized, trace: Any, seed: Optional[int] = None):
        pymc_model = self._define_pymc_model(evidence)
        with pymc_model:
            idata = pm.sample_posterior_predictive(trace=trace, random_seed=seed)

        return pymc_model, idata

    def _define_pymc_model(self, evidence: SyntheticSeriesSerialized):
        coords = {"component_feature": np.array([0]),
                  "coordination_time": np.arange(evidence.num_time_steps_in_coordination_scale),
                  "component_time": np.arange(evidence.num_time_steps_in_component_scale)}

        pymc_model = pm.Model(coords=coords)
        with pymc_model:
            coordination = self.coordination_cpn.update_pymc_model(time_dimension="coordination_time")[1]
            latent_component = self.latent_cpn.update_pymc_model(
                coordination=coordination[evidence.time_steps_in_coordination_scale],
                prev_time_same_subject=evidence.previous_time_same_subject,
                prev_time_diff_subject=evidence.previous_time_diff_subject,
                prev_same_subject_mask=evidence.prev_same_subject_mask,
                prev_diff_subject_mask=evidence.prev_diff_subject_mask,
                subjects=evidence.subjects_in_time,
                gender_map={},
                time_dimension="component_time",
                feature_dimension="component_feature",
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
                                                   feature_dimension="component_feature",
                                                   time_dimension="component_time",
                                                   subjects=evidence.subjects_in_time,
                                                   gender_map={},
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

    # Plot parameter trace
    plot_parameter_trace(model, idata)

    # Plot evidence and coordination side by side
    fig, axs = plt.subplots(1, 2)

    evidence.plot(axs[0], marker_size=8)
    axs[0].set_title("Normalized Evidence")

    posterior_samples.plot(axs[1], show_samples=False, line_width=1)
    axs[1].set_title("Coordination")

    return posterior_samples, idata


def prior_predictive_check(model: Any, evidence: Any, num_samples: int = 1000, seed: int = 0):
    _, idata = model.prior_predictive(evidence=evidence, num_samples=num_samples, seed=seed)
    fig = plt.figure()
    plot_prior_predictive(fig.gca(), idata)
    plt.tight_layout()

    return idata


def posterior_predictive_check(model: Any, evidence: Any, trace: Any, seed: int = 0):
    _, idata = model.posterior_predictive(evidence=evidence, trace=trace, seed=seed)
    fig = plt.figure()
    plot_posterior_predictive(fig.gca(), idata)
    plt.tight_layout()

    return idata


def plot_parameter_trace(model: Any, idata: Any):
    sampled_vars = set(idata.posterior.data_vars)
    var_names = sorted(list(set(model.parameter_names).intersection(sampled_vars)))
    az.plot_trace(idata, var_names=var_names)
    plt.tight_layout()


def plot_predictive_samples(ax: Any, samples: Any, idata: Any):
    num_time_steps = samples.sizes["component_time"]
    obs = idata.observed_data["observation_component"].to_numpy()

    prior_samples = samples.sel(component_feature=0).to_numpy().reshape(-1, num_time_steps)
    num_samples = prior_samples.shape[0]
    ax.plot(np.arange(num_time_steps)[:, None].repeat(num_samples, axis=1), prior_samples.T, color="tab:blue",
            alpha=0.3)
    ax.plot(np.arange(num_time_steps), obs[0], color="white", alpha=1, marker="o", markersize=3)
    ax.set_xlabel(f"Time Step")
    ax.set_ylabel(f"Value")


def plot_prior_predictive(axs: Any, idata: Any):
    samples = idata.prior_predictive["observation_component"]
    plot_predictive_samples(axs, samples, idata)


def plot_posterior_predictive(axs: Any, idata: Any):
    samples = idata.posterior_predictive["observation_component"]
    plot_predictive_samples(axs, samples, idata)


def build_convergence_summary(idata: Any) -> pd.DataFrame:
    header = [
        "variable",
        "mean_rhat",
        "std_rhat"
    ]

    rhat = az.rhat(idata)
    data = []
    for var, values in rhat.data_vars.items():
        entry = [
            var,
            values.to_numpy().mean(),
            values.to_numpy().std()
        ]
        data.append(entry)

    return pd.DataFrame(data, columns=header)


if __name__ == "__main__":
    SEED = 0
    BURN_IN = 1000
    NUM_SAMPLES = 1000
    NUM_CHAINS = 2

    random.seed(SEED)
    np.random.seed(SEED)

    # Vertical shift
    evidence_vertical_shift = SyntheticSeriesSerialized.from_function(fn=np.sin,
                                                                      num_subjects=3,
                                                                      num_time_steps=50,
                                                                      time_step_size=np.pi / 12,
                                                                      noise_scale=None,
                                                                      max_lag=5,
                                                                      vertical_offset_per_subject=np.array([0, 1, 2]))
    evidence_vertical_shift_normalized = evidence_vertical_shift.normalize_per_subject(inplace=False)

    # Noise
    evidence_vertical_shift_noise = SyntheticSeriesSerialized.from_function(fn=np.sin,
                                                                            num_subjects=3,
                                                                            num_time_steps=50,
                                                                            time_step_size=np.pi / 12,
                                                                            noise_scale=0.5,
                                                                            max_lag=5,
                                                                            vertical_offset_per_subject=np.array(
                                                                                [0, 1, 2]))
    evidence_vertical_shift_noise_normalized = evidence_vertical_shift_noise.normalize_per_subject(inplace=False)

    # Anti-Symmetry
    evidence_vertical_shift_anti_symmetry = SyntheticSeriesSerialized.from_function(fn=np.sin,
                                                                                    num_subjects=3,
                                                                                    num_time_steps=50,
                                                                                    time_step_size=np.pi / 12,
                                                                                    noise_scale=None,
                                                                                    max_lag=5,
                                                                                    vertical_offset_per_subject=np.array(
                                                                                        [0, 1, 2]),
                                                                                    horizontal_offset_per_subject=np.array(
                                                                                        [0, -4, 0]))
    evidence_vertical_shift_anti_symmetry_normalized = evidence_vertical_shift_anti_symmetry.normalize_per_subject(
        inplace=False)

    # # Random
    # evidence_random = SyntheticSeriesSerialized.from_function(fn=lambda x: np.random.randn(*x.shape),
    #                                                           num_subjects=3,
    #                                                           time_steps=np.linspace(0, 50 * np.pi / 12, 50),
    #                                                           noise_scale=None)
    # evidence_random_normalized = evidence_random.normalize_per_subject(inplace=False)

    # Lag
    evidence_vertical_shift_lag = SyntheticSeriesSerialized.from_function(fn=np.sin,
                                                                          num_subjects=3,
                                                                          num_time_steps=50,
                                                                          time_step_size=np.pi / 12,
                                                                          noise_scale=None,
                                                                          max_lag=5,
                                                                          vertical_offset_per_subject=np.array(
                                                                              [0, 1, 2]),
                                                                          horizontal_offset_per_subject=np.array(
                                                                              [0, -4, -2]))
    evidence_vertical_shift_lag_normalized = evidence_vertical_shift_lag.normalize_per_subject(inplace=False)

    # fig, axs = plt.subplots(2, 4)
    #
    # evidence_vertical_shift.plot(axs[0, 0], marker_size=8)
    # evidence_vertical_shift_normalized.plot(axs[1, 0], marker_size=8)
    #
    # evidence_vertical_shift_noise.plot(axs[0, 1], marker_size=8)
    # evidence_vertical_shift_noise_normalized.plot(axs[1, 1], marker_size=8)
    #
    # evidence_vertical_shift_anti_symmetry.plot(axs[0, 2], marker_size=8)
    # evidence_vertical_shift_anti_symmetry_normalized.plot(axs[1, 2], marker_size=8)
    #
    # evidence_vertical_shift_lag.plot(axs[0, 3], marker_size=8)
    # evidence_vertical_shift_lag_normalized.plot(axs[1, 3], marker_size=8)
    # plt.tight_layout()
    # plt.show()

    evidence = evidence_vertical_shift

    model = SerializedModel(num_subjects=3,
                            self_dependent=True,
                            sd_mean_uc0=5,
                            sd_sd_uc=1,
                            mean_mean_a0=np.zeros((3, 1)),
                            sd_mean_a0=np.ones((3, 1)),
                            sd_sd_aa=np.ones(1),
                            sd_sd_o=np.ones(1),
                            share_mean_a0_across_subjects=False,
                            share_mean_a0_across_features=False,
                            share_sd_aa_across_subjects=True,
                            share_sd_aa_across_features=False,
                            share_sd_o_across_subjects=True,
                            share_sd_o_across_features=False,
                            initial_coordination=None,
                            mode=Mode.BLENDING,
                            max_lag=0,
                            num_hidden_layers_f=0,
                            dim_hidden_layer_f=5,
                            activation_function_name_f="relu")

    # model.latent_cpn.lag_cpn.parameters.lag.value = np.array([-4, -2, 2])

    # prior_predictive_check(model, evidence)
    posterior_samples_vertical_shift_lag_normalized_fit_lag, idata_vertical_shift_lag_normalized_fit_lag = train(model,
                                                                                                                 evidence,
                                                                                                                 burn_in=1000,
                                                                                                                 num_samples=1000,
                                                                                                                 num_chains=NUM_CHAINS,
                                                                                                                 init_method="jitter+adapt_diag")
    # init_method="advi")
    # _ = posterior_predictive_check(model, evidence, idata_vertical_shift_lag_normalized_fit_lag)
    print(build_convergence_summary(idata_vertical_shift_lag_normalized_fit_lag))
    plt.show()
