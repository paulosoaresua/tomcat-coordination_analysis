from typing import Any, Callable, Optional

import arviz as az
import numpy as np
import pymc as pm
import pytensor.tensor as ptt
from scipy.linalg import expm
from scipy.stats import norm

from coordination.component.coordination_component import SigmoidGaussianCoordinationComponent
from coordination.component.observation_component import SerializedObservationComponent
from coordination.component.serialized_component import SerializedComponent
from coordination.component.utils import feed_forward_logp_f
from coordination.model.coordination_model import CoordinationPosteriorSamples


def logp(serialized_component: Any,
         initial_mean: Any,
         sigma: Any,
         coordination: Any,
         input_layer_f: Any,
         hidden_layers_f: Any,
         output_layer_f: Any,
         activation_function_number_f: ptt.TensorConstant,
         prev_time_same_subject: ptt.TensorConstant,
         prev_time_diff_subject: ptt.TensorConstant,
         prev_same_subject_mask: Any,
         prev_diff_subject_mask: Any,
         pairs: ptt.TensorConstant,
         self_dependent: ptt.TensorConstant,
         F_inv: ptt.TensorConstant):
    # We reshape to guarantee we don't create dimensions with unknown size in case the first dimension of
    # the serialized component is one
    D = serialized_component[..., prev_time_diff_subject].reshape(serialized_component.shape)  # d x t

    if input_layer_f.shape.prod().eval() != 0:
        # Only transform the input data if a NN was specified. There's a similar check inside the feed_forward_logp_f
        # function, but we duplicate it here because if input_layer_f is empty, we want to use D in its original
        # form instead of concatenated with the pairs matrix.
        D = feed_forward_logp_f(input_data=ptt.concatenate([D, pairs], axis=0),
                                input_layer_f=input_layer_f,
                                hidden_layers_f=hidden_layers_f,
                                output_layer_f=output_layer_f,
                                activation_function_number_f=activation_function_number_f)

    C = coordination[None, :]  # 1 x t
    DM = prev_diff_subject_mask[None, :]  # 1 x t

    if self_dependent.eval():
        # Coordination only affects the mean in time steps where there are previous observations from a different subject.
        # If there's no previous observation from the same subject, we use the initial mean.
        S = serialized_component[..., prev_time_same_subject].reshape(serialized_component.shape)  # d x t

        SM = prev_same_subject_mask[None, :]  # 1 x t
        mean = D * C * DM + (1 - C * DM) * (S * SM + (1 - SM) * initial_mean)
    else:
        # Coordination only affects the mean in time steps where there are previous observations from a different subject.
        mean = D * C * DM + (1 - C * DM) * initial_mean

    # This function can only receive tensors up to 2 dimensions because serialized_component has 2 dimensions.
    # This is a limitation of PyMC 5.0.2. So, we reshape F_inv before passing to this function and here we reshape
    # it back to it's original 3 dimensions.
    F_inv_reshaped = F_inv.reshape((F_inv.shape[0], 2, 2))

    # We transform points using the system dynamics so that samples that follow such dynamics are accepted
    # with higher probability. The batch dimension will be time, that's why we transpose serialized_component.
    serialized_component_transformed = ptt.batched_tensordot(F_inv_reshaped, serialized_component.T,
                                                             axes=[(1,), (1,)]).T

    total_logp = pm.logp(pm.Normal.dist(mu=mean, sigma=sigma, shape=D.shape), serialized_component_transformed).sum()

    return total_logp


class SerializedMassSpringDamperComponent(SerializedComponent):

    def __init__(self, uuid: str,
                 num_springs: int,
                 spring_constant: np.ndarray,  # one per spring
                 mass: np.ndarray,  # one per spring
                 damping_coefficient: np.ndarray,  # one per spring
                 dt: float,
                 self_dependent: bool,
                 mean_mean_a0: np.ndarray,
                 sd_mean_a0: np.ndarray,
                 sd_sd_aa: np.ndarray,
                 share_mean_a0_across_springs: bool,
                 share_mean_a0_across_features: bool,
                 share_sd_aa_across_springs: bool,
                 share_sd_aa_across_features: bool,
                 f: Optional[Callable] = None,
                 mean_weights_f: float = 0,
                 sd_weights_f: float = 1,
                 max_lag: int = 0):
        """
        Generates a time series of latent states formed by position and velocity in a mass-spring-damper system. We do
        not consider external force in this implementation but it can be easily added if necessary.
        """
        super().__init__(uuid=uuid,
                         num_subjects=num_springs,
                         dim_value=2,  # 2 dimensions: position and velocity
                         self_dependent=self_dependent,
                         mean_mean_a0=mean_mean_a0,
                         sd_mean_a0=sd_mean_a0,
                         sd_sd_aa=sd_sd_aa,
                         share_mean_a0_across_subjects=share_mean_a0_across_springs,
                         share_mean_a0_across_features=share_mean_a0_across_features,
                         share_sd_aa_across_subjects=share_sd_aa_across_springs,
                         share_sd_aa_across_features=share_sd_aa_across_features,
                         f=f,
                         mean_weights_f=mean_weights_f,
                         sd_weights_f=sd_weights_f,
                         max_lag=max_lag)

        # We assume the spring_constants are known but this can be changed to have them as latent variables if needed.
        assert spring_constant.ndim == 1 and len(spring_constant) == num_springs

        self.spring_constant = spring_constant
        self.mass = mass
        self.damping_coefficient = damping_coefficient
        self.dt = dt  # size of the time step

        # Systems dynamics matrix
        F = []
        F_inv = []
        for spring in range(num_springs):
            A = np.array([
                [0, 1],
                [-self.spring_constant[spring] / self.mass[spring],
                 -self.damping_coefficient[spring] / self.mass[spring]]
            ])
            F.append(expm(A * self.dt)[None, ...])  # Fundamental matrix
            F_inv.append(expm(-A * self.dt)[None, ...])  # Fundamental matrix inverse to estimate backward dynamics

        self.F = np.concatenate(F, axis=0)
        self.F_inv = np.concatenate(F_inv, axis=0)

    def _draw_from_system_dynamics(self, time_steps_in_coordination_scale: np.ndarray, sampled_coordination: np.ndarray,
                                   subjects_in_time: np.ndarray, prev_time_same_subject: np.ndarray,
                                   prev_time_diff_subject: np.ndarray, mean_a0: np.ndarray,
                                   sd_aa: np.ndarray) -> np.ndarray:

        num_time_steps = len(time_steps_in_coordination_scale)
        values = np.zeros((self.dim_value, num_time_steps))

        for t in range(num_time_steps):
            if self.share_mean_a0_across_subjects:
                subject_idx_mean_a0 = 0
            else:
                subject_idx_mean_a0 = subjects_in_time[t]

            if self.share_sd_aa_across_subjects:
                subject_idx_sd_aa = 0
            else:
                subject_idx_sd_aa = subjects_in_time[t]

            sd = sd_aa[subject_idx_sd_aa]

            if prev_time_same_subject[t] < 0:
                # It is not only when t == 0 because the first utterance of a speaker can be later in the future.
                # t_0 is the initial utterance of one of the subjects only.
                mean = mean_a0[subject_idx_mean_a0]

                values[:, t] = norm(loc=mean, scale=sd).rvs(size=self.dim_value)
            else:
                C = sampled_coordination[time_steps_in_coordination_scale[t]]

                if self.self_dependent:
                    # When there's self dependency, the component either depends on the previous value of another subject,
                    # or the previous value of the same subject.
                    S = values[..., prev_time_same_subject[t]]
                else:
                    # When there's no self dependency, the component either depends on the previous value of another subject,
                    # or it is samples around a fixed mean.
                    S = mean_a0[subject_idx_mean_a0]

                prev_diff_mask = (prev_time_diff_subject[t] != -1).astype(int)
                D = values[..., prev_time_diff_subject[t]]

                if self.f is not None:
                    source_subject = subjects_in_time[prev_time_diff_subject[t]]
                    target_subject = subjects_in_time[t]

                    D = self.f(D, source_subject, target_subject)

                blended_state = (D - S) * C * prev_diff_mask + S

                mean = np.dot(self.F[subjects_in_time[t]], blended_state[:, None]).T

                values[:, t] = norm(loc=mean, scale=sd).rvs()

        return values

    def _get_extra_logp_params(self, subjects_in_time: np.ndarray):
        F_inv_reshaped = self.F_inv[subjects_in_time].reshape((len(subjects_in_time), 4))
        return F_inv_reshaped,

    def _get_logp_fn(self):
        return logp

    def _get_random_fn(self):
        # TODO - implement this for prior predictive checks.
        return None


if __name__ == "__main__":
    coordination = SigmoidGaussianCoordinationComponent(sd_mean_uc0=1,
                                                        sd_sd_uc=1)
    state_space = SerializedMassSpringDamperComponent(uuid="model_state",
                                                      num_subjects=2,
                                                      spring_constant=np.array([0.8, 0.2]),
                                                      mass=1,
                                                      damping_coefficient=np.array([0, 0.5]),
                                                      dt=0.2,
                                                      self_dependent=True,
                                                      mean_mean_a0=np.zeros((2, 2)),
                                                      sd_mean_a0=np.ones((2, 2)) * 10,
                                                      sd_sd_aa=np.ones(1),
                                                      share_mean_a0_across_subjects=False,
                                                      share_sd_aa_across_subjects=True,
                                                      share_mean_a0_across_features=False,
                                                      share_sd_aa_across_features=True,
                                                      max_lag=0)
    # f=lambda x, i: np.where((i == 1) | (i == 3), -1, 1)[
    #                    ..., None] * x)
    observations = SerializedObservationComponent(uuid="observation",
                                                  num_subjects=state_space.num_subjects,
                                                  dim_value=2,
                                                  sd_sd_o=np.ones(1),
                                                  share_sd_o_across_subjects=True,
                                                  share_sd_o_across_features=True)

    # For 3 subjects
    # system.parameters.mean_a0.value = np.array([[1, 0], [5, 0], [10, 0]])
    # system.parameters.sd_aa.value = np.ones((3, 2)) * 0.1
    # system.parameters.mixture_weights.value = np.array([[1, 0], [0, 1], [0, 1]])

    # For 4 subjects
    state_space.parameters.mean_a0.value = np.array([[1, 0], [1, 0]])
    state_space.parameters.sd_aa.value = np.ones(1) * 0.01
    # state_space.lag_cpn.parameters.lag.value = np.array([0, -4, -2, 0])
    observations.parameters.sd_o.value = np.ones(1) * 0.01

    C = 0.99
    T = 100
    state_samples = state_space.draw_samples(num_series=1, time_scale_density=1, coordination=np.ones((1, T)) * C,
                                             seed=0, can_repeat_subject=False)
    observation_samples = observations.draw_samples(state_samples.values, subjects=state_samples.subjects)

    import matplotlib.pyplot as plt

    plt.figure()
    for s in range(state_space.num_subjects):
        tt = [t for t, subject in enumerate(state_samples.subjects[0]) if s == subject]
        plt.scatter(tt, observation_samples.values[0][0, tt],
                    label=f"Position subject {s + 1}", s=15)
        plt.title(f"Coordination = {C}")
    plt.legend()
    plt.show()

    # Inference
    coords = {"feature": ["position", "velocity"],
              "coordination_time": np.arange(T),
              "component_time": np.arange(T)}

    pymc_model = pm.Model(coords=coords)
    with pymc_model:
        state_space.clear_parameter_values()
        observations.parameters.clear_values()
        # coordination.parameters.sd_uc.value = np.array([0.1])

        coordination_dist = coordination.update_pymc_model(time_dimension="coordination_time")[1]
        state_space_dist = state_space.update_pymc_model(coordination=coordination_dist,
                                                         feature_dimension="feature",
                                                         time_dimension="component_time",
                                                         num_layers_f=0,
                                                         activation_function_name_f="linear",
                                                         subjects=state_samples.subjects[0],
                                                         prev_time_same_subject=state_samples.prev_time_same_subject[0],
                                                         prev_time_diff_subject=state_samples.prev_time_diff_subject[0],
                                                         prev_same_subject_mask=np.where(
                                                             state_samples.prev_time_same_subject[0] < 0, 0, 1),
                                                         prev_diff_subject_mask=np.where(
                                                             state_samples.prev_time_diff_subject[0] < 0, 0, 1))[0]
        # observed_values=state_samples.values[0])[0]
        observations.update_pymc_model(latent_component=state_space_dist,
                                       subjects=state_samples.subjects[0],
                                       feature_dimension="feature",
                                       time_dimension="component_time",
                                       observed_values=observation_samples.values[0])

        idata = pm.sample(1000, init="jitter+adapt_diag", tune=1000, chains=2, random_seed=0, cores=2)

        sampled_vars = set(idata.posterior.data_vars)
        var_names = sorted(list(
            set(coordination.parameter_names + state_space.parameter_names + observations.parameter_names).intersection(
                sampled_vars)))
        if len(var_names) > 0:
            az.plot_trace(idata, var_names=var_names)
            plt.tight_layout()
            plt.show()

        coordination_posterior = CoordinationPosteriorSamples.from_inference_data(idata)
        coordination_posterior.plot(plt.figure().gca(), show_samples=False)

        plt.figure()
        for s in range(state_space.num_subjects):
            tt = [t for t, subject in enumerate(state_samples.subjects[0]) if s == subject]
            plt.scatter(tt,
                        idata.posterior["model_state"].sel(feature="position").mean(dim=["draw", "chain"])[tt],
                        label=f"Mean Position Subject {s + 1}", s=15)

        plt.legend()
        plt.show()
