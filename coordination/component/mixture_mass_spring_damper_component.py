from typing import Any, Callable, Optional

import numpy as np
import pymc as pm
import pytensor.tensor as ptt
from scipy.linalg import expm
from scipy.stats import norm

from coordination.component.mixture_component import MixtureComponent
from coordination.component.utils import feed_forward_logp_f


def logp(mixture_component: Any,
         initial_mean: Any,
         sigma: Any,
         mixture_weights: Any,
         coordination: Any,
         input_layer_f: Any,
         hidden_layers_f: Any,
         output_layer_f: Any,
         activation_function_number_f: ptt.TensorConstant,
         expander_aux_mask_matrix: ptt.TensorConstant,
         prev_time_diff_subject: Any,
         prev_diff_subject_mask: Any,
         self_dependent: ptt.TensorConstant,
         F_inv: ptt.TensorConstant):
    num_subjects = mixture_component.shape[0]
    num_features = mixture_component.shape[1]

    # Log probability due to the initial time step in the component's scale.
    total_logp = pm.logp(pm.Normal.dist(mu=initial_mean, sigma=sigma, shape=(num_subjects, num_features)),
                         mixture_component[..., 0]).sum()

    # Computes the movement backwards according to the system dynamics, this will make the model learn a latent
    # representation that respects the system dynamics.
    mixture_component_previous_time = mixture_component  # ptt.tensordot(F_inv, mixture_component, axes=(1, 1)).swapaxes(0, 1)

    # Fit a function f(.) to correct anti-symmetry.
    X = feed_forward_logp_f(input_data=mixture_component_previous_time.reshape(
        (mixture_component_previous_time.shape[0] * mixture_component_previous_time.shape[1],
         mixture_component_previous_time.shape[2])),
        input_layer_f=input_layer_f,
        hidden_layers_f=hidden_layers_f,
        output_layer_f=output_layer_f,
        activation_function_number_f=activation_function_number_f).reshape(mixture_component_previous_time.shape)

    # D contains the values from other individuals for each individual
    D = ptt.tensordot(expander_aux_mask_matrix, X, axes=(1, 0))  # s * (s-1) x d x t

    # Get previous values for each time step according to an index matrix. Discard the first time step as
    # there's no previous values in the first time step.
    D = ptt.take_along_axis(D, prev_time_diff_subject[:, None, :], axis=2)[..., 1:]

    if self_dependent.eval():
        # Previous values from every subject
        P = mixture_component[..., :-1]

        # Previous values from the same subjects
        S_extended = ptt.repeat(P, repeats=(num_subjects - 1), axis=0)

    else:
        # Fixed value given by the initial mean for each subject. No self-dependency.
        S_extended = ptt.repeat(initial_mean[:, :, None], repeats=(num_subjects - 1), axis=0)

    # The mask will zero out dependencies on D if we have shifts caused by latent lags. In that case, we cannot infer
    # coordination if the values do not exist on all the subjects because of gaps introduced by the shift. So we can
    # only infer the next value of the latent value from its previous one on the same subject,
    C = coordination[None, None, 1:]  # 1 x 1 x t-1
    mean = (D - S_extended) * C * prev_diff_subject_mask[:, None, 1:] + S_extended

    sd = ptt.repeat(sigma, repeats=(num_subjects - 1), axis=0)[:, :, None]

    # We transform points using the system dynamics so that samples that follow such dynamics are accepted
    # with higher probability.
    mixture_component_transformed = ptt.batched_tensordot(F_inv, mixture_component, axes=[(1,), (1,)])
    # Current values from each subject. We extend S and point such that they match the dimensions of D and S.
    point_extended = ptt.repeat(mixture_component_transformed[..., 1:], repeats=(num_subjects - 1), axis=0)

    logp_extended = pm.logp(pm.Normal.dist(mu=mean, sigma=sd, shape=D.shape), point_extended)
    logp_tmp = logp_extended.reshape((num_subjects, num_subjects - 1, num_features, logp_extended.shape[-1]))
    total_logp += pm.math.logsumexp(logp_tmp + pm.math.log(mixture_weights[:, :, None, None]), axis=1).sum()

    return total_logp


class MixtureMassSpringDamperComponent(MixtureComponent):

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
                 a_mixture_weights: np.ndarray,
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
                         a_mixture_weights=a_mixture_weights,
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
                                   sampled_influencers: np.ndarray, mean_a0: np.ndarray,
                                   sd_aa: np.ndarray) -> np.ndarray:
        num_series = sampled_coordination.shape[0]
        num_time_steps = len(time_steps_in_coordination_scale)

        if self.lag_cpn is not None:
            max_lag = self.lag_cpn.max_lag
            extra_time_steps = 2 * self.lag_cpn.max_lag
        else:
            max_lag = 0
            extra_time_steps = 0

        values = np.zeros((num_series, self.num_subjects, self.dim_value, num_time_steps + extra_time_steps))

        for t in range(max_lag, -1, -1):
            if t == max_lag:
                values[..., t] = norm(loc=mean_a0, scale=sd_aa).rvs(
                    size=(num_series, self.num_subjects, self.dim_value))
            else:
                # Backwards dynamics. This is to guarantee when we apply the lag to each subject, we have mean_a0 as
                # the initial value
                mean = np.einsum("ijk,lij->lik", self.F_inv, values[..., t + 1])
                values[..., t] = norm(loc=mean, scale=sd_aa).rvs()

        for t in range(max_lag + 1, num_time_steps + extra_time_steps):
            if t >= num_time_steps + extra_time_steps - max_lag:
                # There's no coordination for these time steps. They are extra and we use them to fill in the gaps
                # when we roll the samples by each subject's lag in the end.
                mean = np.einsum("ijk,lij->lik", self.F, values[..., t - 1])
            else:
                C = sampled_coordination[:, time_steps_in_coordination_scale[t - max_lag]][:, None]

                P = values[..., t - 1]

                if self.f is not None:
                    D = self.f(values[..., t - 1], sampled_influencers[..., t - max_lag])
                else:
                    D = P

                D = D[:, sampled_influencers[..., t - max_lag]][0]

                if self.self_dependent:
                    S = P
                else:
                    S = mean_a0

                blended_state = (D - S) * C + S

                mean = np.einsum("ijk,lij->lik", self.F, blended_state)

            # Add some noise
            values[..., t] = norm(loc=mean, scale=sd_aa).rvs()

        # Apply lags if any
        if self.lag_cpn is not None:
            truncated_values_per_subject = []
            for subject in range(self.num_subjects):
                lag = -self.lag_cpn.parameters.lag.value[subject]  # lag contains the correction, so we use -lag
                truncated_values_per_subject.append(
                    values[:, subject, :, (max_lag + lag):(num_time_steps + max_lag + lag)][:, None, :, :])

            values = np.concatenate(truncated_values_per_subject, axis=1)

        return values

    def _get_extra_logp_params(self):
        return self.F_inv,

    def _get_logp_fn(self):
        return logp

    def _get_random_fn(self):
        # TODO - implement this for prior predictive checks.
        return None
