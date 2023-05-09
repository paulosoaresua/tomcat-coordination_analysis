from typing import Any, Callable, Optional

from functools import partial

import arviz as az
import numpy as np
import pymc as pm
import pytensor.tensor as ptt
from scipy.linalg import expm
from scipy.stats import norm

from coordination.component.lag import Lag
from coordination.component.mixture_component import MixtureComponent
from coordination.component.observation_component import ObservationComponent
from coordination.component.coordination_component import SigmoidGaussianCoordinationComponent
from coordination.model.coordination_model import CoordinationPosteriorSamples
from coordination.component.utils import feed_forward_logp_f, feed_forward_random_f


def mixture_logp(mixture_component: Any,
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
    mixture_component_previous_time = mixture_component#ptt.tensordot(F_inv, mixture_component, axes=(1, 1)).swapaxes(0, 1)

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

    # Discard last time step because it is not previous to any other time step.
    # D = pt.printing.Print("D")(ptt.take_along_axis(D, prev_time_diff_subject[:, None, :], axis=2)[..., 1:])
    D = ptt.take_along_axis(D, prev_time_diff_subject[:, None, :], axis=2)[..., 1:]

    D = ptt.tensordot(F_inv, D, axes=(1, 1)).swapaxes(0, 1)

    # Current values from each subject. We extend S and point such that they match the dimensions of D.
    point_extended = ptt.repeat(mixture_component[..., 1:], repeats=(num_subjects - 1), axis=0)

    if self_dependent.eval():
        # Previous values from every subject
        P = ptt.tensordot(F_inv, mixture_component[..., :-1], axes=(1, 1)).swapaxes(0, 1)  # s x d x t-1

        # Previous values from the same subjects
        # S_extended = pt.printing.Print("S")(ptt.repeat(P, repeats=(num_subjects - 1), axis=0))
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

    logp_extended = pm.logp(pm.Normal.dist(mu=mean, sigma=sd, shape=D.shape), point_extended)
    logp_tmp = logp_extended.reshape((num_subjects, num_subjects - 1, num_features, logp_extended.shape[-1]))
    total_logp += pm.math.logsumexp(logp_tmp + pm.math.log(mixture_weights[:, :, None, None]), axis=1).sum()

    return total_logp


# def mixture_random(initial_mean: np.ndarray,
#                    sigma: np.ndarray,
#                    mixture_weights: np.ndarray,
#                    coordination: np.ndarray,
#                    input_layer_f: np.ndarray,
#                    hidden_layers_f: np.ndarray,
#                    output_layer_f: np.ndarray,
#                    activation_function_number_f: int,
#                    expander_aux_mask_matrix: np.ndarray,
#                    prev_time_diff_subject: Any,
#                    prev_diff_subject_mask: Any,
#                    self_dependent: bool,
#                    num_subjects: int,
#                    dim_value: int,
#                    rng: Optional[np.random.Generator] = None,
#                    size: Optional[Tuple[int]] = None) -> np.ndarray:
#     num_time_steps = coordination.shape[-1]
#
#     noise = rng.normal(loc=0, scale=1, size=size) * sigma[:, :, None]
#
#     # We sample the influencers in each time step using the mixture weights
#     influencers = []
#     for subject in range(num_subjects):
#         probs = np.insert(mixture_weights[subject], subject, 0)
#         influencer = rng.choice(a=np.arange(num_subjects), p=probs, size=num_time_steps)
#         # We will use the influencer to index a matrix with 6 columns. One for each pair influencer -> influenced
#         influencers.append(subject * (num_subjects - 1) + np.minimum(influencer, num_subjects - 2))
#     influencers = np.array(influencers)
#
#     sample = np.zeros_like(noise)
#     prior_sample = rng.normal(loc=initial_mean, scale=sigma, size=(num_subjects, dim_value))
#     sample[..., 0] = prior_sample
#
#     # TODO - Add treatment for lag
#     activation = ActivationFunction.from_numpy_number(activation_function_number_f)
#     for t in np.arange(1, num_time_steps):
#         D = np.einsum("ij,jlk->ilk", expander_aux_mask_matrix, sample[..., t - 1][..., None])  # s * (s-1) x d x 1
#
#         # Previous sample from a different individual
#         D = feed_forward_random_f(input_data=D.reshape((D.shape[0] * D.shape[1], D.shape[2])),
#                                   input_layer_f=input_layer_f,
#                                   hidden_layers_f=hidden_layers_f,
#                                   output_layer_f=output_layer_f,
#                                   activation=activation).reshape(D.shape)[..., 0]  # s * (s-1) x d x 1
#
#         D = D[influencers[..., t]]  # s x d
#
#         # Previous sample from the same individual
#         if self_dependent:
#             S = sample[..., t - 1]
#         else:
#             S = initial_mean
#
#         mean = ((D - S) * coordination[t] + S)
#
#         transition_sample = rng.normal(loc=mean, scale=sigma)
#
#         sample[..., t] = transition_sample
#
#     return sample + noise


class MassSpringDamperLatentMixtureComponent(MixtureComponent):

    def __init__(self, uuid: str,
                 num_subjects: int,
                 spring_constant: float,
                 mass: float,
                 damping_coefficient: float,
                 dt: float,
                 self_dependent: bool,
                 mean_mean_a0: np.ndarray,
                 sd_mean_a0: np.ndarray,
                 sd_sd_aa: np.ndarray,
                 a_mixture_weights: np.ndarray,
                 share_mean_a0_across_subjects: bool,
                 share_mean_a0_across_features: bool,
                 share_sd_aa_across_subjects: bool,
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
                         num_subjects=num_subjects,
                         dim_value=2,  # 2 dimensions: position and velocity
                         self_dependent=self_dependent,
                         mean_mean_a0=mean_mean_a0,
                         sd_mean_a0=sd_mean_a0,
                         sd_sd_aa=sd_sd_aa,
                         a_mixture_weights=a_mixture_weights,
                         share_mean_a0_across_subjects=share_mean_a0_across_subjects,
                         share_mean_a0_across_features=share_mean_a0_across_features,
                         share_sd_aa_across_subjects=share_sd_aa_across_subjects,
                         share_sd_aa_across_features=share_sd_aa_across_features,
                         f=f,
                         mean_weights_f=mean_weights_f,
                         sd_weights_f=sd_weights_f,
                         max_lag=max_lag)

        self.spring_constant = spring_constant
        self.mass = mass
        self.damping_coefficient = damping_coefficient
        self.dt = dt  # size of the time step

        # Systems dynamics matrix
        A = np.array([
            [0, 1],
            [-self.spring_constant / self.mass, -self.damping_coefficient / self.mass]
        ])
        self.F = expm(A * self.dt)  # Fundamental matrix
        self.F_inv = expm(-A * self.dt)  # Fundamental matrix inverse to estimate backward dynamics

    def _draw_from_system_dynamics(self, time_steps_in_coordination_scale: np.ndarray, sampled_coordination: np.ndarray,
                                   sampled_influencers: np.ndarray, mean_a0: np.ndarray,
                                   sd_aa: np.ndarray) -> np.ndarray:

        num_series = sampled_coordination.shape[0]
        num_time_steps = len(time_steps_in_coordination_scale)
        values = np.zeros((num_series, self.num_subjects, self.dim_value, num_time_steps))

        for t in range(num_time_steps):
            if t == 0:
                values[..., 0] = norm(loc=mean_a0, scale=sd_aa).rvs(
                    size=(num_series, self.num_subjects, self.dim_value))
            else:
                C = sampled_coordination[:, time_steps_in_coordination_scale[t]][:, None]

                P = values[..., t - 1]

                if self.f is not None:
                    D = self.f(values[..., t - 1], sampled_influencers[..., t])
                else:
                    D = P

                D = D[:, sampled_influencers[..., t]][0]

                if self.self_dependent:
                    S = P
                else:
                    S = mean_a0

                blended_state = (D - S) * C + S

                mean = np.einsum("ij,klj->kli", self.F, blended_state)

                values[..., t] = norm(loc=mean, scale=sd_aa).rvs()

        return values

    def update_pymc_model(self,
                          coordination: Any,
                          subject_dimension: str,
                          feature_dimension: str,
                          time_dimension: str,
                          num_time_steps: int,
                          observed_values: Optional[Any] = None,
                          mean_a0: Optional[Any] = None,
                          sd_aa: Optional[Any] = None,
                          mixture_weights: Optional[Any] = None,
                          num_layers_f: int = 0,
                          activation_function_name_f: str = "linear",
                          dim_hidden_layer_f: int = 0) -> Any:

        mean_a0, sd_aa, mixture_weights = self._create_random_parameters(mean_a0, sd_aa, mixture_weights)

        input_layer_f, hidden_layers_f, output_layer_f, activation_function_number_f = self._create_random_weights_f(
            num_layers=num_layers_f, dim_hidden_layer=dim_hidden_layer_f,
            activation_function_name=activation_function_name_f)

        # Auxiliary matricx to compute logp in a vectorized manner without having to loop over the individuals.
        # The expander matrix transforms a s x f x t tensor to a s * (s-1) x f x t tensor where the rows contain
        # values of other subjects for each subject in the set.
        expander_aux_mask_matrix = []
        for subject in range(self.num_subjects):
            expander_aux_mask_matrix.append(np.delete(np.eye(self.num_subjects), subject, axis=0))
            aux = np.zeros((self.num_subjects, self.num_subjects - 1))
            aux[subject] = 1

        expander_aux_mask_matrix = np.concatenate(expander_aux_mask_matrix, axis=0)

        # We fit one lag per pair, so the number of lags is C_s_2, where s is the number of subjects.
        if self.lag_cpn is None:
            lag_mask = np.ones((1, num_time_steps), dtype=int)
            prev_time_diff_subject = ptt.arange(num_time_steps)[None, :].repeat(
                self.num_subjects * (self.num_subjects - 1), axis=0) - 1
        else:
            # We fix a lag zero for the first subject and move the others relative to the fixed one.
            # lag = ptt.concatenate([ptt.zeros(1, dtype=int), self.lag_cpn.update_pymc_model(self.num_subjects - 1)])
            lag = self.lag_cpn.update_pymc_model(self.num_subjects)

            # The difference between the influencee and influencer's lags will tell us which time step we need to look
            # at the influencer for each time step in the influencee.
            influencer_lag = ptt.dot(expander_aux_mask_matrix, lag)
            influencee_lag = ptt.repeat(lag, repeats=(self.num_subjects - 1))
            dlag = ptt.cast(influencee_lag - influencer_lag, "int32")

            lag_mask = MixtureComponent._create_lag_mask(num_time_steps, lag)

            prev_time_diff_subject = ptt.arange(num_time_steps, dtype=int)[None, :] + dlag[:, None] - 1
            prev_time_diff_subject *= lag_mask

        logp_params = (mean_a0,
                       sd_aa,
                       mixture_weights,
                       coordination,
                       ptt.constant(input_layer_f),
                       ptt.constant(hidden_layers_f),
                       ptt.constant(output_layer_f),
                       activation_function_number_f,
                       expander_aux_mask_matrix,
                       prev_time_diff_subject,
                       lag_mask,
                       np.array(self.self_dependent),
                       self.F_inv)
        # random_fn = partial(mixture_random, num_subjects=self.num_subjects, dim_value=self.dim_value)
        mixture_component = pm.CustomDist(self.uuid, *logp_params, logp=mixture_logp,
                                          # random=random_fn,
                                          dims=[subject_dimension, feature_dimension, time_dimension],
                                          observed=observed_values)

        # mixture_logp(observed_values, *logp_params)

        return mixture_component, mean_a0, sd_aa, mixture_weights


if __name__ == "__main__":
    coordination = SigmoidGaussianCoordinationComponent(sd_mean_uc0=1,
                                                        sd_sd_uc=1)
    state_space = MassSpringDamperLatentMixtureComponent(uuid="model_state",
                                                         num_subjects=4,
                                                         spring_constant=5,
                                                         mass=10,
                                                         damping_coefficient=0,
                                                         dt=0.2,
                                                         self_dependent=True,
                                                         mean_mean_a0=np.zeros((4, 2)),
                                                         sd_mean_a0=np.ones((4, 2)) * 20,
                                                         sd_sd_aa=np.ones(1),
                                                         a_mixture_weights=np.ones((4, 3)),
                                                         share_mean_a0_across_subjects=False,
                                                         share_sd_aa_across_subjects=True,
                                                         share_mean_a0_across_features=False,
                                                         share_sd_aa_across_features=True)#,
                                                         # f=lambda x, i: np.where((i == 1) | (i == 3), -1, 1)[
                                                         #                    ..., None] * x)
    observations = ObservationComponent(uuid="observation",
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
    state_space.parameters.mean_a0.value = np.array([[1, 0], [5, 0], [10, 0], [15, 0]])
    state_space.parameters.sd_aa.value = np.ones(1) * 0.1
    state_space.parameters.mixture_weights.value = np.array([[1, 0, 0], [1, 0, 0], [0, 0, 1], [0, 0, 1]])
    observations.parameters.sd_o.value = np.ones(1) * 1

    state_samples = state_space.draw_samples(num_series=1, relative_frequency=1, coordination=np.ones((1, 50)) * 1,
                                             seed=0)
    observation_samples = observations.draw_samples(state_samples.values)

    import matplotlib.pyplot as plt

    plt.figure()
    for s in range(state_space.num_subjects):
        plt.scatter(state_samples.time_steps_in_coordination_scale, observation_samples.values[0, s, 0],
                    label=f"Position subject {s + 1}")
    plt.legend()
    plt.show()

    # Inference
    coords = {"feature": ["position", "velocity"],
              "coordination_time": np.arange(50),
              "subject": np.arange(state_space.num_subjects),
              "component_time": np.arange(50)}

    pymc_model = pm.Model(coords=coords)
    with pymc_model:
        # state_space.parameters.clear_values()
        # observations.parameters.clear_values()
        coordination.parameters.sd_uc.value = np.array([0.1])

        coordination_dist = coordination.update_pymc_model(time_dimension="coordination_time")[1]
        state_space_dist = state_space.update_pymc_model(coordination=coordination_dist,
                                                         subject_dimension="subject",
                                                         feature_dimension="feature",
                                                         time_dimension="component_time",
                                                         num_time_steps=50)[0]
        # observed_values=state_samples.values[0])[0]
        observations.update_pymc_model(latent_component=state_space_dist,
                                       subject_dimension="subject",
                                       feature_dimension="feature",
                                       time_dimension="component_time",
                                       observed_values=observation_samples.values[0])

        idata = pm.sample(1000, init="jitter+adapt_diag", tune=1000, chains=2, random_seed=0, cores=2)

        sampled_vars = set(idata.posterior.data_vars)
        var_names = sorted(list(set(coordination.parameter_names + state_space.parameter_names + observations.parameter_names).intersection(sampled_vars)))
        if len(var_names) > 0:
            az.plot_trace(idata, var_names=var_names)
            plt.tight_layout()
            plt.show()

        coordination_posterior = CoordinationPosteriorSamples.from_inference_data(idata)
        coordination_posterior.plot(plt.figure().gca(), show_samples=False)

        plt.figure()
        for s in range(state_space.num_subjects):
            plt.scatter(state_samples.time_steps_in_coordination_scale,
                        idata.posterior["model_state"].sel(subject=s, feature="position").mean(dim=["draw", "chain"]),
                        label=f"Mean Position Subject {s + 1}")

        plt.legend()
        plt.show()