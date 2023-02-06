from typing import Any, Optional, Tuple

from functools import partial
import numpy as np
import pymc as pm
import pytensor.tensor as ptt
from scipy.stats import norm

from coordination.common.utils import set_random_seed
from coordination.model.parametrization import Parameter, HalfNormalParameterPrior, DirichletParameterPrior


def mixture_logp_with_self_dependency(mixture_component: Any,
                                      initial_mean: Any,
                                      sigma: Any,
                                      mixture_weights: np.ndarray,
                                      coordination: Any,
                                      expander_aux_mask_matrix: ptt.TensorConstant,
                                      aggregation_aux_mask_matrix: ptt.TensorVariable):
    C = coordination[None, None, 1:]  # 1 x 1 x t
    P = mixture_component[..., :-1]  # s x d x t

    # Log probability due to the initial time step in the component's scale.
    total_logp = pm.logp(pm.Normal.dist(mu=initial_mean, sigma=sigma, shape=mixture_component.shape[:-1]),
                         mixture_component[..., 0]).sum()

    # D contains the previous values from other individuals for each individual
    D = ptt.tensordot(expander_aux_mask_matrix, P, axes=(1, 0))  # s * (s-1) x d x t
    P_extended = ptt.repeat(P, repeats=(mixture_component.shape[0] - 1), axis=0)
    point_extended = ptt.repeat(mixture_component[..., 1:], repeats=(mixture_component.shape[0] - 1), axis=0)

    mean = (D - P_extended) * C + P_extended

    pdf = pm.math.exp(
        pm.logp(pm.Normal.dist(mu=mean,
                               sigma=ptt.repeat(sigma, repeats=(mixture_component.shape[0] - 1), axis=0)[:, :, None],
                               shape=D.shape), point_extended))
    total_logp += (pm.math.log(ptt.tensordot(aggregation_aux_mask_matrix, pdf, axes=(1, 0)))).sum()

    return total_logp


def mixture_logp_without_self_dependency(mixture_component: Any,
                                         initial_mean: Any,
                                         sigma: Any,
                                         mixture_weights: np.ndarray,
                                         coordination: Any,
                                         expander_aux_mask_matrix: ptt.TensorConstant,
                                         aggregation_aux_mask_matrix: ptt.TensorVariable):
    C = coordination[None, None, 1:]  # 1 x 1 x t
    P = mixture_component[..., :-1]  # s x d x t

    # Log probability due to the initial time step in the component's scale.
    total_logp = pm.logp(pm.Normal.dist(mu=initial_mean, sigma=sigma, shape=mixture_component.shape[:-1]),
                         mixture_component[..., 0]).sum()

    # D contains the previous values from other individuals for each individual
    D = ptt.tensordot(expander_aux_mask_matrix, P, axes=(1, 0))  # s * (s-1) x d x t

    # We use a fixed mean instead of the previous value from the same individual
    P_extended = ptt.repeat(initial_mean[:, :, None], repeats=(mixture_component.shape[0] - 1), axis=0)
    point_extended = ptt.repeat(mixture_component[..., 1:], repeats=(mixture_component.shape[0] - 1), axis=0)

    mean = (D - P_extended) * C + P_extended

    pdf = pm.math.exp(
        pm.logp(pm.Normal.dist(mu=mean,
                               sigma=ptt.repeat(sigma, repeats=(mixture_component.shape[0] - 1), axis=0)[:, :, None],
                               shape=D.shape), point_extended))
    total_logp += (pm.math.log(ptt.tensordot(aggregation_aux_mask_matrix, pdf, axes=(1, 0)))).sum()

    return total_logp


def mixture_random_with_self_dependency(initial_mean: np.ndarray,
                                        sigma: np.ndarray,
                                        mixture_weights: np.ndarray,
                                        coordination: np.ndarray,
                                        expander_aux_mask_matrix: np.ndarray,
                                        aggregation_aux_mask_matrix: np.ndarray,
                                        num_subjects: int,
                                        dim_value: int,
                                        rng: Optional[np.random.Generator] = None,
                                        size: Optional[Tuple[int]] = None) -> np.ndarray:
    num_time_steps = coordination.shape[-1]

    noise = rng.normal(loc=0, scale=1, size=size) * sigma[:, :, None]

    # We sample the influencers in each time step using the mixture weights
    influencers = []
    for subject in range(num_subjects):
        probs = np.insert(mixture_weights[subject], subject, 0)
        influencers.append(rng.choice(a=np.arange(num_subjects), p=probs, size=num_time_steps))
    influencers = np.array(influencers)

    sample = np.zeros_like(noise)
    prior_sample = rng.normal(loc=initial_mean, scale=sigma, size=(num_subjects, dim_value))
    sample[..., 0] = prior_sample
    for t in np.arange(1, num_time_steps):
        # Previous sample from a different individual
        D = sample[..., t - 1][influencers[..., t]]
        # Previous sample from the same individual
        S = sample[..., t - 1]

        mean = ((D - S) * coordination[t] + S)

        transition_sample = rng.normal(loc=mean, scale=sigma)

        sample[..., t] = transition_sample

    return sample + noise


def mixture_random_without_self_dependency(initial_mean: np.ndarray,
                                           sigma: np.ndarray,
                                           mixture_weights: np.ndarray,
                                           coordination: np.ndarray,
                                           expander_aux_mask_matrix: np.ndarray,
                                           aggregation_aux_mask_matrix: np.ndarray,
                                           num_subjects: int,
                                           dim_value: int,
                                           rng: Optional[np.random.Generator] = None,
                                           size: Optional[Tuple[int]] = None) -> np.ndarray:
    num_time_steps = coordination.shape[-1]

    noise = rng.normal(loc=0, scale=1, size=size) * sigma[:, :, None]

    # We sample the influencers in each time step using the mixture weights
    influencers = []
    for subject in range(num_subjects):
        probs = np.insert(mixture_weights[subject], subject, 0)
        influencers.append(rng.choice(a=np.arange(num_subjects), p=probs, size=num_time_steps))
    influencers = np.array(influencers)

    sample = np.zeros_like(noise)
    prior_sample = rng.normal(loc=initial_mean, scale=sigma, size=(num_subjects, dim_value))
    sample[..., 0] = prior_sample

    # No self-dependency. The transition distribution is a blending between the previous value from another individual,
    # and a fixed mean.
    S = initial_mean
    for t in np.arange(1, num_time_steps):
        # Previous sample from a different individual
        D = sample[..., t - 1][influencers[..., t]]

        mean = ((D - S) * coordination[t] + S)

        transition_sample = rng.normal(loc=mean, scale=sigma)

        sample[..., t] = transition_sample

    return sample + noise


class MixtureComponentParameters:

    def __init__(self, sd_mean_a0: np.ndarray, sd_sd_aa: np.ndarray, a_mixture_weights: np.ndarray):
        self.mean_a0 = Parameter(HalfNormalParameterPrior(sd_mean_a0))
        self.sd_aa = Parameter(HalfNormalParameterPrior(sd_sd_aa))
        self.mixture_weights = Parameter(DirichletParameterPrior(a_mixture_weights))

    def clear_values(self):
        self.mean_a0.value = None
        self.sd_aa.value = None
        self.mixture_weights.value = None


class MixtureComponentSamples:

    def __init__(self):
        self.values = np.array([])

        # For each time step in the component's scale, it contains the time step in the coordination scale
        self.time_steps_in_coordination_scale = np.array([])

    @property
    def num_time_steps(self):
        return self.values.shape[-1]


class MixtureComponent:

    def __init__(self, uuid: str, num_subjects: int, dim_value: int, self_dependent: bool, sd_mean_a0: np.ndarray,
                 sd_sd_aa: np.ndarray, a_mixture_weights: np.ndarray):

        assert (num_subjects, dim_value) == sd_mean_a0.shape
        assert (num_subjects, dim_value) == sd_sd_aa.shape
        assert (num_subjects, num_subjects - 1) == a_mixture_weights.shape

        self.uuid = uuid
        self.num_subjects = num_subjects
        self.dim_value = dim_value
        self.self_dependent = self_dependent

        self.parameters = MixtureComponentParameters(sd_mean_a0, sd_sd_aa, a_mixture_weights)

    def draw_samples(self, num_series: int, seed: Optional[int], relative_frequency: float,
                     coordination: np.ndarray) -> MixtureComponentSamples:

        assert self.num_subjects == self.parameters.mixture_weights.value.shape[0]
        assert (self.num_subjects - 1) == self.parameters.mixture_weights.value.shape[1]
        assert self.num_subjects == self.parameters.mean_a0.value.shape[0]
        assert self.num_subjects == self.parameters.sd_aa.value.shape[0]
        assert self.dim_value == self.parameters.mean_a0.value.shape[1]
        assert self.dim_value == self.parameters.sd_aa.value.shape[1]

        set_random_seed(seed)

        samples = MixtureComponentSamples()

        # Number of time steps in the component's scale
        num_time_steps_in_cpn_scale = int(coordination.shape[-1] / relative_frequency)
        samples.values = np.zeros((num_series, self.num_subjects, self.dim_value, num_time_steps_in_cpn_scale))
        samples.time_steps_in_coordination_scale = np.full((num_series, num_time_steps_in_cpn_scale), fill_value=-1,
                                                           dtype=int)

        # Sample influencers in each time step
        influencers = []
        for subject in range(self.num_subjects):
            probs = np.insert(self.parameters.mixture_weights.value[subject], subject, 0)
            influencers.append(
                np.random.choice(a=np.arange(self.num_subjects), p=probs,
                                 size=(num_series, num_time_steps_in_cpn_scale)))
        influencers = np.array(influencers).swapaxes(0, 1)

        for t in range(num_time_steps_in_cpn_scale):
            if t == 0:
                samples.values[..., 0] = norm(loc=self.parameters.mean_a0.value[None, :],
                                              scale=self.parameters.sd_aa.value[None, :]).rvs(
                    size=(num_series, self.num_subjects, self.dim_value))
            else:
                time_in_coord_scale = relative_frequency * t

                C = coordination[:, time_in_coord_scale][:, None]
                P = samples.values[..., t - 1]
                D = P[:, influencers[..., t]][0]

                if self.self_dependent:
                    mean = (D - P) * C + P
                else:
                    mean = (D - self.parameters.mean_a0.value[None, :]) * C + self.parameters.mean_a0.value[None, :]

                samples.values[..., t] = norm(loc=mean, scale=self.parameters.sd_aa.value[None, :]).rvs()
                samples.time_steps_in_coordination_scale[..., t] = time_in_coord_scale

        return samples

    def update_pymc_model(self, coordination: Any, subject_dimension: str, feature_dimension: str, time_dimension: str,
                          observation: Optional[Any] = None) -> Any:

        mean_a0 = pm.HalfNormal(name=f"mean_a0_{self.uuid}", sigma=self.parameters.mean_a0.prior.sd,
                                size=(self.num_subjects, self.dim_value),
                                observed=self.parameters.mean_a0.value)
        sd_aa = pm.HalfNormal(name=f"sd_aa_{self.uuid}", sigma=self.parameters.sd_aa.prior.sd,
                              size=(self.num_subjects, self.dim_value),
                              observed=self.parameters.sd_aa.value)
        mixture_weights = pm.Dirichlet(name=f"mixture_weights_{self.uuid}",
                                       a=self.parameters.mixture_weights.prior.a,
                                       observed=self.parameters.mixture_weights.value)

        # Auxiliary matrices to compute logp in a vectorized manner without having to loop over the individuals.
        expander_aux_mask_matrix = []
        aggregator_aux_mask_matrix = []
        for subject in range(self.num_subjects):
            expander_aux_mask_matrix.append(np.delete(np.eye(self.num_subjects), subject, axis=0))
            aux = np.zeros((self.num_subjects, self.num_subjects - 1))
            aux[subject] = 1
            aux = aux * mixture_weights[subject][None, :]
            aggregator_aux_mask_matrix.append(aux)

        expander_aux_mask_matrix = np.concatenate(expander_aux_mask_matrix, axis=0)
        aggregator_aux_mask_matrix = ptt.concatenate(aggregator_aux_mask_matrix, axis=1)

        if self.self_dependent:
            logp_params = (mean_a0,
                           sd_aa,
                           mixture_weights,
                           coordination,
                           ptt.constant(expander_aux_mask_matrix),
                           aggregator_aux_mask_matrix)
            random_fn = partial(mixture_random_with_self_dependency,
                                num_subjects=self.num_subjects, dim_value=self.dim_value)
            mixture_component = pm.CustomDist(self.uuid, *logp_params, logp=mixture_logp_with_self_dependency,
                                              random=random_fn,
                                              dims=[subject_dimension, feature_dimension, time_dimension],
                                              observed=observation)
        else:
            logp_params = (mean_a0,
                           sd_aa,
                           mixture_weights,
                           coordination,
                           ptt.constant(expander_aux_mask_matrix),
                           aggregator_aux_mask_matrix)

            random_fn = partial(mixture_random_without_self_dependency,
                                num_subjects=self.num_subjects, dim_value=self.dim_value)
            mixture_component = pm.CustomDist(self.uuid, *logp_params, logp=mixture_logp_without_self_dependency,
                                              random=random_fn,
                                              dims=[subject_dimension, feature_dimension, time_dimension],
                                              observed=observation)

        return mixture_component
