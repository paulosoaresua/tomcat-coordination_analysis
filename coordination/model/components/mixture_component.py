from typing import Any, Optional

import numpy as np
import pymc as pm
import pytensor as pt
import pytensor.tensor as ptt
from scipy.stats import norm

from coordination.common.utils import set_random_seed


def mixture_logp_with_self_dependency(mixture_component: Any,
                                      initial_mean: Any,
                                      sigma: Any,
                                      mixture_weights: Any,
                                      expanded_mixture_weights: Any,
                                      coordination: Any,
                                      prev_time: Any,
                                      prev_time_mask: ptt.TensorConstant,
                                      subject_mask: ptt.TensorConstant,
                                      num_subjects: ptt.TensorConstant,
                                      dim_value: ptt.TensorConstant):
    C = coordination[None, :]  # 1 x t
    P = mixture_component[..., prev_time]  # s x d x t
    PTM = prev_time_mask[None, :]  # 1 x t
    SM = subject_mask[None, :]  # 1 x t

    num_time_steps = coordination.shape[-1]

    # num_subjects, dim_value, num_time_steps = mixture_component.shape
    # num_subjects = num_subjects.eval()

    # Log probability due to the initial step, when there's no previous observation (PTM == 0).
    total_logp = (pm.logp(
        pm.Normal.dist(mu=(1 - PTM) * initial_mean, sigma=sigma, shape=(num_subjects, dim_value, num_time_steps)),
        mixture_component) * PTM * SM).sum()

    # Log probability due to the influence of distinct subjects according to the mixture weight
    for s1 in ptt.arange(num_subjects).eval():
        w = 0
        likelihood_per_subject = 0

        for s2 in ptt.arange(num_subjects).eval():
            if s1 == s2:
                continue

            means = (P[s2] - P[s1]) * C + P[s1]
            logp_s1_from_s2 = pm.logp(
                pm.Normal.dist(mu=means, sigma=sigma, shape=(dim_value, num_time_steps)),
                mixture_component[s1])

            likelihood_per_subject += mixture_weights[0, w] * pm.math.exp(logp_s1_from_s2)
            w += 1

        # Multiply by PTM to ignore entries already accounted for in the initial time step (no previous observation)
        # Multiply by SM to ignore entries in time step without observations
        likelihood_per_subject = pm.math.log(likelihood_per_subject) * PTM * SM

        total_logp += likelihood_per_subject.sum()

    return total_logp


def mixture_logp_without_self_dependency(mixture_component: Any,
                                         initial_mean: Any,
                                         sigma: Any,
                                         mixture_weights: Any,
                                         expanded_mixture_weights: Any,
                                         coordination: Any,
                                         prev_time: Any,
                                         prev_time_mask: ptt.TensorConstant,
                                         subject_mask: ptt.TensorConstant,
                                         num_subjects: ptt.TensorConstant,
                                         dim_value: ptt.TensorConstant):
    C = coordination[None, :]  # 1 x t
    P = mixture_component[..., prev_time]  # s x d x t
    PTM = prev_time_mask[None, :]  # 1 x t
    SM = subject_mask[None, :]  # 1 x t

    num_time_steps = coordination.shape[-1]

    # num_subjects, dim_value, num_time_steps = mixture_component.shape
    # num_subjects = num_subjects.eval()

    # Log probability due to the initial step, when there's no previous observation (PTM == 0).
    total_logp = (pm.logp(
        pm.Normal.dist(mu=(1 - PTM) * initial_mean, sigma=sigma, shape=(num_subjects, dim_value, num_time_steps)),
        mixture_component) * PTM * SM).sum()

    # Log probability due to the influence of distinct subjects according to the mixture weight
    for s1 in ptt.arange(num_subjects).eval():
        w = 0
        likelihood_per_subject = 0

        for s2 in ptt.arange(num_subjects).eval():
            if s1 == s2:
                continue

            means = (P[s2] - initial_mean) * C + initial_mean
            logp_s1_from_s2 = pm.logp(
                pm.Normal.dist(mu=means, sigma=sigma, shape=(dim_value, num_time_steps)),
                mixture_component[s1])

            likelihood_per_subject += mixture_weights[s1, w] * pm.math.exp(logp_s1_from_s2)
            w += 1

        # Multiply by PTM to ignore entries already accounted for in the initial time step (no previous observation)
        # Multiply by SM to ignore entries in time step without observations
        likelihood_per_subject = pm.math.log(likelihood_per_subject) * PTM * SM

        total_logp += likelihood_per_subject.sum()

    return total_logp


def random(initial_mean: Any,
           sigma: Any,
           mixture_weights: Any,
           expanded_mixture_weights: Any,
           coordination: Any,
           prev_time: Any,
           prev_time_mask: ptt.TensorConstant,
           subject_mask: ptt.TensorConstant,
           num_subjects: ptt.TensorConstant,
           dim_value: ptt.TensorConstant,
           *args,
           **kwargs):
    num_time_steps = coordination.shape[-1]
    # noise = pm.draw(pm.Normal.dist(mu=0, sigma=1, shape=(num_subjects, dim_value, num_time_steps)), 1) * sigma
    noise = ptt.random.normal(loc=0, scale=1, size=(num_subjects, dim_value, num_time_steps)) * sigma

    # Prior
    # prior_dist = pm.Normal.dist(mu=initial_mean, sigma=sigma, shape=(num_subjects, dim_value))
    # sample = ptt.ones((num_time_steps, num_subjects, dim_value)) * pm.draw(prior_dist, 1)[None, :] * (
    #         1 - prev_time_mask[:, None, None]) * subject_mask[:, None, None]
    prior_sample = ptt.random.normal(loc=initial_mean, scale=sigma, size=(1, num_subjects, dim_value))
    sample = ptt.ones((num_time_steps, num_subjects, dim_value)) * prior_sample * (
                1 - prev_time_mask[:, None, None]) * subject_mask[:, None, None]

    # We sample the influencers in each time step using the mixture weights
    # influencers_dist = pm.Categorical.dist(p=expanded_mixture_weights, shape=num_subjects)
    influencers = ptt.random.categorical(p=expanded_mixture_weights, size=(num_time_steps, num_subjects))
    # influencers = ptt.constant(pm.draw(influencers_dist, num_time_steps.eval().astype(np.int32)))  # t x s

    # CumSum adapted for coupled chain
    def sample_from_mixture(sample, mask, influencers, coordination, prev_val):
        # For time steps out of the component's scale, we just repeat the previous sampled values from all
        # individuals
        return sample + (prev_val[influencers] * coordination + prev_val * (1 - coordination)) * mask + prev_val * (
                1 - mask)

    results, updates = pt.scan(fn=sample_from_mixture,
                               outputs_info=ptt.zeros((num_subjects, dim_value)),
                               sequences=[sample, subject_mask, influencers, coordination])

    return results.dimshuffle(1, 2, 0) + noise


class MixtureComponentParameters:

    def __init__(self):
        self.mean_a0 = None
        self.sd_aa = None
        self.mixture_weights = None

    def reset(self):
        self.mean_a0 = None
        self.sd_aa = None
        self.mixture_weights = None


class MixtureComponentSamples:

    def __init__(self):
        self.values = np.array([])

        # 1 for time steps in which the component exists, 0 otherwise.
        self.mask = np.array([])

        # Time indices indicating the previous occurrence of the component produced by the subjects. Mixture compo-
        # nents assumes observations are continuous and measurable for all subjects periodically in a specific time-
        # scale. The variable below, maps from the coordination scale to the component's scale.
        self.prev_time = np.array([])


class MixtureComponent:

    def __init__(self, uuid: str, num_subjects: int, dim_value: int, self_dependent: bool):
        self.uuid = uuid
        self.num_subjects = num_subjects
        self.dim_value = dim_value
        self.self_dependent = self_dependent

        self.parameters = MixtureComponentParameters()

    def draw_samples(self, num_series: int, num_time_steps: int,
                     seed: Optional[int], relative_frequency: float,
                     coordination: np.ndarray) -> MixtureComponentSamples:

        # assert self.num_subjects == self.parameters.mixture_weights.shape[0]
        # assert (self.num_subjects - 1) == self.parameters.mixture_weights.shape[1]
        # assert self.num_subjects == self.parameters.mean_a0.shape[0]
        # assert self.num_subjects == self.parameters.sd_aa.shape[0]
        # assert self.dim_value == self.parameters.mean_a0.shape[1]
        # assert self.dim_value == self.parameters.sd_aa.shape[1]

        set_random_seed(seed)

        samples = MixtureComponentSamples()

        samples.values = np.zeros((num_series, self.num_subjects, self.dim_value, num_time_steps))
        samples.mask = np.zeros(shape=(num_series, num_time_steps))
        samples.prev_time = np.full(shape=(num_series, num_time_steps), fill_value=-1)

        for t in range(num_time_steps):
            if (t + 1) == relative_frequency:
                samples.values[..., 0] = norm(loc=self.parameters.mean_a0,
                                              scale=self.parameters.sd_aa).rvs(
                    size=(num_series, self.num_subjects, self.dim_value))
                samples.mask[..., 0] = 1
            elif (t + 1) % relative_frequency == 0:
                samples.mask[..., t] = 1
                samples.prev_time[..., t] = t - relative_frequency

                C = coordination[:, t][:, None]

                for subject1 in range(self.num_subjects):
                    if self.self_dependent:
                        S = samples.values[:, subject1, :, samples.prev_time[:, t]][:, 0]  # n x d
                    else:
                        S = self.parameters.mean_a0

                    samples_from_mixture = []
                    for subject2 in range(self.num_subjects):
                        if subject1 == subject2:
                            continue

                        D = samples.values[:, subject2, :, samples.prev_time[:, t]][:, 0]  # n x d

                        means = (D - S) * C + S
                        samples_from_mixture.append(
                            norm(loc=means, scale=self.parameters.sd_aa).rvs(size=(num_series, self.dim_value)))

                    # Choose samples from specific influencers according to the mixture weights
                    influencer_indices = np.random.choice(a=np.arange(self.num_subjects - 1), size=num_series,
                                                          p=self.parameters.mixture_weights[0])
                    samples_from_mixture = np.array(samples_from_mixture)

                    samples.values[:, subject1, :, t] = samples_from_mixture[influencer_indices]

        return samples

    def update_pymc_model(self, coordination: Any, prev_time: ptt.TensorConstant, prev_time_mask: ptt.TensorConstant,
                          subject_mask: ptt.TensorConstant, subject_dimension: str, feature_dimension: str,
                          time_dimension: str) -> Any:

        # mean_a0 = pm.HalfNormal(name=f"mean_a0_{self.uuid}", sigma=1, size=(self.num_subjects, self.dim_value),
        #                         observed=self.parameters.mean_a0)
        # sd_aa = pm.HalfNormal(name=f"sd_aa_{self.uuid}", sigma=1, size=(self.num_subjects, self.dim_value),
        #                       observed=self.parameters.sd_aa)
        # mixture_weights = pm.Dirichlet(name=f"mixture_weights_{self.uuid}",
        #                                a=ptt.ones((self.num_subjects, self.num_subjects - 1)),
        #                                observed=self.parameters.mixture_weights)
        mean_a0 = pm.HalfNormal(name=f"mean_a0_{self.uuid}", sigma=1, size=1,
                                observed=self.parameters.mean_a0)
        sd_aa = pm.HalfNormal(name=f"sd_aa_{self.uuid}", sigma=1, size=1,
                              observed=self.parameters.sd_aa)
        mixture_weights = pm.Dirichlet(name=f"mixture_weights_{self.uuid}",
                                       a=ptt.ones(self.num_subjects - 1),
                                       size=1,
                                       observed=self.parameters.mixture_weights)

        weight_expander_matrix = []
        for i in range(self.num_subjects):
            weight_expander_matrix.append(np.delete(np.eye(self.num_subjects), i, axis=1))

        weight_expander_matrix = np.concatenate(weight_expander_matrix, axis=0)
        expanded_mixture_weights = pm.Deterministic(f"expanded_mixture_weights_{self.uuid}",
                                                    (ptt.constant(weight_expander_matrix) @ ptt.transpose(
                                                        mixture_weights)).reshape(
                                                        (self.num_subjects,
                                                         self.num_subjects)))

        if self.self_dependent:
            logp_params = (mean_a0,
                           sd_aa,
                           mixture_weights,
                           expanded_mixture_weights,
                           coordination,
                           prev_time,
                           prev_time_mask,
                           subject_mask,
                           ptt.constant(self.num_subjects),
                           ptt.constant(self.dim_value))
            mixture_component = pm.CustomDist(self.uuid, *logp_params, logp=mixture_logp_with_self_dependency,
                                               random=random,
                                               dims=[subject_dimension, feature_dimension, time_dimension])
        else:
            logp_params = (mean_a0,
                           sd_aa,
                           mixture_weights,
                           expanded_mixture_weights,
                           coordination,
                           prev_time,
                           prev_time_mask,
                           subject_mask,
                           ptt.constant(self.num_subjects),
                           ptt.constant(self.dim_value))

            mixture_component = pm.CustomDist(self.uuid, *logp_params, logp=mixture_logp_without_self_dependency,
                                               random=random,
                                               dims=[subject_dimension, feature_dimension, time_dimension])

        return mixture_component
