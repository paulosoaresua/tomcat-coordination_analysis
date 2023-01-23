from typing import Any, Optional

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from scipy.stats import norm

from coordination.common.utils import set_random_seed


def mixture_logp_with_self_dependency(mixture_component: Any,
                                      initial_mean: Any,
                                      sigma: Any,
                                      mixture_weights: Any,
                                      coordination: Any,
                                      prev_time: Any,
                                      prev_time_mask: pt.TensorConstant,
                                      subject_mask: pt.TensorConstant):
    C = coordination[None, :]
    P = mixture_component[..., prev_time]
    PTM = prev_time_mask[None, :]
    SM = subject_mask[None, :]

    num_subjects, dim_value, num_time_steps = mixture_component.shape
    num_subjects = num_subjects.eval()

    total_logp = 0
    for s1 in range(num_subjects):
        w = 0
        likelihood_per_subject = 0
        for s2 in range(num_subjects):
            if s1 == s2:
                continue

            means = (P[s2] - P[s1]) * C * PTM + P[s1] * PTM
            logp_s1_from_s2 = pm.logp(
                pm.Normal.dist(mu=means, sigma=sigma[s1][:, None], shape=(dim_value, num_time_steps)),
                mixture_component[s1])

            likelihood_per_subject += mixture_weights[s1, w] * pm.math.exp(logp_s1_from_s2)

            w += 1

        likelihood_per_subject = pm.math.log(likelihood_per_subject) * PTM * SM

        # First time step with observed component
        means = (1 - PTM) * initial_mean[s1][:, None]
        likelihood_per_subject += pm.logp(
            pm.Normal.dist(mu=means, sigma=sigma[s1][:, None], shape=(dim_value, num_time_steps)),
            mixture_component[s1]) * (1 - PTM) * SM

        total_logp += likelihood_per_subject.sum()

    return total_logp


def mixture_logp_without_self_dependency(mixture_component: Any,
                                         initial_mean: Any,
                                         sigma: Any,
                                         mixture_weights: Any,
                                         coordination: Any,
                                         prev_time: Any,
                                         prev_time_mask: pt.TensorConstant,
                                         subject_mask: pt.TensorConstant):
    C = coordination[None, :]
    P = mixture_component[..., prev_time]

    num_subjects, dim_value, num_time_steps = mixture_component.shape
    num_subjects = num_subjects.eval()

    total_logp = 0
    for s1 in range(num_subjects):
        w = 0
        likelihood_per_subject = 0
        for s2 in range(num_subjects):
            if s1 == s2:
                continue

            means = (P[s2] - initial_mean[s1][:, None]) * C * prev_time_mask + initial_mean[s1][:,
                                                                               None] * prev_time_mask
            logp_s1_from_s2 = pm.logp(
                pm.Normal.dist(mu=means, sigma=sigma[s1][:, None], shape=(dim_value, num_time_steps)),
                mixture_component[s1])

            likelihood_per_subject += mixture_weights[s1, w] * pm.math.exp(logp_s1_from_s2)

            w += 1

        likelihood_per_subject = pm.math.log(likelihood_per_subject) * prev_time_mask * subject_mask

        # First time step with observed component
        means = (1 - prev_time_mask) * initial_mean[s1][:, None]
        likelihood_per_subject += pm.logp(
            pm.Normal.dist(mu=means, sigma=sigma[s1][:, None], shape=(dim_value, num_time_steps)),
            mixture_component[s1]) * (1 - prev_time_mask) * subject_mask

        total_logp += likelihood_per_subject.sum()

    return total_logp


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

        assert self.num_subjects == self.parameters.mixture_weights.shape[0]
        assert (self.num_subjects - 1) == self.parameters.mixture_weights.shape[1]
        assert self.num_subjects == self.parameters.mean_a0.shape[0]
        assert self.num_subjects == self.parameters.sd_aa.shape[0]
        assert self.dim_value == self.parameters.mean_a0.shape[1]
        assert self.dim_value == self.parameters.sd_aa.shape[1]

        set_random_seed(seed)

        samples = MixtureComponentSamples()

        samples.values = np.zeros((num_series, self.num_subjects, self.dim_value, num_time_steps))
        samples.mask = np.zeros(shape=(num_series, num_time_steps))
        samples.prev_time = np.full(shape=(num_series, num_time_steps), fill_value=-1)

        for t in range(num_time_steps):
            if (t + 1) == relative_frequency:
                samples.values[..., 0] = norm(loc=self.parameters.mean_a0[None, :],
                                              scale=self.parameters.sd_aa[None, :]).rvs(
                    size=(num_series, self.num_subjects, self.dim_value))
                samples.mask[..., 0] = 1
            elif (t + 1) % relative_frequency == 0:
                samples.mask[..., 0] = 1
                samples.prev_time[..., t] = t - relative_frequency

                for subject1 in range(self.num_subjects):
                    samples_from_mixture = []
                    for subject2 in range(self.num_subjects):
                        if subject1 == subject2:
                            continue

                        D = samples.values[:, subject2, :, samples.prev_time[:, t]]
                        C = coordination[:, t][:, None]

                        if self.self_dependent:
                            S = samples.values[:, subject1, :, samples.prev_time[:, t]]
                        else:
                            S = self.parameters.mean_a0[subject1][None, :]

                        means = (D - S) * C + S
                        samples_from_mixture.append(
                            norm(loc=means[:, 0], scale=self.parameters.sd_aa[subject1][None, :]).rvs(
                                size=(num_series, self.dim_value)))

                    influencer_indices = np.random.choice(a=np.arange(self.num_subjects - 1), size=num_series,
                                                          p=self.parameters.mixture_weights[subject1])
                    samples_from_mixture = np.array(samples_from_mixture)

                    samples.values[:, subject1, :, t] = samples_from_mixture[influencer_indices]

        return samples

    def update_pymc_model(self, coordination: Any, prev_time: pt.TensorConstant, prev_time_mask: pt.TensorConstant,
                          subject_mask: pt.TensorConstant, subject_dimension: str, feature_dimension: str,
                          time_dimension: str) -> Any:

        mean_a0 = pm.HalfNormal(name=f"mean_a0_{self.uuid}", sigma=1, size=(self.num_subjects, self.dim_value),
                                observed=self.parameters.mean_a0)
        sd_aa = pm.HalfNormal(name=f"sd_aa_{self.uuid}", sigma=1, size=(self.num_subjects, self.dim_value),
                              observed=self.parameters.sd_aa)
        mixture_weights = pm.Dirichlet(name=f"mixture_weights_{self.uuid}",
                                       a=pt.ones((self.num_subjects, self.num_subjects - 1)),
                                       observed=self.parameters.mixture_weights)

        if self.self_dependent:
            logp_params = (mean_a0,
                           sd_aa,
                           mixture_weights,
                           coordination,
                           prev_time,
                           prev_time_mask,
                           subject_mask)
            mixture_component = pm.DensityDist(self.uuid, *logp_params, logp=mixture_logp_with_self_dependency,
                                               dims=[subject_dimension, feature_dimension, time_dimension])
        else:
            logp_params = (mean_a0,
                           sd_aa,
                           mixture_weights,
                           coordination,
                           prev_time,
                           prev_time_mask,
                           subject_mask)
            mixture_component = pm.DensityDist(self.uuid, *logp_params, logp=mixture_logp_without_self_dependency,
                                               dims=[subject_dimension, feature_dimension, time_dimension])

        return mixture_component
