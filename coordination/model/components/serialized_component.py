from typing import Any, Optional

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from scipy.stats import norm

from coordination.common.utils import set_random_seed


def serialized_logp_with_self_dependency(serialized_component: Any,
                                         initial_mean: Any,
                                         sigma: Any,
                                         coordination: Any,
                                         prev_time_same_subject: pt.TensorConstant,
                                         prev_time_diff_subject: pt.TensorConstant,
                                         prev_same_subject_mask: pt.TensorConstant,
                                         prev_diff_subject_mask: pt.TensorConstant,
                                         subject_mask: pt.TensorConstant):
    C = coordination[None, :]
    S = serialized_component[..., prev_time_same_subject]
    D = serialized_component[..., prev_time_diff_subject]

    means = (D - S * prev_same_subject_mask) * prev_diff_subject_mask * C + S * prev_same_subject_mask + (
            1 - prev_same_subject_mask) * initial_mean

    return (pm.logp(pm.Normal.dist(mu=means, sigma=sigma, shape=serialized_component.shape),
                    serialized_component) * subject_mask).sum()


def serialized_logp_without_self_dependency(serialized_component: Any,
                                            initial_mean: Any,
                                            sigma: Any,
                                            coordination: Any,
                                            prev_time_diff_subject: pt.TensorConstant,
                                            prev_diff_subject_mask: pt.TensorConstant,
                                            subject_mask: pt.TensorConstant):
    C = coordination[None, :]
    D = serialized_component[..., prev_time_diff_subject]

    means = (D - initial_mean) * prev_diff_subject_mask * C + initial_mean

    return (pm.logp(pm.Normal.dist(mu=means, sigma=sigma, shape=serialized_component.shape),
                    serialized_component) * subject_mask).sum()


class SerializedComponentParameters:

    def __init__(self):
        self.mean_a0 = None
        self.sd_aa = None

    def reset(self):
        self.mean_a0 = None
        self.sd_aa = None


class SerializedComponentSamples:

    def __init__(self):
        self.values = np.array([])

        # 1 for time steps in which the component exists, 0 otherwise.
        self.mask = np.array([])

        # Number indicating which subject is associated to the component at a time (e.g. the current speaker for
        # a vocalics component).
        self.subjects = np.array([])

        # Time indices indicating the previous occurrence of the component produced by the same subject and the most
        # recent different one. For instance, the last time when the current speaker talked and a different speaker.
        self.prev_time_same_subject = np.array([])
        self.prev_time_diff_subject = np.array([])


class SerializedComponent:

    def __init__(self, uuid: str, num_subjects: int, dim_value: int, self_dependent: bool):
        self.uuid = uuid
        self.num_subjects = num_subjects
        self.dim_value = dim_value
        self.self_dependent = self_dependent

        self.parameters = SerializedComponentParameters()

    def draw_samples(self, num_series: int, num_time_steps: int,
                     seed: Optional[int], time_scale_density: float,
                     coordination: np.ndarray) -> SerializedComponentSamples:

        assert self.dim_value == self.parameters.mean_a0.shape[0]
        assert self.dim_value == self.parameters.sd_aa.shape[0]

        set_random_seed(seed)

        samples = SerializedComponentSamples()
        samples.subjects = self._draw_random_subjects(num_series, num_time_steps, time_scale_density)

        samples.values = np.array((num_series, self.dim_value, num_time_steps))
        samples.prev_time_same_subject = np.full(shape=(num_series, num_time_steps), fill_value=-1)
        samples.prev_time_diff_subject = np.full(shape=(num_series, num_time_steps), fill_value=-1)

        # Fill indices of previous subjects
        for s in range(num_series):
            prev_time_per_subject = {}

            for t in range(num_time_steps):
                samples.prev_time_same_subject[s, t] = prev_time_per_subject.get(samples.subjects[s, t], -1)

                for subject, time in prev_time_per_subject.items():
                    if subject == samples.subjects[s, t]:
                        continue

                    # Most recent time from a different subject
                    samples.prev_time_diff_subject = time if samples.prev_time_diff_subject == -1 else max(
                        samples.prev_time_diff_subject, time)

                prev_time_per_subject[samples.subjects[s, t]] = t

        for t in range(num_time_steps):
            c = coordination[:, t][:, None]

            # Indices of series with no previous value from any subject
            no_prev_subject = np.logical_and(samples.prev_time_same_subject[:, t] == -1,
                                             samples.prev_time_diff_subject[:, t] == -1)[:, None]

            if self.self_dependent:
                # When there's self dependency, the component either depends on the previous value of another subject,
                # or the previous value of the same subject.
                prev_same_mask = (samples.prev_time_same_subject[:, t] != -1).astype(int)[:, None]
                prev_same_value = samples.values[..., samples.prev_time_same_subject]
            else:
                # When there's no self dependency, the component either depends on the previous value of another subject,
                # or it is samples around a fixed mean.
                prev_same_mask = np.ones((num_series, 1))
                prev_same_value = np.ones((num_series, self.dim_value)) * self.parameters.mean_a0

            prev_diff_mask = (samples.prev_time_diff_subject[:, t] != -1).astype(int)[:, None]
            prev_diff_value = samples.values[..., samples.prev_time_diff_subject]

            mean_with_prev_subject = (prev_diff_value - prev_same_value * prev_same_mask) * prev_diff_mask * c + \
                                     prev_same_value * prev_same_mask

            means = np.where(no_prev_subject, self.parameters.mean_a0, mean_with_prev_subject)

            samples.values[..., t] = norm(loc=means, scale=self.parameters.sd_aa).rvs()

        return samples

    def _draw_random_subjects(self, num_series: int, num_time_steps: int, time_scale_density: float) -> np.ndarray:
        # Subject 0 is "No Subject"
        transition_matrix = np.full(shape=(self.num_subjects + 1, self.num_subjects + 1),
                                    fill_value=time_scale_density / (self.num_subjects - 1))
        transition_matrix[:, 0] = 1 - time_scale_density
        initial_prob = transition_matrix[0]

        subjects = np.zeros((num_series, num_time_steps))

        for t in range(num_time_steps):
            if t == 0:
                subjects[:, t] = np.random.choice(self.num_subjects + 1, num_series, p=initial_prob)
            else:
                probs = transition_matrix[subjects[:, t - 1]]
                cum_prob = np.cumsum(probs, axis=-1)
                u = np.random.uniform(size=num_series)
                subjects[:, t] = np.argmax(u < cum_prob)

        return subjects

    def update_pymc_model(self, coordination: Any, prev_time_same_subject: pt.TensorConstant,
                          prev_time_diff_subject: pt.TensorConstant, prev_same_subject_mask: pt.TensorConstant,
                          prev_diff_subject_mask: pt.TensorConstant, subject_mask: pt.TensorConstant,
                          feature_dimension: str, time_dimension: str) -> Any:

        mean_a0 = pm.HalfNormal(name=f"mean_a0_{self.uuid}", sigma=1, size=self.dim_value,
                                observed=self.parameters.mean_a0)
        sd_aa = pm.HalfNormal(name=f"sd_aa_{self.uuid}", sigma=1, size=self.dim_value, observed=self.parameters.sd_aa)

        if self.self_dependent:
            logp_params = (mean_a0,
                           sd_aa,
                           coordination,
                           pt.constant(prev_time_same_subject),
                           pt.constant(prev_time_diff_subject),
                           pt.constant(prev_same_subject_mask),
                           pt.constant(prev_diff_subject_mask),
                           pt.constant(subject_mask))
            serialized_component = pm.DensityDist(self.uuid, *logp_params, logp=serialized_logp_with_self_dependency,
                                                  dims=[feature_dimension, time_dimension])
        else:
            logp_params = (mean_a0,
                           sd_aa,
                           coordination,
                           pt.constant(prev_time_diff_subject),
                           pt.constant(prev_diff_subject_mask),
                           pt.constant(subject_mask))
            serialized_component = pm.DensityDist(self.uuid, *logp_params, logp=serialized_logp_without_self_dependency,
                                                  dims=[feature_dimension, time_dimension])

        return serialized_component
