from typing import Any, List, Optional, Tuple

import numpy as np
import pymc as pm
import pytensor.tensor as ptt
from scipy.stats import norm

from coordination.common.utils import set_random_seed
from coordination.model.parametrization import Parameter, HalfNormalParameterPrior


def serialized_logp_with_self_dependency(serialized_component: Any,
                                         initial_mean: Any,
                                         sigma: Any,
                                         coordination: Any,
                                         prev_time_same_subject: ptt.TensorConstant,
                                         prev_time_diff_subject: ptt.TensorConstant,
                                         prev_same_subject_mask: ptt.TensorConstant,
                                         prev_diff_subject_mask: ptt.TensorConstant):
    C = coordination[None, :]  # 1 x t-1
    S = serialized_component[..., prev_time_same_subject]  # d x t-1
    D = serialized_component[..., prev_time_diff_subject]  # d x t-1

    SM = prev_same_subject_mask[None, :]
    DM = prev_diff_subject_mask[None, :]

    # Already contains the initial time step.
    mean = (D - S * SM) * C * DM + S * SM + initial_mean * (1 - SM) * (1 - DM)

    total_logp = pm.logp(pm.Normal.dist(mu=mean, sigma=sigma, shape=D.shape), serialized_component).sum()

    return total_logp


def serialized_logp_without_self_dependency(serialized_component: Any,
                                            initial_mean: Any,
                                            sigma: Any,
                                            coordination: Any,
                                            prev_time_diff_subject: ptt.TensorConstant,
                                            prev_diff_subject_mask: ptt.TensorConstant):
    C = coordination[None, 1:]  # 1 x t-1
    S = initial_mean[..., 1:]  # d x t-1. Sampled around a fixed value if there's no coordination
    D = serialized_component[..., prev_time_diff_subject][..., 1:]  # d x t-1

    DM = prev_diff_subject_mask[None, 1:]

    # Initial time step
    total_logp = pm.logp(
        pm.Normal.dist(mu=initial_mean[..., 0], sigma=sigma[..., 0], shape=serialized_component.shape[:-1]),
        serialized_component[..., 0]).sum()

    # Transition
    mean = (D - S) * C * DM + S

    total_logp += pm.logp(pm.Normal.dist(mu=mean, sigma=sigma[..., 1:], shape=D.shape),
                          serialized_component[..., 1:]).sum()

    return total_logp


def serialized_random_with_self_dependency(initial_mean: np.ndarray,
                                           sigma: np.ndarray,
                                           coordination: np.ndarray,
                                           prev_time_same_subject: np.ndarray,
                                           prev_time_diff_subject: np.ndarray,
                                           prev_same_subject_mask: ptt.TensorConstant,
                                           prev_diff_subject_mask: ptt.TensorConstant,
                                           rng: Optional[np.random.Generator] = None,
                                           size: Optional[Tuple[int]] = None) -> np.ndarray:
    num_time_steps = coordination.shape[-1]

    noise = rng.normal(loc=0, scale=1, size=size) * sigma

    sample = np.zeros_like(noise)
    prior_sample = rng.normal(loc=initial_mean[..., 0], scale=sigma[..., 0])
    sample[..., 0] = prior_sample
    for t in np.arange(1, num_time_steps):
        # Previous sample from a different individual
        D = sample[..., prev_time_diff_subject[t]]
        # Previous sample from the same individual
        S = sample[..., prev_time_same_subject[t]] * prev_same_subject_mask[t]

        mean = ((D - S) * coordination[t] * prev_diff_subject_mask[t] + S)

        transition_sample = rng.normal(loc=mean, scale=sigma[..., t])

        sample[..., t] = transition_sample

    return sample + noise


def serialized_random_without_self_dependency(initial_mean: np.ndarray,
                                              sigma: np.ndarray,
                                              coordination: np.ndarray,
                                              prev_time_diff_subject: np.ndarray,
                                              prev_diff_subject_mask: ptt.TensorConstant,
                                              rng: Optional[np.random.Generator] = None,
                                              size: Optional[Tuple[int]] = None) -> np.ndarray:
    num_time_steps = coordination.shape[-1]

    noise = rng.normal(loc=0, scale=1, size=size) * sigma

    sample = np.zeros_like(noise)

    for t in np.arange(1, num_time_steps):
        # Previous sample from a different individual
        D = sample[..., prev_time_diff_subject[t]]

        # No self-dependency. The transition distribution is a blending between the previous value from another individual,
        # and a fixed mean.
        S = initial_mean[..., t]

        mean = ((D - S) * coordination[t] * prev_diff_subject_mask[t] + S)

        transition_sample = rng.normal(loc=mean, scale=sigma[..., t])

        sample[..., t] = transition_sample

    return sample + noise


class SerializedComponentParameters:

    def __init__(self, sd_mean_a0: np.ndarray, sd_sd_aa: np.ndarray):
        self.mean_a0 = Parameter(HalfNormalParameterPrior(sd_mean_a0))
        self.sd_aa = Parameter(HalfNormalParameterPrior(sd_sd_aa))

    def clear_values(self):
        self.mean_a0.value = None
        self.sd_aa.value = None


class SerializedComponentSamples:
    """
    If the density is smaller than one, each time series will have a different number of time steps. So we store
    # each one in a list instead of in the first dimension of a numpy array.
    """

    def __init__(self):
        self.values: List[np.ndarray] = []

        # Number indicating which subject is associated to the component at a time (e.g. the current speaker for
        # a vocalics component).
        self.subjects: List[np.ndarray] = []

        # Time indices indicating the previous occurrence of the component produced by the same subject and the most
        # recent different one. For instance, the last time when the current speaker talked and a different speaker.
        self.prev_time_same_subject: List[np.ndarray] = []
        self.prev_time_diff_subject: List[np.ndarray] = []

        # For each time step in the component's scale, it contains the time step in the coordination scale
        self.time_steps_in_coordination_scale: List[np.ndarray] = []

    @property
    def num_time_steps(self):
        if len(self.values) == 0:
            return 0

        return self.values[0].shape[-1]

    @property
    def prev_time_same_subject_mask(self):
        return [np.where(x >= 0, 1, 0) for x in self.prev_time_same_subject]

    @property
    def prev_time_diff_subject_mask(self):
        return [np.where(x >= 0, 1, 0) for x in self.prev_time_diff_subject]


class SerializedComponent:

    def __init__(self, uuid: str, num_subjects: int, dim_value: int, self_dependent: bool, sd_mean_a0: np.ndarray,
                 sd_sd_aa: np.ndarray):
        self.uuid = uuid
        self.num_subjects = num_subjects
        self.dim_value = dim_value
        self.self_dependent = self_dependent

        self.parameters = SerializedComponentParameters(sd_mean_a0, sd_sd_aa)

    def draw_samples(self, num_series: int, time_scale_density: float,
                     coordination: np.ndarray, can_repeat_subject: bool,
                     seed: Optional[int] = None) -> SerializedComponentSamples:

        assert 0 <= time_scale_density <= 1
        assert (self.num_subjects, self.dim_value) == self.parameters.mean_a0.value.shape
        assert (self.num_subjects, self.dim_value) == self.parameters.sd_aa.value.shape

        set_random_seed(seed)

        samples = SerializedComponentSamples()
        sparse_subjects = self._draw_random_subjects(num_series, coordination.shape[-1], time_scale_density,
                                                     can_repeat_subject)

        for s in range(num_series):
            samples.subjects.append(np.array([s for s in sparse_subjects[s] if s >= 0], dtype=int))
            samples.time_steps_in_coordination_scale.append(
                np.array([t for t, s in enumerate(sparse_subjects[s]) if s >= 0], dtype=int))

            num_time_steps_in_cpn_scale = len(samples.time_steps_in_coordination_scale[s])

            samples.values.append(np.zeros((self.dim_value, num_time_steps_in_cpn_scale)))
            samples.prev_time_same_subject.append(
                np.full(shape=num_time_steps_in_cpn_scale, fill_value=-1, dtype=int))
            samples.prev_time_diff_subject.append(
                np.full(shape=num_time_steps_in_cpn_scale, fill_value=-1, dtype=int))

            prev_time_per_subject = {}

            for t in range(num_time_steps_in_cpn_scale):
                samples.prev_time_same_subject[s][t] = prev_time_per_subject.get(samples.subjects[s][t], -1)

                for subject, time in prev_time_per_subject.items():
                    if subject == samples.subjects[s][t]:
                        continue

                    # Most recent time from a different subject
                    samples.prev_time_diff_subject[s][t] = time if samples.prev_time_diff_subject[s][t] == -1 else max(
                        samples.prev_time_diff_subject[s][t], time)

                prev_time_per_subject[samples.subjects[s][t]] = t

                if t == 0:
                    samples.values[s][:, t] = norm(loc=self.parameters.mean_a0.value[samples.subjects[s][t]],
                                                   scale=self.parameters.sd_aa.value[samples.subjects[s][t]]).rvs(
                        size=self.dim_value)
                else:
                    C = coordination[s, samples.time_steps_in_coordination_scale[s][t]]

                    if self.self_dependent:
                        # When there's self dependency, the component either depends on the previous value of another subject,
                        # or the previous value of the same subject.
                        prev_same_mask = (samples.prev_time_same_subject[s][t] != -1).astype(int)
                        S = samples.values[s][..., samples.prev_time_same_subject[s][t]]
                    else:
                        # When there's no self dependency, the component either depends on the previous value of another subject,
                        # or it is samples around a fixed mean.
                        prev_same_mask = 1
                        S = self.parameters.mean_a0.value[samples.subjects[s][t]]

                    prev_diff_mask = (samples.prev_time_diff_subject[s][t] != -1).astype(int)
                    D = samples.values[s][..., samples.prev_time_diff_subject[s][t]]

                    mean = (D - S * prev_same_mask) * C * prev_diff_mask + S * prev_same_mask

                    samples.values[s][:, t] = norm(loc=mean,
                                                   scale=self.parameters.sd_aa.value[samples.subjects[s][t]]).rvs()

        return samples

    def _draw_random_subjects(self, num_series: int, num_time_steps: int, time_scale_density: float,
                              can_repeat_subject: bool) -> np.ndarray:
        # Subject 0 is "No Subject"
        if can_repeat_subject:
            transition_matrix = np.full(shape=(self.num_subjects + 1, self.num_subjects + 1),
                                        fill_value=time_scale_density / self.num_subjects)
            transition_matrix[:, 0] = 1 - time_scale_density
        else:
            transition_matrix = np.full(shape=(self.num_subjects + 1, self.num_subjects + 1),
                                        fill_value=time_scale_density / (self.num_subjects - 1))
            transition_matrix[0, 1:] = time_scale_density / self.num_subjects
            transition_matrix = transition_matrix * (1 - np.eye(self.num_subjects + 1))
            transition_matrix[:, 0] = 1 - time_scale_density

        initial_prob = transition_matrix[0]
        subjects = np.zeros((num_series, num_time_steps), dtype=int)

        for t in range(num_time_steps):
            if t == 0:
                subjects[:, t] = np.random.choice(self.num_subjects + 1, num_series, p=initial_prob)
            else:
                probs = transition_matrix[subjects[:, t - 1]]
                cum_prob = np.cumsum(probs, axis=-1)
                u = np.random.uniform(size=(num_series, 1))
                subjects[:, t] = np.argmax(u < cum_prob, axis=-1)

        # Map None to -1
        subjects -= 1
        return subjects

    def update_pymc_model(self, coordination: Any, prev_time_same_subject: np.ndarray,
                          prev_time_diff_subject: np.ndarray, prev_same_subject_mask: np.ndarray,
                          prev_diff_subject_mask: np.ndarray, subjects: np.ndarray,
                          feature_dimension: str, time_dimension: str, observed_values: Optional[Any] = None) -> Any:

        mean_a0 = pm.HalfNormal(name=f"mean_a0_{self.uuid}", sigma=self.parameters.mean_a0.prior.sd,
                                size=(self.num_subjects, self.dim_value), observed=self.parameters.mean_a0.value)
        sd_aa = pm.HalfNormal(name=f"sd_aa_{self.uuid}", sigma=self.parameters.sd_aa.prior.sd,
                              size=(self.num_subjects, self.dim_value), observed=self.parameters.sd_aa.value)

        if self.self_dependent:
            logp_params = (mean_a0[subjects].transpose(),  # One mean and sd per time step matching their subjects.
                           sd_aa[subjects].transpose(),
                           coordination,
                           ptt.constant(prev_time_same_subject),
                           ptt.constant(prev_time_diff_subject),
                           ptt.constant(prev_same_subject_mask),
                           ptt.constant(prev_diff_subject_mask))
            random_fn = serialized_random_with_self_dependency
            serialized_component = pm.DensityDist(self.uuid, *logp_params, logp=serialized_logp_with_self_dependency,
                                                  random=random_fn, dims=[feature_dimension, time_dimension],
                                                  observed=observed_values)
        else:
            logp_params = (mean_a0[subjects].transpose(),
                           sd_aa[subjects].transpose(),
                           coordination,
                           ptt.constant(prev_time_diff_subject),
                           ptt.constant(prev_diff_subject_mask))
            random_fn = serialized_random_without_self_dependency
            serialized_component = pm.DensityDist(self.uuid, *logp_params, logp=serialized_logp_without_self_dependency,
                                                  random=random_fn, dims=[feature_dimension, time_dimension],
                                                  observed=observed_values)

        return serialized_component, mean_a0, sd_aa
