from typing import Any, List, Optional, Tuple

import numpy as np
import pymc as pm
import pytensor.tensor as ptt
from scipy.stats import norm

from coordination.common.utils import set_random_seed
from coordination.module.parametrization import Parameter, HalfNormalParameterPrior, \
    NormalParameterPrior


def logp(sample: Any,
         initial_mean: Any,
         sigma: Any,
         coordination: Any,
         prev_time_same_subject: ptt.TensorConstant,
         prev_time_diff_subject: ptt.TensorConstant,
         prev_same_subject_mask: Any,
         prev_diff_subject_mask: Any,
         self_dependent: ptt.TensorConstant):
    """
    This function computes the log-probability of a serial component. We use the following definition in the
    comments below:

    d: number of dimensions/features of the component
    T: number of time steps in the component's scale
    """

    # We use 'prev_time_diff_subject' as meta-data to get the values from partners of the subjects in each time
    # step. We reshape to guarantee we don't create dimensions with unknown size in case the first dimension of
    # the sample component is one.
    prev_other = sample[..., prev_time_diff_subject].reshape(sample.shape)  # d x T

    # We use this binary mask to zero out entries with no observations from partners.
    mask_other = prev_diff_subject_mask[None, :]  # 1 x T

    c = coordination[None, :]  # 1 x T
    if self_dependent.eval():
        # The component's value for a subject depends on its previous value for the same subject.
        prev_same = sample[..., prev_time_same_subject].reshape(sample.shape)  # d x T

        # We use this binary mask to zero out entries with no previous observations from the subjects. We use this
        # to determine the time steps that belong to the initial values of the component. Each subject will have their
        # initial value in a different time step hence we cannot just use t=0.
        mask_same = prev_same_subject_mask[None, :]  # 1 x t
        blended_mean = prev_other * c * mask_other + (1 - c * mask_other) * (
                prev_same * mask_same + (1 - mask_same) * initial_mean)
    else:
        # The component's value for a subject does not depend on its previous value for the same subject.
        # At every time step, the value from others is blended with a fixed value given by the component's initial
        # mean.
        blended_mean = prev_other * c * mask_other + (1 - c * mask_other) * initial_mean

    total_logp = pm.logp(pm.Normal.dist(mu=blended_mean, sigma=sigma, shape=prev_other.shape),
                         sample).sum()

    return total_logp


def random(initial_mean: np.ndarray,
           sigma: np.ndarray,
           coordination: np.ndarray,
           prev_time_same_subject: np.ndarray,
           prev_time_diff_subject: np.ndarray,
           prev_same_subject_mask: np.ndarray,
           prev_diff_subject_mask: np.ndarray,
           self_dependent: bool,
           rng: Optional[np.random.Generator] = None,
           size: Optional[Tuple[int]] = None) -> np.ndarray:
    """
    This function generates samples from of a serial component for prior predictive checks. We use the following
    definition in the comments below:

    d: number of dimensions/features of the component
    T: number of time steps in the component's scale
    """

    T = coordination.shape[-1]

    noise = rng.normal(loc=0, scale=1, size=size) * sigma

    sample = np.zeros_like(noise)

    mean_0 = initial_mean if initial_mean.ndim == 1 else initial_mean[..., 0]
    sd_0 = sigma if sigma.ndim == 1 else sigma[..., 0]

    sample[..., 0] = rng.normal(loc=mean_0, scale=sd_0)

    for t in np.arange(1, T):
        prev_other = sample[..., prev_time_diff_subject[t]]  # d

        # Previous sample from the same individual
        if self_dependent and prev_same_subject_mask[t] == 1:
            prev_same = sample[..., prev_time_same_subject[t]]
        else:
            # When there's no self-dependency, the transition distribution is a blending between the previous value
            # from another individual, and a fixed mean.
            if initial_mean.shape[1] == 1:
                prev_same = initial_mean[..., 0]
            else:
                prev_same = initial_mean[..., t]

        mean = ((prev_other - prev_same) * coordination[t] * prev_diff_subject_mask[t] + prev_same)

        if sigma.shape[1] == 1:
            # Parameter sharing across subjects
            transition_sample = rng.normal(loc=mean, scale=sigma[..., 0])
        else:
            transition_sample = rng.normal(loc=mean, scale=sigma[..., t])

        sample[..., t] = transition_sample

    return sample + noise


class SerialComponentParameters:

    def __init__(self, mean_mean_a0: np.ndarray, sd_mean_a0: np.ndarray, sd_sd_aa: np.ndarray):
        self.mean_a0 = Parameter(NormalParameterPrior(mean_mean_a0, sd_mean_a0))
        self.sd_aa = Parameter(HalfNormalParameterPrior(sd_sd_aa))

    def clear_values(self):
        self.mean_a0.value = None
        self.sd_aa.value = None


class SerialComponentSamples:

    def __init__(self):
        """
        If the density of observations is smaller than one, each time series will have a different number of time steps.
        So we store each one in a list instead of in the first dimension of a numpy array.
        """

        self.values: List[np.ndarray] = []

        # Number indicating which subject is associated to the component at a time (e.g. the current speaker for
        # a vocalics component).
        self.subjects: List[np.ndarray] = []

        # Time indices indicating the previous occurrence of the component produced by the same subject and their
        # most recent partner. For instance, the last time when the current speaker talked and a different speaker.
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


class SerialComponent:
    """
    This class models a serial latent component which individual subject's dynamics influence that of the other
    subjects as controlled by coordination.
    """

    def __init__(self,
                 uuid: str,
                 num_subjects: int,
                 dim_value: int,
                 self_dependent: bool,
                 mean_mean_a0: np.ndarray,
                 sd_mean_a0: np.ndarray,
                 sd_sd_aa: np.ndarray,
                 share_mean_a0_across_subjects: bool,
                 share_mean_a0_across_features: bool,
                 share_sd_aa_across_subjects: bool,
                 share_sd_aa_across_features: bool):

        # Check dimensionality of the hyper-prior parameters
        if share_mean_a0_across_features:
            dim_mean_a0_features = 1
        else:
            dim_mean_a0_features = dim_value

        if share_sd_aa_across_features:
            dim_sd_aa_features = 1
        else:
            dim_sd_aa_features = dim_value

        if share_mean_a0_across_subjects:
            assert (dim_mean_a0_features,) == mean_mean_a0.shape
            assert (dim_mean_a0_features,) == sd_mean_a0.shape
        else:
            assert (num_subjects, dim_mean_a0_features) == mean_mean_a0.shape
            assert (num_subjects, dim_mean_a0_features) == sd_mean_a0.shape

        if share_sd_aa_across_subjects:
            assert (dim_sd_aa_features,) == sd_sd_aa.shape
        else:
            assert (num_subjects, dim_sd_aa_features) == sd_sd_aa.shape

        self.uuid = uuid
        self.num_subjects = num_subjects
        self.dim_value = dim_value
        self.self_dependent = self_dependent
        self.share_mean_a0_across_subjects = share_mean_a0_across_subjects
        self.share_mean_a0_across_features = share_mean_a0_across_features
        self.share_sd_aa_across_subjects = share_sd_aa_across_subjects
        self.share_sd_aa_across_features = share_sd_aa_across_features

        self.parameters = SerialComponentParameters(mean_mean_a0=mean_mean_a0,
                                                    sd_mean_a0=sd_mean_a0,
                                                    sd_sd_aa=sd_sd_aa)

    @property
    def parameter_names(self) -> List[str]:
        names = [
            self.mean_a0_name,
            self.sd_aa_name
        ]

        return names

    @property
    def mean_a0_name(self) -> str:
        return f"mean_a0_{self.uuid}"

    @property
    def sd_aa_name(self) -> str:
        return f"sd_aa_{self.uuid}"

    def clear_parameter_values(self):
        self.parameters.clear_values()

    def draw_samples(self,
                     num_series: int,
                     time_scale_density: float,
                     coordination: np.ndarray,
                     can_repeat_subject: bool,
                     seed: Optional[int] = None,
                     fixed_subject_sequence: bool = False) -> SerialComponentSamples:

        # Check dimensionality of the parameters
        if self.share_mean_a0_across_features:
            dim_mean_a0_features = 1
        else:
            dim_mean_a0_features = self.dim_value

        if self.share_sd_aa_across_features:
            dim_sd_aa_features = 1
        else:
            dim_sd_aa_features = self.dim_value

        if self.share_mean_a0_across_subjects:
            assert (dim_mean_a0_features,) == self.parameters.mean_a0.value.shape
        else:
            assert (self.num_subjects, dim_mean_a0_features) == self.parameters.mean_a0.value.shape

        if self.share_sd_aa_across_subjects:
            assert (dim_sd_aa_features,) == self.parameters.sd_aa.value.shape
        else:
            assert (self.num_subjects, dim_sd_aa_features) == self.parameters.sd_aa.value.shape

        assert 0 <= time_scale_density <= 1

        # Adjust dimensions according to parameter sharing specification
        if self.share_mean_a0_across_subjects:
            mean_a0 = self.parameters.mean_a0.value[None, :].repeat(self.num_subjects, axis=0)
        else:
            mean_a0 = self.parameters.mean_a0.value

        if self.share_sd_aa_across_subjects:
            sd_aa = self.parameters.sd_aa.value[None, :].repeat(self.num_subjects, axis=0)
        else:
            sd_aa = self.parameters.sd_aa.value

        # Generate samples
        set_random_seed(seed)

        samples = SerialComponentSamples()
        samples.values = []
        for s in range(num_series):
            sparse_subjects = self._draw_random_subjects(num_series,
                                                         coordination.shape[-1],
                                                         time_scale_density,
                                                         can_repeat_subject,
                                                         fixed_subject_sequence)
            samples.subjects.append(np.array([s for s in sparse_subjects[s] if s >= 0], dtype=int))
            samples.time_steps_in_coordination_scale.append(
                np.array([t for t, s in enumerate(sparse_subjects[s]) if s >= 0], dtype=int))

            num_time_steps_in_cpn_scale = len(samples.time_steps_in_coordination_scale[s])

            samples.prev_time_same_subject.append(
                np.full(shape=num_time_steps_in_cpn_scale, fill_value=-1, dtype=int))
            samples.prev_time_diff_subject.append(
                np.full(shape=num_time_steps_in_cpn_scale, fill_value=-1, dtype=int))

            # Fill dependencies
            prev_time_per_subject = {}
            for t in range(num_time_steps_in_cpn_scale):
                samples.prev_time_same_subject[s][t] = prev_time_per_subject.get(
                    samples.subjects[s][t], -1)

                for subject, time in prev_time_per_subject.items():
                    if subject == samples.subjects[s][t]:
                        continue

                    # Most recent time from a different subject
                    samples.prev_time_diff_subject[s][t] = time if \
                        samples.prev_time_diff_subject[s][t] == -1 else max(
                        samples.prev_time_diff_subject[s][t], time)

                prev_time_per_subject[samples.subjects[s][t]] = t

            # Draw values from the system dynamics. The default model generates samples by following a Gaussian random
            # walk with blended values from different subjects according to the coordination levels over time. Child
            # classes can implement their own dynamics, like spring-mass-damping systems for instance.
            values = self._draw_from_system_dynamics(
                time_steps_in_coordination_scale=samples.time_steps_in_coordination_scale[s],
                sampled_coordination=coordination[s],
                subjects_in_time=samples.subjects[s],
                prev_time_same_subject=samples.prev_time_same_subject[s],
                prev_time_diff_subject=samples.prev_time_diff_subject[s],
                mean_a0=mean_a0,
                sd_aa=sd_aa)
            samples.values.append(values)

        return samples

    def _draw_from_system_dynamics(self,
                                   time_steps_in_coordination_scale: np.ndarray,
                                   sampled_coordination: np.ndarray,
                                   subjects_in_time: np.ndarray,
                                   prev_time_same_subject: np.ndarray,
                                   prev_time_diff_subject: np.ndarray,
                                   mean_a0: np.ndarray,
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
                # Sample from prior. It is not only when t=0 because the first utterance of a speaker can be later in
                # the future. t=0 is the initial state of one of the subjects only.

                mean = mean_a0[subject_idx_mean_a0]
                values[:, t] = norm(loc=mean, scale=sd).rvs(size=self.dim_value)
            else:
                c = sampled_coordination[time_steps_in_coordination_scale[t]]

                if self.self_dependent:
                    # When there's self dependency, the component either depends on the previous value of another
                    # subject or the previous value of the same subject.
                    prev_same = values[..., prev_time_same_subject[t]]
                else:
                    # When there's no self dependency, the component either depends on the previous value of another
                    # subject or a fixed value.
                    prev_same = mean_a0[subject_idx_mean_a0]

                mask_other = (prev_time_diff_subject[t] != -1).astype(int)
                prev_other = values[..., prev_time_diff_subject[t]]

                blended_mean = (prev_other - prev_same) * c * mask_other + prev_same

                values[:, t] = norm(loc=blended_mean, scale=sd).rvs()

        return values

    def _draw_random_subjects(self,
                              num_series: int,
                              num_time_steps: int,
                              time_scale_density: float,
                              can_repeat_subject: bool,
                              fixed_subject_sequence: bool) -> np.ndarray:

        # Subject 0 is "No Subject"
        if can_repeat_subject:
            # We allow the same subject to appear in subsequent observations
            if fixed_subject_sequence:
                transition_matrix = np.zeros(shape=(self.num_subjects + 1, self.num_subjects + 1))
                transition_matrix[:, 0] = 1 - time_scale_density
                transition_matrix[0, 1] = time_scale_density
                transition_matrix[-1, 1] = time_scale_density / 2
                transition_matrix[-1, -1] = time_scale_density / 2
                for s1 in range(1, self.num_subjects):
                    for s2 in range(1, self.num_subjects+1):
                        if s1 == s2 or s2 == s1 + 1:
                            transition_matrix[s1, s2] = time_scale_density/2

            else:
                transition_matrix = np.full(shape=(self.num_subjects + 1, self.num_subjects + 1),
                                            fill_value=time_scale_density / self.num_subjects)
                transition_matrix[:, 0] = 1 - time_scale_density
        else:
            if fixed_subject_sequence:
                transition_matrix = np.zeros(shape=(self.num_subjects + 1, self.num_subjects + 1))
                transition_matrix[:, 0] = 1 - time_scale_density
                transition_matrix[:-1, 1:] = np.eye(self.num_subjects) * time_scale_density
                transition_matrix[-1, 1] = time_scale_density
            else:
                transition_matrix = np.full(shape=(self.num_subjects + 1, self.num_subjects + 1),
                                            fill_value=time_scale_density / (
                                                        self.num_subjects - 1))
                transition_matrix[0, 1:] = time_scale_density / self.num_subjects
                transition_matrix = transition_matrix * (1 - np.eye(self.num_subjects + 1))
                transition_matrix[:, 0] = 1 - time_scale_density

        initial_prob = transition_matrix[0]
        subjects = np.zeros((num_series, num_time_steps), dtype=int)

        for t in range(num_time_steps):
            if t == 0:
                subjects[:, t] = np.random.choice(self.num_subjects + 1, num_series,
                                                  p=initial_prob)
            else:
                probs = transition_matrix[subjects[:, t - 1]]
                cum_prob = np.cumsum(probs, axis=-1)
                u = np.random.uniform(size=(num_series, 1))
                subjects[:, t] = np.argmax(u < cum_prob, axis=-1)

        # Map 0 to -1
        subjects -= 1
        return subjects

    def _create_random_parameters(self,
                                  subjects: np.ndarray,
                                  mean_a0: Optional[Any] = None,
                                  sd_aa: Optional[Any] = None):
        """
        This function creates the initial mean and standard deviation of the serial component distribution as
        random variables.
        """

        # Adjust feature dimensionality according to sharing options
        if self.share_mean_a0_across_features:
            dim_mean_a0_features = 1
        else:
            dim_mean_a0_features = self.dim_value

        if self.share_sd_aa_across_features:
            dim_sd_aa_features = 1
        else:
            dim_sd_aa_features = self.dim_value

        # Initialize mean_a0 parameter if it hasn't been defined previously
        if mean_a0 is None:
            if self.share_mean_a0_across_subjects:
                mean_a0 = pm.Normal(name=self.mean_a0_name,
                                    mu=self.parameters.mean_a0.prior.mean,
                                    sigma=self.parameters.mean_a0.prior.sd,
                                    size=dim_mean_a0_features,
                                    observed=self.parameters.mean_a0.value)
                mean_a0 = mean_a0[:, None]  # feature x time = 1 (broadcast across time)
            else:
                mean_a0 = pm.Normal(name=self.mean_a0_name,
                                    mu=self.parameters.mean_a0.prior.mean,
                                    sigma=self.parameters.mean_a0.prior.sd,
                                    size=(self.num_subjects, dim_mean_a0_features),
                                    observed=self.parameters.mean_a0.value)
                mean_a0 = mean_a0[subjects].transpose()  # feature x time

            if self.share_mean_a0_across_features:
                mean_a0 = mean_a0.repeat(self.dim_value, axis=0)

        # Initialize sd_aa parameter if it hasn't been defined previously
        if sd_aa is None:
            if self.share_sd_aa_across_subjects:
                sd_aa = pm.HalfNormal(name=self.sd_aa_name,
                                      sigma=self.parameters.sd_aa.prior.sd,
                                      size=dim_sd_aa_features,
                                      observed=self.parameters.sd_aa.value)
                sd_aa = sd_aa[:, None]  # feature x time = 1 (broadcast across time)
            else:
                sd_aa = pm.HalfNormal(name=self.sd_aa_name,
                                      sigma=self.parameters.sd_aa.prior.sd,
                                      size=(self.num_subjects, dim_sd_aa_features),
                                      observed=self.parameters.sd_aa.value)
                sd_aa = sd_aa[subjects].transpose()  # feature x time

            if self.share_sd_aa_across_features:
                sd_aa = sd_aa.repeat(self.dim_value, axis=0)

        return mean_a0, sd_aa

    def update_pymc_model(self,
                          coordination: Any,
                          prev_time_same_subject: np.ndarray,
                          prev_time_diff_subject: np.ndarray,
                          prev_same_subject_mask: np.ndarray,
                          prev_diff_subject_mask: np.ndarray,
                          subjects: np.ndarray,
                          feature_dimension: str,
                          time_dimension: str,
                          observed_values: Optional[Any] = None,
                          mean_a0: Optional[Any] = None,
                          sd_aa: Optional[Any] = None) -> Any:

        mean_a0, sd_aa = self._create_random_parameters(subjects, mean_a0, sd_aa)

        logp_params = (mean_a0,
                       sd_aa,
                       coordination,
                       prev_time_same_subject,
                       prev_time_diff_subject,
                       prev_same_subject_mask,
                       prev_diff_subject_mask,
                       np.array(self.self_dependent),
                       *self._get_extra_logp_params(subjects)
                       )
        logp_fn = self._get_logp_fn()
        random_fn = self._get_random_fn()
        serial_component = pm.DensityDist(self.uuid, *logp_params,
                                          logp=logp_fn,
                                          random=random_fn,
                                          dims=[feature_dimension, time_dimension],
                                          observed=observed_values)

        return serial_component, mean_a0, sd_aa

    def _get_extra_logp_params(self, subjects_in_time: np.ndarray):
        """
        Child classes can pass extra parameters to the logp and random functions
        """
        return ()

    def _get_logp_fn(self):
        """
        Child classes can define their own logp functions
        """
        return logp

    def _get_random_fn(self):
        """
        Child classes can define their own random functions for prior predictive checks
        """
        return random
