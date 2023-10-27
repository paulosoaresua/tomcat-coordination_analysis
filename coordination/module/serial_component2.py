from __future__ import annotations
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pymc as pm
import pytensor.tensor as ptt
from scipy.stats import norm

from coordination.common.utils import set_random_seed
from coordination.module.parametrization import Parameter, HalfNormalParameterPrior, \
    NormalParameterPrior
from coordination.module.latent_component import LatentComponent, LatentComponentSamples, \
    LatentComponentParameters


class SerialComponent(LatentComponent):
    """
    This class represents a serial latent component where there's only one observation per subject
    at a time in the component's scale, and subjects are influenced in a pairwise manner.
    """

    def __init__(self,
                 uuid: str,
                 num_subjects: int,
                 dimension_size: int,
                 self_dependent: bool,
                 mean_mean_a0: np.ndarray,
                 sd_mean_a0: np.ndarray,
                 sd_sd_aa: np.ndarray,
                 share_mean_a0_across_subjects: bool,
                 share_mean_a0_across_dimensions: bool,
                 share_sd_aa_across_subjects: bool,
                 share_sd_aa_across_dimensions: bool,
                 dimension_names: Optional[List[str]] = None):
        """
        Creates a serial latent component.

        @param uuid: String uniquely identifying the latent component in the model.
        @param num_subjects: the number of subjects that possess the component.
        @param dimension_size: the number of dimensions in the latent component.
        @param self_dependent: whether the latent variables in the component are tied to the
        past values from the same subject. If False, coordination will blend the previous latent
        value of a different subject with the value of the component at time t = 0 for the current
        subject (the latent component's prior for that subject).
        @param mean_mean_a0: mean of the hyper-prior of mu_a0 (mean of the initial value of the
        latent component).
        @param sd_sd_aa: std of the hyper-prior of sigma_aa (std of the Gaussian random walk of
        the latent component).
        @param share_mean_a0_across_subjects: whether to use the same mu_a0 for all subjects.
        @param share_mean_a0_across_dimensions: whether to use the same mu_a0 for all dimensions.
        @param share_sd_aa_across_subjects: whether to use the same sigma_aa for all subjects.
        @param share_sd_aa_across_dimensions: whether to use the same sigma_aa for all dimensions.
        @param dimension_names: the names of each dimension of the latent component. If not
        informed, this will be filled with numbers 0,1,2 up to dimension_size - 1.
        """
        super().__init__(uuid=uuid,
                         num_subjects=num_subjects,
                         dimension_size=dimension_size,
                         self_dependent=self_dependent,
                         mean_mean_a0=mean_mean_a0,
                         sd_mean_a0=sd_mean_a0,
                         sd_sd_aa=sd_sd_aa,
                         share_mean_a0_across_subjects=share_mean_a0_across_subjects,
                         share_mean_a0_across_dimensions=share_mean_a0_across_dimensions,
                         share_sd_aa_across_subjects=share_sd_aa_across_subjects,
                         share_sd_aa_across_dimensions=share_sd_aa_across_dimensions,
                         dimension_names=dimension_names)

    def draw_samples(self,
                     coordination: np.ndarray,
                     seed: Optional[int],
                     time_scale_density: float = None,
                     can_repeat_subject: bool = None,
                     fixed_subject_sequence: bool = None,
                     **kwargs) -> LatentComponentSamples:
        """
        Draws latent component samples using ancestral sampling and pairwise blending with
        coordination and different subjects.

        @param coordination: sampled coordination values.
        @param seed: random seed for reproducibility.
        @param time_scale_density: a number between 0 and 1 indicating the frequency in which we
        have observations for the latent component. If 1, the latent component is observed at every
        time in the coordination time scale, if 0.5, in average, only half of the time. The final
        number of observations is not a deterministic function of the time density, as the density
        is used to define transition probabilities between a subjects and a non-existing subject.
        @param can_repeat_subject: whether subsequent observations can come from the same subject.
        @param fixed_subject_sequence: whether the sequence of subjects is fixed (0,1,...,n,0,1...)
        or randomized.
        @param kwargs: extra arguments to be defined by subclasses.

        @return: latent component samples for each coordination series.
        """
        super().draw_samples(coordination, seed)

        self._check_parameter_dimensionality_consistency()

        # Adjust dimensions according to parameter sharing specification
        if self.share_mean_a0_across_subjects:
            mean_a0 = self.parameters.mean_a0.value[None, :].repeat(self.num_subjects, axis=0)
        else:
            mean_a0 = self.parameters.mean_a0.value

        if self.share_sd_aa_across_subjects:
            sd_aa = self.parameters.sd_aa.value[None, :].repeat(self.num_subjects, axis=0)
        else:
            sd_aa = self.parameters.sd_aa.value

        assert 0 <= time_scale_density <= 1

        # Generate samples
        set_random_seed(seed)

        num_series = coordination.shape[0]
        num_time_steps = coordination.shape[1]
        sampled_subjects = []
        sampled_values = []
        time_steps_in_coordination_scale = []
        prev_time_same_subject = []
        prev_time_diff_subject = []
        for s in range(num_series):
            sparse_subjects = self._draw_random_subjects(num_series,
                                                         num_time_steps,
                                                         time_scale_density,
                                                         can_repeat_subject,
                                                         fixed_subject_sequence)
            sampled_subjects.append(np.array([s for s in sparse_subjects[s] if s >= 0], dtype=int))

            time_steps_in_coordination_scale.append(
                np.array([t for t, s in enumerate(sparse_subjects[s]) if s >= 0], dtype=int))

            num_time_steps_in_cpn_scale = len(samples.time_steps_in_coordination_scale[s])

            prev_time_same_subject.append(
                np.full(shape=num_time_steps_in_cpn_scale, fill_value=-1, dtype=int))
            prev_time_diff_subject.append(
                np.full(shape=num_time_steps_in_cpn_scale, fill_value=-1, dtype=int))

            # Fill dependencies
            prev_time_per_subject = {}
            for t in range(num_time_steps_in_cpn_scale):
                prev_time_same_subject[s][t] = prev_time_per_subject.get(
                    sampled_subjects[s][t], -1
                )

                for subject, time in prev_time_per_subject.items():
                    if subject == sampled_subjects[s][t]:
                        continue

                    # Most recent time from a different subject
                    prev_time_diff_subject[s][t] = time if \
                        prev_time_diff_subject[s][t] == -1 else max(
                        prev_time_diff_subject[s][t], time)

                prev_time_per_subject[sampled_subjects[s][t]] = t

            values = self._draw_from_system_dynamics(
                time_steps_in_coordination_scale=time_steps_in_coordination_scale[s],
                sampled_coordination=coordination[s],
                subjects_in_time=sampled_subjects[s],
                prev_time_same_subject=prev_time_same_subject[s],
                prev_time_diff_subject=prev_time_diff_subject[s],
                mean_a0=mean_a0,
                sd_aa=sd_aa)
            sampled_values.append(values)

        return SerialComponentSamples(
            values=sampled_values,
            time_steps_in_coordination_scale=time_steps_in_coordination_scale,
            subjects=sampled_subjects,
            prev_time_same_subject=prev_time_same_subject,
            prev_time_diff_subject=prev_time_diff_subject
        )

    def _draw_from_system_dynamics(self,
                                   time_steps_in_coordination_scale: np.ndarray,
                                   sampled_coordination: np.ndarray,
                                   subjects_in_time: np.ndarray,
                                   prev_time_same_subject: np.ndarray,
                                   prev_time_diff_subject: np.ndarray,
                                   mean_a0: np.ndarray,
                                   sd_aa: np.ndarray) -> np.ndarray:
        """
        Draw values from the system dynamics. The default serialized component generates samples
        by following a Gaussian random walk with mean defined by pairwise blended values from
        different subjects according to the coordination levels over time. Child classes can
        implement their own dynamics, like spring-mass-damping systems for instance.

        @param time_steps_in_coordination_scale: an array of indices representing time steps in the
        coordination scale that match those of the component's scale.
        @param sampled_coordination: sampled values of coordination.
        @param subjects_in_time: an array indicating which subject is responsible for a latent
        component observation at a time.
        @param prev_time_same_subject: an array containing indices to most recent previous times
        the component from the same subject was observed in the component's time scale.
        @param prev_time_diff_subject: an array containing indices to most recent previous times
        the component from the a different subject was observed in the component's time scale.
        @param mean_a0: initial mean of the latent component.
        @param sd_aa: standard deviation of the Gaussian transition distribution.

        @return: sampled values.
        """

        num_time_steps = len(time_steps_in_coordination_scale)
        values = np.zeros((self.dimension_size, num_time_steps))

        for t in range(num_time_steps):

            # The way we index subjects change depending on the sharing options. If shared, the
            # the parameters will have a single dimension across subjects so we can index by 0,
            # otherwise we use the subject number to index the correct parameter for that subject.
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
                # This is the first time we see an observation for the subject. We then sample
                # from its prior. This doesn't happen only when t=0 because the first observation
                # of a subject can be later in the future. So, t=0 is the initial state of one of
                # the subjects, but not all of them.

                mean = mean_a0[subject_idx_mean_a0]
                values[:, t] = norm(loc=mean, scale=sd).rvs(size=self.dimension_size)
            else:
                c = sampled_coordination[time_steps_in_coordination_scale[t]]

                if self.self_dependent:
                    # When there's self dependency, the component either depends on the previous
                    # value of another subject or the previous value of he same subject.
                    prev_same = values[..., prev_time_same_subject[t]]
                else:
                    # When there's no self dependency, the component either depends on the previous
                    # value of another subject or a fixed value (the subject's prior).
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
        """
        Draw a random sequence of subjects over time. We define subject 0 as "No Subject", meaning
        the latent component is not observed when that subject is the one sampled in a given time
        step.

        @param num_series: number of sampled series.
        @param num_time_steps: number of time steps in the coordination time scale.
        @param time_scale_density: a number between 0 and 1 indicating the frequency in which we
        have observations for the latent component. If 1, the latent component is observed at every
        time in the coordination time scale, if 0.5, in average, only half of the time. The final
        number of observations is not a deterministic function of the time density, as the density
        is used to define transition probabilities between a subjects and a non-existing subject.
        @param can_repeat_subject: whether subsequent observations can come from the same subject.
        @param fixed_subject_sequence: whether the sequence of subjects is fixed (0,1,...,n,0,1...)
        or randomized.

        @return: a matrix of sampled subject series.
        """

        #
        if can_repeat_subject:
            # We allow the same subject to appear in subsequent observations
            if fixed_subject_sequence:
                transition_matrix = np.zeros(shape=(self.num_subjects + 1, self.num_subjects + 1))
                transition_matrix[:, 0] = 1 - time_scale_density
                transition_matrix[0, 1] = time_scale_density
                transition_matrix[-1, 1] = time_scale_density / 2
                transition_matrix[-1, -1] = time_scale_density / 2
                for s1 in range(1, self.num_subjects):
                    for s2 in range(1, self.num_subjects + 1):
                        if s1 == s2 or s2 == s1 + 1:
                            transition_matrix[s1, s2] = time_scale_density / 2

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

    def _transform_initial_mean(self, mean_a0: pm.Distribution) -> pm.Distribution:
        if self.share_mean_a0_across_subjects:
            mean_a0 = mean_a0[:, None]  # feature x time = 1 (broadcast across time)
        else:
            mean_a0 = mean_a0[subjects].transpose()  # feature x time

        if self.share_mean_a0_across_dimensions:
            mean_a0 = mean_a0.repeat(self.dimension_size, axis=0)


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
                          dimension_names: List[str],
                          observed_values: Optional[TensorTypes] = None,
                          mean_a0: Optional[TensorTypes] = None,
                          sd_aa: Optional[TensorTypes] = None,

                          prev_time_same_subject: np.ndarray,
                          prev_time_diff_subject: np.ndarray,
                          prev_same_subject_mask: np.ndarray,
                          prev_diff_subject_mask: np.ndarray,
                          subjects: np.ndarray,
                          feature_dimension: str,
                          time_dimension: str,
                          observed_values: Optional[Any] = None,
                          **kwargs) -> Any:

        mean_a0 = self._create_initial_mean_variable() if mean_a0 is None else mean_a0
        sd_aa = self._create_transition_standard_deviation_variable() if sd_aa is None else sd_aa

        if self.share_mean_a0_across_subjects:
            mean_a0 = mean_a0[:, None]  # feature x time = 1 (broadcast across time)
        else:
            mean_a0 = mean_a0[subjects].transpose()  # feature x time

        if self.share_mean_a0_across_dimensions:
            mean_a0 = mean_a0.repeat(self.dimension_size, axis=0)

        if self.share_sd_aa_across_subjects:
            sd_aa = sd_aa[:, None]  # feature x time = 1 (broadcast across time)
        else:
            sd_aa = sd_aa[subjects].transpose()  # feature x time

        if self.share_sd_aa_across_dimensions:
            sd_aa = sd_aa.repeat(self.dimension_size, axis=0)

        logp_params = (mean_a0,
                       sd_aa,
                       coordination,
                       prev_time_same_subject,
                       prev_time_diff_subject,
                       prev_same_subject_mask,
                       prev_diff_subject_mask,
                       np.array(self.self_dependent),
                       *self._get_extra_logp_params(subjects, prev_time_same_subject,
                                                    prev_time_diff_subject)
                       )
        logp_fn = self._get_logp_fn()
        random_fn = self._get_random_fn()
        serial_component = pm.DensityDist(self.uuid, *logp_params,
                                          logp=logp_fn,
                                          random=random_fn,
                                          dims=[feature_dimension, time_dimension],
                                          observed=observed_values)

        return serial_component, mean_a0, sd_aa

    def _get_extra_logp_params(self, subjects_in_time: np.ndarray,
                               prev_time_same_subject: np.ndarray,
                               prev_time_diff_subject: np.ndarray):
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


###################################################################################################
# AUXILIARY CLASSES
###################################################################################################

class SerialComponentSamples(LatentComponentSamples):

    def __init__(self,
                 values: List[np.ndarray],
                 time_steps_in_coordination_scale: List[np.ndarray],
                 subjects: List[np.ndarray],
                 prev_time_same_subject: List[np.ndarray],
                 prev_time_diff_subject: List[np.ndarray]):
        """
        Creates an object to store samples.

        @param values: sampled values of the latent component. For serial components, this will be
        a list of time series of values of different sizes. For non-serial components, this will be
        a tensor as the number of observations in time do not change for different sampled time
        series.
        @param time_steps_in_coordination_scale: indexes to the coordination used to generate the
        sample. If the component is in a different time scale from the time scale used to compute
        coordination, this mapping will tell which value of coordination to map to each sampled
        value of the latent component. For serial components, this will be a list of time series of
        indices of different sizes. For non-serial components, this will be a tensor as the number
        of observations in time do not change for different sampled time series.
        @param subjects: number indicating which subject is associated to the component at every
        time step (e.g. the current speaker for a speech component). In serial components, only one
        user's latent component is observed at a time. This array indicates which user that is.
        @param prev_time_same_subject: time indices indicating the previous observation of the
        latent component produced by the same subject at a given time. For instance, the last time
        when the current speaker talked.
        @param prev_time_diff_subject: similar to the above but it indicates the most recent time
        when the latent component was observed for a different subject.
        """
        super().__init__(values, time_steps_in_coordination_scale)

        self.subjects = subjects
        self.prev_time_same_subject = prev_time_same_subject
        self.prev_time_diff_subject = prev_time_diff_subject

    @property
    def prev_time_same_subject_mask(self):
        return [np.where(x >= 0, 1, 0) for x in self.prev_time_same_subject]

    @property
    def prev_time_diff_subject_mask(self):
        return [np.where(x >= 0, 1, 0) for x in self.prev_time_diff_subject]


###################################################################################################
# AUXILIARY FUNCTIONS
###################################################################################################

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
