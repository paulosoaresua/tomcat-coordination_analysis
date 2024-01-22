from __future__ import annotations

from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import pymc as pm
import pytensor.tensor as ptt
from scipy.stats import norm

from coordination.common.types import TensorTypes
from coordination.common.utils import adjust_dimensions
from coordination.module.constants import (DEFAULT_FIXED_SUBJECT_SEQUENCE_FLAG,
                                           DEFAULT_LATENT_DIMENSION_SIZE,
                                           DEFAULT_LATENT_MEAN_PARAM,
                                           DEFAULT_LATENT_SD_PARAM,
                                           DEFAULT_NUM_SUBJECTS,
                                           DEFAULT_SAMPLING_TIME_SCALE_DENSITY,
                                           DEFAULT_SELF_DEPENDENCY,
                                           DEFAULT_SHARING_ACROSS_DIMENSIONS,
                                           DEFAULT_SHARING_ACROSS_SUBJECTS,
                                           DEFAULT_SUBJECT_REPETITION_FLAG)
from coordination.module.latent_component.gaussian_latent_component import \
    GaussianLatentComponent
from coordination.module.latent_component.latent_component import \
    LatentComponentSamples
from coordination.module.module import ModuleSamples


class SerialGaussianLatentComponent(GaussianLatentComponent):
    """
    This class represents a serial latent component where there's only one subject per observation
    at a time in the component's scale, and subjects are influenced in a pair-wised manner.
    """

    def __init__(
            self,
            uuid: str,
            pymc_model: pm.Model,
            num_subjects: int = DEFAULT_NUM_SUBJECTS,
            dimension_size: int = DEFAULT_LATENT_DIMENSION_SIZE,
            self_dependent: bool = DEFAULT_SELF_DEPENDENCY,
            mean_mean_a0: np.ndarray = DEFAULT_LATENT_MEAN_PARAM,
            sd_mean_a0: np.ndarray = DEFAULT_LATENT_SD_PARAM,
            sd_sd_a: np.ndarray = DEFAULT_LATENT_SD_PARAM,
            share_mean_a0_across_subjects: bool = DEFAULT_SHARING_ACROSS_SUBJECTS,
            share_mean_a0_across_dimensions: bool = DEFAULT_SHARING_ACROSS_DIMENSIONS,
            share_sd_a_across_subjects: bool = DEFAULT_SHARING_ACROSS_SUBJECTS,
            share_sd_a_across_dimensions: bool = DEFAULT_SHARING_ACROSS_DIMENSIONS,
            dimension_names: Optional[List[str]] = None,
            coordination_samples: Optional[ModuleSamples] = None,
            coordination_random_variable: Optional[pm.Distribution] = None,
            latent_component_random_variable: Optional[pm.Distribution] = None,
            mean_a0_random_variable: Optional[pm.Distribution] = None,
            sd_a_random_variable: Optional[pm.Distribution] = None,
            sampling_time_scale_density: float = DEFAULT_SAMPLING_TIME_SCALE_DENSITY,
            allow_sampled_subject_repetition: bool = DEFAULT_SUBJECT_REPETITION_FLAG,
            fix_sampled_subject_sequence: bool = DEFAULT_FIXED_SUBJECT_SEQUENCE_FLAG,
            time_steps_in_coordination_scale: Optional[np.array] = None,
            subject_indices: Optional[np.ndarray] = None,
            prev_time_same_subject: Optional[np.ndarray] = None,
            prev_time_diff_subject: Optional[np.ndarray] = None,
            observed_values: Optional[TensorTypes] = None,
            mean_a0: Optional[Union[float, np.ndarray]] = None,
            sd_a: Optional[Union[float, np.ndarray]] = None,
            posterior_samples: Optional[np.ndarray] = None
    ):
        """
        Creates a serial latent component.

        @param uuid: String uniquely identifying the latent component in the model.
        @param pymc_model: a PyMC model instance where modules are to be created at.
        @param num_subjects: the number of subjects that possess the component.
        @param dimension_size: the number of dimensions in the latent component.
        @param self_dependent: whether the latent variables in the component are tied to the past
            values from the same subject. If False, coordination will blend the previous latent
            value of a different subject with the value of the component at time t = 0 for the
            current subject (the latent component's prior for that subject).
        @param mean_mean_a0: mean of the hyper-prior of mu_a0 (mean of the initial value of the
            latent component).
        @param sd_sd_a: std of the hyper-prior of sigma_a (std of the Gaussian random walk of
        the latent component).
        @param share_mean_a0_across_subjects: whether to use the same mu_a0 for all subjects.
        @param share_mean_a0_across_dimensions: whether to use the same mu_a0 for all dimensions.
        @param share_sd_a_across_subjects: whether to use the same sigma_a for all subjects.
        @param share_sd_a_across_dimensions: whether to use the same sigma_a for all dimensions.
        @param dimension_names: the names of each dimension of the latent component. If not
        informed, this will be filled with numbers 0,1,2 up to dimension_size - 1.
        @param coordination_samples: coordination samples to be used in a call to draw_samples.
            This variable must be set before such a call.
        @param coordination_random_variable: coordination random variable to be used in a call to
            update_pymc_model. This variable must be set before such a call.
        @param latent_component_random_variable: latent component random variable to be used in a
            call to update_pymc_model. If not set, it will be created in such a call.
        @param mean_a0_random_variable: random variable to be used in a call to
            update_pymc_model. If not set, it will be created in such a call.
        @param sd_a_random_variable: random variable to be used in a call to
            update_pymc_model. If not set, it will be created in such a call.
        @param sampling_time_scale_density: a number between 0 and 1 indicating the frequency in
            which we have observations for the latent component. If 1, the latent component is
            observed at every time in the coordination timescale. If 0.5, in average, only half of
            the time. The final number of observations is not a deterministic function of the time
            density, as the density is used to define transition probabilities between a subjects
            and a non-existing subject.
        @param allow_sampled_subject_repetition: whether subsequent observations can come from the
            same subject.
        @param fix_sampled_subject_sequence: whether the sequence of subjects is fixed
            (0,1,2,...,0,1,2...) or randomized.
        @param time_steps_in_coordination_scale: time indexes in the coordination scale for
            each index in the latent component scale.
        @param subject_indices: array of numbers indicating which subject is associated to the
            latent component at every time step (e.g. the current speaker for a speech component).
            In serial components, only one user's latent component is observed at a time. This
            array indicates which user that is. This array contains no gaps. The size of the array
            is the number of observed latent component in time, i.e., latent component time
            indices with an associated subject.
        @param prev_time_same_subject: time indices indicating the previous observation of the
            latent component produced by the same subject at a given time. For instance, the last
            time when the current speaker talked. This variable must be set before a call to
            update_pymc_model.
        @param prev_time_diff_subject: similar to the above but it indicates the most recent time
            when the latent component was observed for a different subject. This variable must be
            set before a call to update_pymc_model.
        @param observed_values: observations for the serial latent component random variable. If
            a value is set, the variable is not latent anymore.
        @param mean_a0: initial value of the latent component. It needs to be given for sampling
            but not for inference if it needs to be inferred. If not provided now, it can be set
            later via the module parameters variable.
        @param sd_a: standard deviation of the latent component Gaussian random walk. It needs to
            be given for sampling but not for inference if it needs to be inferred. If not
            provided now, it can be set later via the module parameters variable.
        @param posterior_samples: samples from the posterior to use during a call to draw_samples.
            This is useful to do predictive checks by sampling data in the future.
        """
        super().__init__(
            uuid=uuid,
            pymc_model=pymc_model,
            num_subjects=num_subjects,
            dimension_size=dimension_size,
            self_dependent=self_dependent,
            mean_mean_a0=mean_mean_a0,
            sd_mean_a0=sd_mean_a0,
            sd_sd_a=sd_sd_a,
            share_mean_a0_across_subjects=share_mean_a0_across_subjects,
            share_mean_a0_across_dimensions=share_mean_a0_across_dimensions,
            share_sd_a_across_subjects=share_sd_a_across_subjects,
            share_sd_a_across_dimensions=share_sd_a_across_dimensions,
            dimension_names=dimension_names,
            coordination_samples=coordination_samples,
            coordination_random_variable=coordination_random_variable,
            latent_component_random_variable=latent_component_random_variable,
            mean_a0_random_variable=mean_a0_random_variable,
            sd_a_random_variable=sd_a_random_variable,
            time_steps_in_coordination_scale=time_steps_in_coordination_scale,
            observed_values=observed_values,
            mean_a0=mean_a0,
            sd_a=sd_a,
        )

        self.sampling_time_scale_density = sampling_time_scale_density
        self.allow_sampled_subject_repetition = allow_sampled_subject_repetition
        self.fix_sampled_subject_sequence = fix_sampled_subject_sequence
        self.subject_indices = subject_indices
        self.prev_time_same_subject = prev_time_same_subject
        self.prev_time_diff_subject = prev_time_diff_subject
        self.posterior_samples = posterior_samples

    def draw_samples(
            self, seed: Optional[int], num_series: int
    ) -> SerialGaussianLatentComponentSamples:
        """
        Draws latent component samples using ancestral sampling and pairwise blending with
        coordination and different subjects.

        @param seed: random seed for reproducibility.
        @param num_series: how many series of samples to generate.
        @return: latent component samples for each coordination series.
        """
        super().draw_samples(seed, num_series)

        if (
                self.sampling_time_scale_density <= 0
                or self.sampling_time_scale_density > 1
        ):
            raise ValueError(
                f"The time scale density ({self.sampling_time_scale_density}) "
                f"must be a float number in the range (0,1]."
            )

        dim_mean_a_subjects = (
            1 if self.share_mean_a0_across_subjects else self.num_subjects
        )
        dim_mean_a_dimensions = (
            1 if self.share_mean_a0_across_dimensions else self.dimension_size
        )
        dim_sd_a_subjects = 1 if self.share_sd_a_across_subjects else self.num_subjects
        dim_sd_a_dimensions = (
            1 if self.share_sd_a_across_dimensions else self.dimension_size
        )

        # Adjust dimensions according to parameter sharing specification
        if (isinstance(self.parameters.mean_a0.value,
                       np.ndarray) and self.parameters.mean_a0.value.ndim == 3):
            # A different value per series. We expect it's already in the correct dimensions.
            mean_a0 = self.parameters.mean_a0.value
        else:
            mean_a0 = adjust_dimensions(
                self.parameters.mean_a0.value,
                num_rows=dim_mean_a_subjects,
                num_cols=dim_mean_a_dimensions,
            )
            if self.share_mean_a0_across_subjects:
                mean_a0 = mean_a0[None, :].repeat(self.num_subjects, axis=0)

            mean_a0 = mean_a0[None, :].repeat(num_series, axis=0)

        if (isinstance(self.parameters.sd_a.value,
                       np.ndarray) and self.parameters.sd_a.value.ndim == 3):
            # A different value per series. We expect it's already in the correct dimensions.
            sd_a = self.parameters.sd_a.value
        else:
            sd_a = adjust_dimensions(
                self.parameters.sd_a.value,
                num_rows=dim_sd_a_subjects,
                num_cols=dim_sd_a_dimensions,
            )
            if self.share_sd_a_across_subjects:
                sd_a = sd_a[None, :].repeat(self.num_subjects, axis=0)

            sd_a = sd_a[None, :].repeat(num_series, axis=0)

        sampled_values = []
        subject_indices = (
            [] if self.posterior_samples is None else [self.subject_indices] * num_series)
        time_steps_in_coordination_scale = (
            [] if self.posterior_samples is None else [self.time_steps_in_coordination_scale] * num_series)
        prev_time_same_subject = (
            [] if self.posterior_samples is None else [self.prev_time_same_subject] * num_series)
        prev_time_diff_subject = (
            [] if self.posterior_samples is None else [self.prev_time_diff_subject] * num_series)
        for s in range(num_series):
            if self.posterior_samples is None:
                sparse_subjects = self._draw_random_subjects(num_series)
                subject_indices.append(
                    np.array([s for s in sparse_subjects[s] if s >= 0], dtype=int)
                )

                time_steps_in_coordination_scale.append(
                    np.array(
                        [t for t, s in enumerate(sparse_subjects[s]) if s >= 0], dtype=int
                    )
                )

                num_time_steps_in_cpn_scale = len(time_steps_in_coordination_scale[s])

                prev_time_same_subject.append(
                    np.full(shape=num_time_steps_in_cpn_scale, fill_value=-1, dtype=int)
                )
                prev_time_diff_subject.append(
                    np.full(shape=num_time_steps_in_cpn_scale, fill_value=-1, dtype=int)
                )

                # Fill dependencies
                prev_time_per_subject = {}
                for t in range(num_time_steps_in_cpn_scale):
                    prev_time_same_subject[s][t] = prev_time_per_subject.get(
                        subject_indices[s][t], -1
                    )

                    for subject, time in prev_time_per_subject.items():
                        if subject == subject_indices[s][t]:
                            continue

                        # Most recent time from a different subject
                        prev_time_diff_subject[s][t] = (
                            time
                            if prev_time_diff_subject[s][t] == -1
                            else max(prev_time_diff_subject[s][t], time)
                        )

                    prev_time_per_subject[subject_indices[s][t]] = t

            values = self._draw_from_system_dynamics(
                coordination_sampled_series=self.coordination_samples.values[s],
                time_steps_in_coordination_scale=
                time_steps_in_coordination_scale[s],
                subjects_in_time=subject_indices[s],
                prev_time_same_subject=prev_time_same_subject[s],
                prev_time_diff_subject=prev_time_diff_subject[s],
                mean_a0=mean_a0[s],
                sd_a=sd_a[s],
                init_values=None if self.posterior_samples is None else self.posterior_samples[s]
            )
            sampled_values.append(values)

        return SerialGaussianLatentComponentSamples(
            values=sampled_values,
            time_steps_in_coordination_scale=time_steps_in_coordination_scale,
            subject_indices=subject_indices,
            prev_time_same_subject=prev_time_same_subject,
            prev_time_diff_subject=prev_time_diff_subject,
        )

    def _draw_random_subjects(self, num_series: int) -> np.ndarray:
        """
        Draws random sequences of subjects over time. We define subject -1 as "No Subject", meaning
        the latent component is not observed when that subject is the one sampled in a given time
        step.

        @param num_series: how many series of samples to generate.
        @return: a matrix of sampled subject series.
        """

        if self.allow_sampled_subject_repetition:
            # We allow the same subject to appear in subsequent observations
            if self.fix_sampled_subject_sequence:
                transition_matrix = np.zeros(
                    shape=(self.num_subjects + 1, self.num_subjects + 1)
                )
                transition_matrix[:, 0] = 1 - self.sampling_time_scale_density
                transition_matrix[0, 1] = self.sampling_time_scale_density
                transition_matrix[-1, 1] = self.sampling_time_scale_density / 2
                transition_matrix[-1, -1] = self.sampling_time_scale_density / 2
                for s1 in range(1, self.num_subjects):
                    for s2 in range(1, self.num_subjects + 1):
                        if s1 == s2 or s2 == s1 + 1:
                            transition_matrix[s1, s2] = (
                                    self.sampling_time_scale_density / 2
                            )

            else:
                transition_matrix = np.full(
                    shape=(self.num_subjects + 1, self.num_subjects + 1),
                    fill_value=self.sampling_time_scale_density / self.num_subjects,
                )
                transition_matrix[:, 0] = 1 - self.sampling_time_scale_density
        else:
            if self.fix_sampled_subject_sequence:
                transition_matrix = np.zeros(
                    shape=(self.num_subjects + 1, self.num_subjects + 1)
                )
                transition_matrix[:, 0] = 1 - self.sampling_time_scale_density
                transition_matrix[:-1, 1:] = (
                        np.eye(self.num_subjects) * self.sampling_time_scale_density
                )
                transition_matrix[-1, 1] = self.sampling_time_scale_density
            else:
                transition_matrix = np.full(
                    shape=(self.num_subjects + 1, self.num_subjects + 1),
                    fill_value=self.sampling_time_scale_density
                               / (self.num_subjects - 1),
                )
                transition_matrix[0, 1:] = (
                        self.sampling_time_scale_density / self.num_subjects
                )
                transition_matrix = transition_matrix * (
                        1 - np.eye(self.num_subjects + 1)
                )
                transition_matrix[:, 0] = 1 - self.sampling_time_scale_density

        initial_prob = transition_matrix[0]
        subjects = np.zeros(
            (
                self.coordination_samples.num_series,
                self.coordination_samples.num_time_steps,
            ),
            dtype=int,
        )

        for t in range(self.coordination_samples.num_time_steps):
            if t == 0:
                subjects[:, t] = np.random.choice(
                    self.num_subjects + 1, num_series, p=initial_prob
                )
            else:
                probs = transition_matrix[subjects[:, t - 1]]
                cum_prob = np.cumsum(probs, axis=-1)
                u = np.random.uniform(size=(num_series, 1))
                subjects[:, t] = np.argmax(u < cum_prob, axis=-1)

        # Map 0 to -1
        subjects -= 1
        return subjects

    def _draw_from_system_dynamics(
            self,
            coordination_sampled_series: np.ndarray,
            time_steps_in_coordination_scale: np.ndarray,
            subjects_in_time: np.ndarray,
            prev_time_same_subject: np.ndarray,
            prev_time_diff_subject: np.ndarray,
            mean_a0: np.ndarray,
            sd_a: np.ndarray,
            init_values: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Draws values from the system dynamics. The default serialized component generates samples
        by following a Gaussian random walk with mean defined by pairwise blended values from
        different subjects according to the coordination levels over time. Child classes can
        implement their own dynamics, like spring-mass-damping systems for instance.

        @param coordination_sampled_series: sampled values of coordination series.
        @param time_steps_in_coordination_scale: an array of indices representing time steps in the
        coordination scale that match those of the component's scale.
        @param subjects_in_time: an array indicating which subject is responsible for a latent
        component observation at a time.
        @param prev_time_same_subject: an array containing indices to most recent previous times
        the component from the same subject was observed in the component's timescale.
        @param prev_time_diff_subject: an array containing indices to most recent previous times
        the component from a different subject was observed in the component's timescale.
        @param mean_a0: initial mean of the latent component.
        @param sd_a: standard deviation of the Gaussian transition distribution.
        @param init_values: initial values if the series was pre-sampled up to some time step.

        @return: sampled values.
        """

        num_time_steps = len(time_steps_in_coordination_scale)
        values = np.zeros((self.dimension_size, num_time_steps))
        t0 = 0 if init_values is None else init_values.shape[-1]
        if init_values is not None:
            values[..., :t0] = init_values

        for t in range(t0, num_time_steps):
            # The way we index subjects change depending on the sharing options. If shared, the
            # parameters will have a single dimension across subjects, so we can index by 0,
            # otherwise we use the subject number to index the correct parameter for that subject.
            if self.share_mean_a0_across_subjects:
                subject_idx_mean_a0 = 0
            else:
                subject_idx_mean_a0 = subjects_in_time[t]

            if self.share_sd_a_across_subjects:
                subject_idx_sd_a = 0
            else:
                subject_idx_sd_a = subjects_in_time[t]

            sd = sd_a[subject_idx_sd_a]

            if prev_time_same_subject[t] < 0:
                # This is the first time we see an observation for the subject. We then sample
                # from its prior. This doesn't happen only when t=0 because the first observation
                # of a subject can be later in the future. So, t=0 is the initial state of one of
                # the subjects, but not all of them.

                mean = mean_a0[subject_idx_mean_a0]
                values[:, t] = norm(loc=mean, scale=sd).rvs(size=self.dimension_size)
            else:
                c = coordination_sampled_series[time_steps_in_coordination_scale[t]]

                if self.self_dependent:
                    # When there's self dependency, the component either depends on the previous
                    # value of another subject or the previous value of the same subject.
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

    def create_random_variables(self):
        """
        Creates parameters and serial latent component variables in a PyMC model.

        @raise ValueError: if either subject_indices, time_steps_in_coordination_scale,
            prev_time_same_subject, or prev_time_diff_subject are undefined.
        """
        super().create_random_variables()

        # Adjust dimensions for proper indexing and broadcast in the log_prob function.
        if self.share_mean_a0_across_subjects:
            # dimension x time = 1 (broadcast across time)
            mean_a0 = self.mean_a0_random_variable[0, :, None]
        else:
            # dimension x time
            mean_a0 = self.mean_a0_random_variable[self.subject_indices].transpose()

        if self.share_mean_a0_across_dimensions:
            mean_a0 = mean_a0.repeat(self.dimension_size, axis=0)

        if self.share_sd_a_across_subjects:
            # dimension x time = 1 (broadcast across time)
            sd_a = self.sd_a_random_variable[0, :, None]
        else:
            # dimension x time
            sd_a = self.sd_a_random_variable[self.subject_indices].transpose()

        if self.share_sd_a_across_dimensions:
            sd_a = sd_a.repeat(self.dimension_size, axis=0)

        if self.latent_component_random_variable is not None:
            return

        if self.subject_indices is None:
            raise ValueError("subject_indices is undefined.")

        if self.time_steps_in_coordination_scale is None:
            raise ValueError("time_steps_in_coordination_scale is undefined.")

        if self.prev_time_same_subject is None:
            raise ValueError("prev_time_same_subject is undefined.")

        if self.prev_time_diff_subject is None:
            raise ValueError("prev_time_diff_subject is undefined.")

        # Mask with 1 for time steps where there is observation for a subject (subject index >= 0)
        prev_same_subject_mask = np.array(
            [np.where(x >= 0, 1, 0) for x in self.prev_time_same_subject]
        )
        prev_diff_subject_mask = np.array(
            [np.where(x >= 0, 1, 0) for x in self.prev_time_diff_subject]
        )

        log_prob_params = (
            mean_a0,
            sd_a,
            self.coordination_random_variable[self.time_steps_in_coordination_scale],
            self.prev_time_same_subject,
            self.prev_time_diff_subject,
            prev_same_subject_mask,
            prev_diff_subject_mask,
            np.array(self.self_dependent),
            *self._get_extra_logp_params(),
        )

        with self.pymc_model:
            self.latent_component_random_variable = pm.DensityDist(
                self.uuid,
                *log_prob_params,
                logp=self._get_log_prob_fn(),
                random=self._get_random_fn(),
                dims=[self.dimension_axis_name, self.time_axis_name],
                observed=self.observed_values,
            )

    def _add_coordinates(self):
        """
        Adds relevant coordinates to the model. Overrides superclass.

         @raise ValueError: if either subject_indices, time_steps_in_coordination_scale are
            undefined.
        """
        super()._add_coordinates()

        if self.subject_indices is None:
            raise ValueError("subject_indices is undefined.")

        if self.time_steps_in_coordination_scale is None:
            raise ValueError("time_steps_in_coordination_scale is undefined.")

        # Add information about which subject is associated with each timestep in the time
        # coordinate.
        self.pymc_model.add_coord(
            name=self.time_axis_name,
            values=[
                f"{sub}#{time}"
                for sub, time in zip(
                    self.subject_indices, self.time_steps_in_coordination_scale
                )
            ],
        )

    def _get_extra_logp_params(self) -> Tuple[Union[TensorTypes, pm.Distribution], ...]:
        """
        Gets extra parameters to be passed to the log_prob and random functions.
        """
        return ()

    def _get_log_prob_fn(self) -> Callable:
        """
        Gets a reference to a log_prob function.
        """
        return log_prob

    def _get_random_fn(self) -> Callable:
        """
        Gets a reference to a random function for prior predictive checks.
        """
        return random


###################################################################################################
# AUXILIARY CLASSES
###################################################################################################


class SerialGaussianLatentComponentSamples(LatentComponentSamples):
    def __init__(
            self,
            values: List[np.ndarray],
            time_steps_in_coordination_scale: List[np.ndarray],
            subject_indices: List[np.ndarray],
            prev_time_same_subject: List[np.ndarray],
            prev_time_diff_subject: List[np.ndarray],
    ):
        """
        Creates an object to store samples.

        @param values: sampled values of the latent component. This is a list of time series of
        values of different sizes because each sampled series may have a different sparsity level.
        @param time_steps_in_coordination_scale: indexes to the coordination used to generate the
        sample. If the component is in a different timescale from the timescale used to compute
        coordination, this mapping will tell which value of coordination to map to each sampled
        value of the latent component.
        @param subject_indices: series indicating which subject is associated to the component at
            every time step (e.g. the current speaker for a speech component). In serial
            components, only one user's latent component is observed at a time. This array
            indicates which user that is.
        @param prev_time_same_subject: time indices indicating the previous observation of the
        latent component produced by the same subject at a given time. For instance, the last time
        when the current speaker talked.
        @param prev_time_diff_subject: similar to the above but it indicates the most recent time
        when the latent component was observed for a different subject.
        """
        super().__init__(values, time_steps_in_coordination_scale)

        self.subject_indices = subject_indices
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


def log_prob(
        sample: ptt.TensorVariable,
        initial_mean: ptt.TensorVariable,
        sigma: ptt.TensorVariable,
        coordination: ptt.TensorVariable,
        prev_time_same_subject: ptt.TensorConstant,
        prev_time_diff_subject: ptt.TensorConstant,
        prev_same_subject_mask: ptt.TensorConstant,
        prev_diff_subject_mask: ptt.TensorConstant,
        self_dependent: ptt.TensorConstant,
) -> float:
    """
    Computes the log-probability function of a sample.

    Legend:
    D: number of dimensions
    T: number of time steps

    @param sample: (dimension x time) a single samples series.
    @param initial_mean: (dimension x time) a series of mean at t0. At each time the mean is
        associated with the subject at that time. The initial mean is only used the first time the
        user speaks, but we repeat the values here over time for uniform vector operations (e.g.,
        we can multiply this with other tensors) and we fix the behavior with mask tensors.
    @param sigma: (dimension x time) a series of standard deviations. At each time the standard
        deviation is associated with the subject at that time.
    @param coordination: (time) a series of coordination values.
    @param prev_time_same_subject: (time) a series of time steps pointing to the previous time step
        associated with the same subject. For instance, prev_time_same_subject[t] points to the
        most recent time step where the subject at time t had an observation. If there's no such a
        time, prev_time_same_subject[t] will be -1.
    @param prev_time_diff_subject: (time)  a series of time steps pointing to the previous time
        step associated with a different subject. For instance, prev_time_diff_subject[t] points
        to the most recent time step where a different subject than the one at time t had an
        observation. If there's no such a time, prev_time_diff_subject[t] will be -1.
    @param prev_same_subject_mask: (time) a binary mask with 0 whenever prev_time_same_subject
        is -1.
    @param prev_diff_subject_mask: (time) a binary mask with 0 whenever prev_time_diff_subject
        is -1.
    @param self_dependent: a boolean indicating whether subjects depend on their previous values.
    @return: log-probability of the sample.
    """

    # We use 'prev_time_diff_subject' as meta-data to get the values from partners of the subjects
    # in each time step. We reshape to guarantee we don't create dimensions with unknown size in
    # case the first dimension of the sample component is one.
    prev_other = sample[..., prev_time_diff_subject].reshape(
        sample.shape
    )  # (dimension x time)

    # We use this binary mask to zero out entries with no observations from partners. We create an
    # extra dimension for broadcasting across the dimension axis (axis 0).
    mask_other = prev_diff_subject_mask[None, :]  # (1 x T)

    # We create an extra dimension for broadcasting across the dimension axis (axis 0).
    c = coordination[None, :]  # (1 x T)

    if self_dependent.eval():
        # The component's value for a subject depends on previous value of the same subject.
        prev_same = sample[..., prev_time_same_subject].reshape(sample.shape)  # (D x T)

        # We use this binary mask to zero out entries with no previous values of the same subjects.
        # We use this to determine the time steps that belong to the initial values of the
        # component. Each subject will have their initial value in a different time step hence
        # we cannot just use t=0.
        # We create an extra dimension for broadcasting across the dimension axis (axis 0).
        mask_same = prev_same_subject_mask[None, :]  # (1 x T)
        blended_mean = prev_other * c * mask_other + (1 - c * mask_other) * (
                prev_same * mask_same + (1 - mask_same) * initial_mean
        )  # (D x T)
    else:
        # The component's value for a subject doesn't depend on previous value of the same subject.
        # At every time step, the value from other subjects is blended with a fixed value given
        # by the component's initial means associated with the subjects over time.
        # (D x T)
        blended_mean = prev_other * c * mask_other + (1 - c * mask_other) * initial_mean

    return pm.logp(
        rv=pm.Normal.dist(mu=blended_mean, sigma=sigma, shape=prev_other.shape),
        value=sample,
    ).sum()


def random(
        initial_mean: np.ndarray,
        sigma: np.ndarray,
        coordination: np.ndarray,
        prev_time_same_subject: np.ndarray,
        prev_time_diff_subject: np.ndarray,
        prev_same_subject_mask: np.ndarray,
        prev_diff_subject_mask: np.ndarray,
        self_dependent: bool,
        rng: Optional[np.random.Generator] = None,
        size: Optional[Tuple[int]] = None,
) -> np.ndarray:
    """
    Generates samples from of a serial latent component for prior predictive checks.

    Legend:
    D: number of dimensions
    T: number of time steps

    @param initial_mean: (dimension x time) a series of mean at t0. At each time the mean is
        associated with the subject at that time. The initial mean is only used the first time the
        user speaks, but we repeat the values here over time for uniform vector operations (e.g.,
        we can multiply this with other tensors) and we fix the behavior with mask tensors.
    @param sigma: (dimension x time) a series of standard deviations. At each time the standard
        deviation is associated with the subject at that time.
    @param coordination: (time) a series of coordination values.
    @param prev_time_same_subject: (time) a series of time steps pointing to the previous time step
        associated with the same subject. For instance, prev_time_same_subject[t] points to the
        most recent time step where the subject at time t had an observation. If there's no such a
        time, prev_time_same_subject[t] will be -1.
    @param prev_time_diff_subject: (time)  a series of time steps pointing to the previous time
        step associated with a different subject. For instance, prev_time_diff_subject[t] points
        to the most recent time step where a different subject than the one at time t had an
        observation. If there's no such a time, prev_time_diff_subject[t] will be -1.
    @param prev_same_subject_mask: (time) a binary mask with 0 whenever prev_time_same_subject
        is -1.
    @param prev_diff_subject_mask: (time) a binary mask with 0 whenever prev_time_diff_subject
        is -1.
    @param self_dependent: a boolean indicating whether subjects depend on their previous values.
    @param rng: random number generator.
    @param size: size of the sample.

    @return: a serial latent component sample.
    """

    # TODO: Unify this with the class sampling method.

    T = coordination.shape[-1]

    sample = np.zeros(size)

    mean_0 = initial_mean if initial_mean.ndim == 1 else initial_mean[..., 0]
    sd_0 = sigma if sigma.ndim == 1 else sigma[..., 0]

    sample[..., 0] = rng.normal(loc=mean_0, scale=sd_0)

    for t in np.arange(1, T):
        prev_other = sample[..., prev_time_diff_subject[t]]  # D

        # Previous sample from the same individual
        if self_dependent and prev_same_subject_mask[t] == 1:
            prev_same = sample[..., prev_time_same_subject[t]]
        else:
            # When there's no self-dependency, the transition distribution is a blending between
            # the previous value from another individual, and a fixed mean.
            if initial_mean.shape[1] == 1:
                prev_same = initial_mean[..., 0]
            else:
                prev_same = initial_mean[..., t]

        mean = (prev_other - prev_same) * coordination[t] * prev_diff_subject_mask[
            t
        ] + prev_same

        if sigma.shape[1] == 1:
            # Parameter sharing across subjects
            transition_sample = rng.normal(loc=mean, scale=sigma[..., 0])
        else:
            transition_sample = rng.normal(loc=mean, scale=sigma[..., t])

        sample[..., t] = transition_sample

    return sample
