from __future__ import annotations

from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import pymc as pm
import pytensor.tensor as ptt
from scipy.stats import norm
import pytensor as pt

from coordination.common.types import TensorTypes
from coordination.module.constants import (DEFAULT_LATENT_MEAN_PARAM,
                                           DEFAULT_LATENT_SD_PARAM,
                                           DEFAULT_NUM_SUBJECTS,
                                           DEFAULT_SAMPLING_RELATIVE_FREQUENCY,
                                           DEFAULT_SHARING_ACROSS_DIMENSIONS,
                                           DEFAULT_SHARING_ACROSS_SUBJECTS)
from coordination.module.latent_component.non_serial_gaussian_latent_component import \
    NonSerialGaussianLatentComponent
from coordination.module.module import ModuleSamples


class CommonCauseGaussian2D(NonSerialGaussianLatentComponent):
    """
    This class represents a 2D common cause time series variable with position and velocity. The
    series dynamics is given by Newtonian equations of uniform movement and Gaussian transitions.
    """

    def __init__(
            self,
            uuid: str,
            pymc_model: pm.Model,
            mean_mean_cc0: np.ndarray = DEFAULT_LATENT_MEAN_PARAM,
            sd_mean_cc0: np.ndarray = DEFAULT_LATENT_SD_PARAM,
            sd_sd_cc: np.ndarray = DEFAULT_LATENT_SD_PARAM,
            share_mean_cc0_across_dimensions: bool = DEFAULT_SHARING_ACROSS_DIMENSIONS,
            share_sd_cc_across_dimensions: bool = DEFAULT_SHARING_ACROSS_DIMENSIONS,
            common_cause_random_variable: Optional[pm.Distribution] = None,
            mean_cc0_random_variable: Optional[pm.Distribution] = None,
            sd_cc_random_variable: Optional[pm.Distribution] = None,
            time_steps_in_coordination_scale: Optional[np.array] = None,
            observed_values: Optional[TensorTypes] = None,
            mean_cc0: Optional[Union[float, np.ndarray]] = None,
            sd_cc: Optional[Union[float, np.ndarray]] = None,
            initial_samples: Optional[np.ndarray] = None
    ):
        """
        Creates a non-serial 2D Gaussian latent component.

        @param uuid: String uniquely identifying the latent component in the model.
        @param pymc_model: a PyMC model instance where modules are to be created at.
        @param mean_mean_cc0: mean of the hyper-prior of mu_cc0 (mean of the initial value of the
            common cause).
        @param sd_sd_cc: std of the hyper-prior of sigma_a (std of the Gaussian random walk of
        the common cause).
        @param share_mean_cc0_across_dimensions: whether to use the same mu_cc0 for all dimensions.
        @param share_sd_cc_across_dimensions: whether to use the same sigma_cc for all dimensions.
        @param common_cause_random_variable: common cause random variable to be used in a call to
            update_pymc_model. This variable must be set before such a call.
        @param mean_cc0_random_variable: random variable to be used in a call to
            update_pymc_model. If not set, it will be created in such a call.
        @param sd_cc_random_variable: random variable to be used in a call to
            update_pymc_model. If not set, it will be created in such a call.
        @param time_steps_in_coordination_scale: time indexes in the coordination scale for
            each index in the common cause scale.
        @param observed_values: observations for the serial latent component random variable. If
            a value is set, the variable is not latent anymore.
        @param mean_cc0: initial value of the common cause. It needs to be given for
            sampling but not for inference if it needs to be inferred. If not provided now, it can
            be set later via the module parameters variable.
        @param sd_cc: standard deviation of the common cause Gaussian random walk. It needs to
            be given for sampling but not for inference if it needs to be inferred. If not
            provided now, it can be set later via the module parameters variable.
        @param initial_samples: samples from the posterior to use during a call to draw_samples.
            This is useful to do predictive checks by sampling data in the future.
        """
        super().__init__(
            uuid=uuid,
            pymc_model=pymc_model,
            num_subjects=1,  # the common cause itself
            dimension_size=2,  # position and velocity
            self_dependent=True,
            mean_mean_a0=mean_mean_cc0,
            sd_mean_a0=sd_mean_cc0,
            sd_sd_a=sd_sd_cc,
            share_mean_a0_across_subjects=False,
            share_mean_a0_across_dimensions=share_mean_cc0_across_dimensions,
            share_sd_a_across_dimensions=share_sd_cc_across_dimensions,
            dimension_names=["position", "speed"],
            coordination_samples=None,  # Coordination is not used directly in the common cause
            coordination_random_variable=None,
            latent_component_random_variable=common_cause_random_variable,
            mean_a0_random_variable=mean_cc0_random_variable,
            sd_a_random_variable=sd_cc_random_variable,
            time_steps_in_coordination_scale=time_steps_in_coordination_scale,
            observed_values=observed_values,
            mean_a0=mean_cc0,
            sd_a=sd_cc,
            initial_samples=initial_samples,
            asymmetric_coordination=False,
            single_chain=False
        )

    def _draw_from_system_dynamics(
            self,
            sampled_coordination: np.ndarray,
            time_steps_in_coordination_scale: np.ndarray,
            mean_a0: np.ndarray,  # mean_a0 = mean_cc0
            sd_a: np.ndarray,  # sd_a = sd_cc
            init_values: Optional[np.ndarray] = None,
            sampled_common_cause: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Draws values with the following updating equations for the state of the common cause:

        S(t) = S(t-1) + V(t-1)dt
        V(t) = V(t-1),

        where, "S" is position and "V" velocity. These are Newton's equations of uniform movement.

        @param sampled_coordination: sampled values of coordination (all series included).
        @param time_steps_in_coordination_scale: an array of indices representing time steps in the
        coordination scale that match those of the common cause's scale.
        @param mean_a0: initial mean of the common cause.
        @param sd_a: standard deviation of the Gaussian transition distribution.

        @return: sampled values.
        """
        num_series = sampled_coordination.shape[0]
        num_time_steps = len(time_steps_in_coordination_scale)

        # shape = (N, 1 subject = common cause itself, position and velocity, T)
        values = np.zeros((num_series, 1, self.dimension_size, num_time_steps))

        # This function can be called to continue sampling from a specific time step. In that case,
        # the initial time step is not t = 0.
        t0 = 0 if init_values is None else init_values.shape[-1]
        if init_values is not None:
            values[..., :t0] = init_values

        for t in range(t0, num_time_steps):
            if t == 0:
                values[..., 0] = norm(loc=mean_a0, scale=sd_a).rvs(
                    size=(num_series, 1, self.dimension_size))
            else:
                previous = values[..., t - 1]  # N x 1 x D
                upd_matrix = np.array([[1.0, 1.0], [0.0, 1.0]])[None, :].repeat(num_series, axis=0)
                mean_next_state = np.einsum("kij,klj->kli", upd_matrix, previous)

                values[..., t] = norm(loc=mean_next_state, scale=sd_a[None, :]).rvs()

        return values  # N x 1 x D

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
# AUXILIARY FUNCTIONS
###################################################################################################

def log_prob(
        sample: ptt.TensorVariable,
        initial_mean: ptt.TensorVariable,
        sigma: ptt.TensorVariable,
        coordination: ptt.TensorVariable = None,
        self_dependent: ptt.TensorConstant = None,
        symmetry_mask: int = 1,
) -> ptt.TensorVariable:
    """
    Computes the log-probability function of a common cause sample.

    Legend:
    D: number of dimensions
    S: number of subjects
    T: number of time steps

    @param sample: (1 x 2 x time) a single samples series.
    @param initial_mean: (1 x 2) mean at t0 for position and velocity.
    @param sigma: (1 x 2) standard deviation for position and velocity transitions.
    @param coordination: (time) a series of coordination values. This is not used to calculate
        the log probability of the common cause.
    @param self_dependent: a boolean indicating whether subjects depend on their previous values.
        This is not used to calculate the log probability of the common cause since the common
        cause is always self-dependent.
    @param symmetry_mask: -1 if coordination is asymmetric, 1 otherwise. This is not used to
        calculate the log probability of the common cause.
    @return: log-probability of the sample.
    """

    S = sample.shape[0]  # This should be 1 since the common cause is shared among all subjects
    D = sample.shape[1]  # This should be 2: position and velocity
    T = sample.shape[2]

    # log-probability at the initial time step
    total_logp = pm.logp(
        pm.Normal.dist(mu=initial_mean, sigma=sigma, shape=(S, D)), sample[..., 0]
    ).sum()

    previous = sample[..., :-1]  # 1 x 2 x T-1
    upd_matrix = ptt.as_tensor(np.array([[[1.0, 1.0], [0.0, 1.0]]])).repeat(T-1, axis=0)
    mean_next_state = ptt.batched_tensordot(upd_matrix, previous.T, axes=[(2,), (1,)]).T

    # Match the dimensions of the standard deviation with that of the next state mean by adding
    # another dimension for time.
    sd = sigma[:, :, None]

    # Index samples starting from the second index (i = 1) so that we can effectively compare
    # current values against previous ones.
    total_logp += pm.logp(
        pm.Normal.dist(mu=mean_next_state, sigma=sd, shape=mean_next_state.shape),
        sample[..., 1:],
    ).sum()

    return total_logp


def random(
        initial_mean: np.ndarray,
        sigma: np.ndarray,
        coordination: np.ndarray,
        self_dependent: bool,
        symmetry_mask: int,
        rng: Optional[np.random.Generator] = None,
        size: Optional[Tuple[int]] = None,
) -> np.ndarray:
    """
    Generates samples from of the common cause for prior predictive checks.

    Legend:
    D: number of dimensions
    S: number of subjects
    T: number of time steps

    @param initial_mean: (1 x 2) mean at t0 for position and velocity.
    @param sigma: (1 x 2) standard deviation for position and velocity transitions.
    @param coordination: (time) a series of coordination values. This is not used to calculate
        the log probability of the common cause.
    @param self_dependent: a boolean indicating whether subjects depend on their previous values.
        This is not used to calculate the log probability of the common cause since the common
        cause is always self-dependent.
    @param symmetry_mask: -1 if coordination is asymmetric, 1 otherwise. This is not used to
        calculate the log probability of the common cause.
    @param rng: random number generator.
    @param size: size of the sample.

    @return: a serial latent component sample.
    """

    # TODO: Unify this with the class sampling method.

    T = coordination.shape[-1]
    S = initial_mean.shape[0]

    sample = np.zeros(size)

    # Sample from prior in the initial time step
    sample[..., 0] = rng.normal(loc=initial_mean, scale=sigma, size=size[:-1])

    for t in np.arange(1, T):
        previous = sample[..., t - 1]
        upd_matrix = np.array([[1, 1], [0, 1]])
        mean_next_state = np.einsum("ij,lj->li", upd_matrix, previous)

        sample[..., t] = rng.normal(loc=mean_next_state, scale=sigma)

    return sample
