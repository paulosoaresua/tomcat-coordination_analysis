from typing import Any, List, Optional, Tuple

from functools import partial
import numpy as np
import pymc as pm
import pytensor.tensor as ptt
from scipy.stats import norm

from coordination.common.utils import set_random_seed
from coordination.model.parametrization import Parameter, HalfNormalParameterPrior, NormalParameterPrior


def mixture_logp(sample: Any,
                 initial_mean: Any,
                 sigma: Any,
                 coordination: Any,
                 self_dependent: ptt.TensorConstant):
    """
    This function computes the log-probability of a mixture component. We use the following definition in the
    comments below:

    s: number of subjects
    d: number of dimensions/features of the component
    T: number of time steps in the component's scale
    """

    N = sample.shape[0]
    D = sample.shape[1]

    # logp at the initial time step
    total_logp = pm.logp(pm.Normal.dist(mu=initial_mean, sigma=sigma, shape=(N, D)), sample[..., 0]).sum()

    # Contains the sum of previous values of other subjects for each subject scaled by 1/(s-1).
    # We discard the last value as that is not a previous value of any other.
    sum_matrix_others = (ptt.ones((N, N)) - ptt.eye(N)) / (N - 1)
    prev_others = ptt.tensordot(sum_matrix_others, sample, axes=(1, 0))[..., :-1]

    if self_dependent.eval():
        # The component's value for a subject depends on its previous value for the same subject.
        prev_same = sample[..., :-1]
    else:
        # The component's value for a subject does not depend on its previous value for the same subject.
        # At every time step, the value from others is blended with a fixed value given by the component's initial
        # mean.
        prev_same = initial_mean[:, :, None]

    # Coordination does not affect the component in the first time step because the subjects have no previous
    # dependencies at that time.
    c = coordination[None, None, 1:]  # 1 x 1 x t-1

    blended_mean = (prev_others - prev_same) * c + prev_same

    # Match the dimensions of the standard deviation with that of the blended mean
    sd = sigma[:, :, None]

    # Index samples starting from the second index (i = 1) so that we can effectively compare current values against
    # previous ones (prev_others and prev_same).
    total_logp += pm.logp(pm.Normal.dist(mu=blended_mean, sigma=sd, shape=blended_mean.shape), sample[..., 1:]).sum()

    return total_logp


def mixture_random(initial_mean: np.ndarray,
                   sigma: np.ndarray,
                   coordination: np.ndarray,
                   self_dependent: bool,
                   num_subjects: int,
                   dim_value: int,
                   rng: Optional[np.random.Generator] = None,
                   size: Optional[Tuple[int]] = None) -> np.ndarray:
    """
    This function generates samples from of a mixture component for prior predictive checks. We use the following
    definition in the comments below:

    s: number of subjects
    d: number of dimensions/features of the component
    T: number of time steps in the component's scale
    """

    T = coordination.shape[-1]
    N = num_subjects

    noise = rng.normal(loc=0, scale=1, size=size) * sigma[:, :, None]

    sample = np.zeros_like(noise)

    # Sample from prior in the initial time step
    sample[..., 0] = rng.normal(loc=initial_mean, scale=sigma, size=(num_subjects, dim_value))

    sum_matrix_others = (ptt.ones((N, N)) - ptt.eye(N)) / (N - 1)
    for t in np.arange(1, T):
        prev_others = np.dot(sum_matrix_others, sample[..., t - 1])  # s x d

        if self_dependent:
            # Previous sample from the same subject
            prev_same = sample[..., t - 1]
        else:
            # No dependency on the same subject. Sample from prior.
            prev_same = initial_mean

        blended_mean = (prev_others - prev_same) * coordination[t] + prev_same

        sample[..., t] = rng.normal(loc=blended_mean, scale=sigma)

    return sample + noise


class MixtureComponentParameters:

    def __init__(self, mean_mean_a0: np.ndarray, sd_mean_a0: np.ndarray, sd_sd_aa: np.ndarray):
        self.mean_a0 = Parameter(NormalParameterPrior(mean_mean_a0, sd_mean_a0))
        self.sd_aa = Parameter(HalfNormalParameterPrior(sd_sd_aa))

    def clear_values(self):
        self.mean_a0.value = None
        self.sd_aa.value = None


class MixtureComponentSamples:

    def __init__(self):
        self.values = np.array([])

        # For each time step in the component's scale, it contains the matching time step in the coordination scale
        self.time_steps_in_coordination_scale = np.array([])

    @property
    def num_time_steps(self):
        return self.values.shape[-1]


class MixtureComponent:
    """
    This class models a non-serial latent component which individual subject's dynamics influence that of the other
    subjects as controlled by coordination.
    """

    def __init__(self, uuid: str, num_subjects: int, dim_value: int, self_dependent: bool, mean_mean_a0: np.ndarray,
                 sd_mean_a0: np.ndarray, sd_sd_aa: np.ndarray, a_mixture_weights: np.ndarray,
                 share_mean_a0_across_subjects: bool, share_mean_a0_across_features: bool,
                 share_sd_aa_across_subjects: bool, share_sd_aa_across_features: bool):

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

        assert (num_subjects, num_subjects - 1) == a_mixture_weights.shape

        self.uuid = uuid
        self.num_subjects = num_subjects
        self.dim_value = dim_value
        self.self_dependent = self_dependent
        self.share_mean_a0_across_subjects = share_mean_a0_across_subjects
        self.share_mean_a0_across_features = share_mean_a0_across_features
        self.share_sd_aa_across_subjects = share_sd_aa_across_subjects
        self.share_sd_aa_across_features = share_sd_aa_across_features

        self.parameters = MixtureComponentParameters(mean_mean_a0=mean_mean_a0,
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

    def draw_samples(self, relative_frequency: float, coordination: np.ndarray,
                     seed: Optional[int] = None) -> MixtureComponentSamples:

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

        assert relative_frequency >= 1

        # Adjust dimensions according to parameter sharing specification
        if self.share_mean_a0_across_subjects:
            mean_a0 = self.parameters.mean_a0.value[None, None, :]
        else:
            mean_a0 = self.parameters.mean_a0.value[None, :]

        if self.share_sd_aa_across_subjects:
            sd_aa = self.parameters.sd_aa.value[None, None, :]
        else:
            sd_aa = self.parameters.sd_aa.value[None, :]

        # Generate samples
        set_random_seed(seed)

        samples = MixtureComponentSamples()

        num_time_steps_in_cpn_scale = int(coordination.shape[-1] / relative_frequency)
        samples.time_steps_in_coordination_scale = (np.arange(num_time_steps_in_cpn_scale) * relative_frequency).astype(
            int)

        # Draw values from the system dynamics. The default model generates samples by following a Gaussian random walk
        # with blended values from different subjects according to the coordination levels over time. Child classes
        # can implement their own dynamics, like spring-mass-damping systems for instance.
        samples.values = self._draw_from_system_dynamics(
            time_steps_in_coordination_scale=samples.time_steps_in_coordination_scale,
            sampled_coordination=coordination,
            mean_a0=mean_a0,
            sd_aa=sd_aa)

        return samples

    def _draw_from_system_dynamics(self, time_steps_in_coordination_scale: np.ndarray, sampled_coordination: np.ndarray,
                                   mean_a0: np.ndarray, sd_aa: np.ndarray) -> np.ndarray:
        """
        In this function we use the following notation in the comments:

        n: number of series/samples (first dimension of coordination)
        s: number of subjects
        d: number of features
        """

        num_series = sampled_coordination.shape[0]
        num_time_steps = len(time_steps_in_coordination_scale)
        values = np.zeros((num_series, self.num_subjects, self.dim_value, num_time_steps))

        N = self.num_subjects
        sum_matrix_others = (np.ones((N, N)) - np.eye(N)) / (N - 1)

        for t in range(num_time_steps):
            if t == 0:
                values[..., 0] = norm(loc=mean_a0, scale=sd_aa).rvs(
                    size=(num_series, self.num_subjects, self.dim_value))
            else:
                c = sampled_coordination[:, time_steps_in_coordination_scale[t]][:, None, None]  # n x 1 x 1

                prev_others = np.dot(sum_matrix_others, values[..., t - 1])  # n x s x d

                if self.self_dependent:
                    prev_same = values[..., t - 1]  # n x s x d
                else:
                    prev_same = mean_a0  # n x s x d

                blended_mean = (prev_others - prev_same) * c + prev_same  # n x s x d

                values[..., t] = norm(loc=blended_mean, scale=sd_aa).rvs()

        return values

    def _create_random_parameters(self, mean_a0: Optional[Any] = None, sd_aa: Optional[Any] = None):
        """
        This function creates the initial mean and standard deviation of the serialized component distribution as
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

                mean_a0 = mean_a0[None, :].repeat(self.num_subjects, axis=0)  # subject x feature
            else:
                mean_a0 = pm.Normal(name=self.mean_a0_name,
                                    mu=self.parameters.mean_a0.prior.mean,
                                    sigma=self.parameters.mean_a0.prior.sd,
                                    size=(self.num_subjects, dim_mean_a0_features),
                                    observed=self.parameters.mean_a0.value)

        # Initialize sd_aa parameter if it hasn't been defined previously
        if sd_aa is None:
            if self.share_sd_aa_across_subjects:
                sd_aa = pm.HalfNormal(name=self.sd_aa_name,
                                      sigma=self.parameters.sd_aa.prior.sd,
                                      size=dim_sd_aa_features,
                                      observed=self.parameters.sd_aa.value)
                sd_aa = sd_aa[None, :].repeat(self.num_subjects, axis=0)  # subject x feature
            else:
                sd_aa = pm.HalfNormal(name=self.sd_aa_name,
                                      sigma=self.parameters.sd_aa.prior.sd,
                                      size=(self.num_subjects, dim_sd_aa_features),
                                      observed=self.parameters.sd_aa.value)

        return mean_a0, sd_aa

    def update_pymc_model(self, coordination: Any, subject_dimension: str, feature_dimension: str, time_dimension: str,
                          observed_values: Optional[Any] = None, mean_a0: Optional[Any] = None,
                          sd_aa: Optional[Any] = None) -> Any:

        mean_a0, sd_aa = self._create_random_parameters(mean_a0, sd_aa)

        logp_params = (mean_a0,
                       sd_aa,
                       coordination,
                       np.array(self.self_dependent),
                       *self._get_extra_logp_params()
                       )
        logp_fn = self._get_logp_fn()
        random_fn = self._get_random_fn()
        mixture_component = pm.CustomDist(self.uuid, *logp_params,
                                          logp=logp_fn,
                                          random=random_fn,
                                          dims=[subject_dimension, feature_dimension, time_dimension],
                                          observed=observed_values)

        return mixture_component, mean_a0, sd_aa

    def _get_extra_logp_params(self):
        """
        Child classes can pass extra parameters to the logp and random functions
        """
        return ()

    def _get_logp_fn(self):
        """
        Child classes can define their own logp functions
        """
        return mixture_logp

    def _get_random_fn(self):
        """
        Child classes can define their own random functions for prior predictive checks
        """
        return partial(mixture_random, num_subjects=self.num_subjects, dim_value=self.dim_value)
