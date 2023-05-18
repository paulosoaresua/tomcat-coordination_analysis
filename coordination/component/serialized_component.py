from typing import Any, List, Optional, Tuple

from enum import Enum
import numpy as np
import pymc as pm
import pytensor.tensor as ptt
from scipy.stats import norm

from coordination.common.activation_function import ActivationFunction
from coordination.common.utils import set_random_seed
from coordination.model.parametrization import Parameter, HalfNormalParameterPrior, NormalParameterPrior


class Mode(Enum):
    BLENDING = 0
    MIXTURE = 1


def blending_logp(serialized_component: Any,
                  initial_mean: Any,
                  sigma: Any,
                  coordination: Any,
                  prev_time_same_subject: ptt.TensorConstant,
                  prev_time_diff_subject: ptt.TensorConstant,
                  prev_same_subject_mask: Any,
                  prev_diff_subject_mask: Any,
                  self_dependent: ptt.TensorConstant):
    # We reshape to guarantee we don't create dimensions with unknown size in case the first dimension of
    # the serialized component is one
    D = serialized_component[..., prev_time_diff_subject].reshape(serialized_component.shape)  # d x t

    C = coordination[None, :]  # 1 x t
    DM = prev_diff_subject_mask[None, :]  # 1 x t

    if self_dependent.eval():
        # Coordination only affects the mean in time steps where there are previous observations from a different subject.
        # If there's no previous observation from the same subject, we use the initial mean.
        S = serialized_component[..., prev_time_same_subject].reshape(serialized_component.shape)  # d x t
        SM = prev_same_subject_mask[None, :]  # 1 x t
        mean = D * C * DM + (1 - C * DM) * (S * SM + (1 - SM) * initial_mean)
    else:
        # Coordination only affects the mean in time steps where there are previous observations from a different subject.
        mean = D * C * DM + (1 - C * DM) * initial_mean

    total_logp = pm.logp(pm.Normal.dist(mu=mean, sigma=sigma, shape=D.shape), serialized_component).sum()

    return total_logp


def blending_random(initial_mean: np.ndarray,
                    sigma: np.ndarray,
                    coordination: np.ndarray,
                    prev_time_same_subject: np.ndarray,
                    prev_time_diff_subject: np.ndarray,
                    prev_same_subject_mask: np.ndarray,
                    prev_diff_subject_mask: np.ndarray,
                    self_dependent: bool,
                    rng: Optional[np.random.Generator] = None,
                    size: Optional[Tuple[int]] = None) -> np.ndarray:
    num_time_steps = coordination.shape[-1]

    noise = rng.normal(loc=0, scale=1, size=size) * sigma

    sample = np.zeros_like(noise)

    mean_0 = initial_mean if initial_mean.ndim == 1 else initial_mean[..., 0]
    sd_0 = sigma if sigma.ndim == 1 else sigma[..., 0]

    prior_sample = rng.normal(loc=mean_0, scale=sd_0)
    sample[..., 0] = prior_sample

    for t in np.arange(1, num_time_steps):
        # Previous sample from a different individual
        D = sample[..., prev_time_diff_subject[t]]  # d-vector

        # Previous sample from the same individual
        if self_dependent and prev_same_subject_mask[t] == 1:
            S = sample[..., prev_time_same_subject[t]]
        else:
            # When there's no self-dependency, the transition distribution is a blending between the previous value
            # from another individual, and a fixed mean.
            if initial_mean.shape[1] == 1:
                S = initial_mean[..., 0]
            else:
                S = initial_mean[..., t]

        mean = ((D - S) * coordination[t] * prev_diff_subject_mask[t] + S)

        if sigma.shape[1] == 1:
            # Parameter sharing across subjects
            transition_sample = rng.normal(loc=mean, scale=sigma[..., 0])
        else:
            transition_sample = rng.normal(loc=mean, scale=sigma[..., t])

        sample[..., t] = transition_sample

    return sample + noise


def mixture_logp(serialized_component: Any,
                 initial_mean: Any,
                 sigma: Any,
                 coordination: Any,
                 input_layer_f: Any,
                 hidden_layers_f: Any,
                 output_layer_f: Any,
                 activation_function_number_f: ptt.TensorConstant,
                 prev_time_same_subject: ptt.TensorConstant,
                 prev_time_diff_subject: ptt.TensorConstant,
                 prev_same_subject_mask: Any,
                 prev_diff_subject_mask: Any,
                 pairs: ptt.TensorConstant,
                 self_dependent: ptt.TensorConstant):
    # We reshape to guarantee we don't create dimensions with unknown size in case the first dimension of
    # the serialized component is one
    D = serialized_component[..., prev_time_diff_subject].reshape(serialized_component.shape)  # d x t

    # Coordination only affects the mean in time steps where there are previous observations from a different subject.
    # If there's no previous observation from the same subject, we use the initial mean.
    if self_dependent.eval():
        S = serialized_component[..., prev_time_same_subject].reshape(serialized_component.shape)  # d x t
        SM = prev_same_subject_mask[None, :]  # 1 x t
        logp_same = pm.logp(pm.Normal.dist(mu=(S * SM + (1 - SM) * initial_mean), sigma=sigma, shape=D.shape),
                            serialized_component)
    else:
        logp_same = pm.logp(pm.Normal.dist(mu=initial_mean, sigma=sigma, shape=D.shape), serialized_component)

    logp_diff = pm.logp(pm.Normal.dist(mu=D, sigma=sigma, shape=D.shape), serialized_component)

    C = coordination[None, :]  # 1 x t
    DM = prev_diff_subject_mask[None, :]  # 1 x t
    total_logp = pm.math.log(C * DM * pm.math.exp(logp_diff) + (1 - C * DM) * pm.math.exp(logp_same)).sum()

    return total_logp


def mixture_random(initial_mean: np.ndarray,
                   sigma: np.ndarray,
                   coordination: np.ndarray,
                   input_layer_f: np.ndarray,
                   hidden_layers_f: np.ndarray,
                   output_layer_f: np.ndarray,
                   activation_function_number_f: int,
                   prev_time_same_subject: np.ndarray,
                   prev_time_diff_subject: np.ndarray,
                   prev_same_subject_mask: np.ndarray,
                   prev_diff_subject_mask: np.ndarray,
                   pairs: np.ndarray,
                   self_dependent: bool,
                   rng: Optional[np.random.Generator] = None,
                   size: Optional[Tuple[int]] = None) -> np.ndarray:
    num_time_steps = coordination.shape[-1]

    noise = rng.normal(loc=0, scale=1, size=size) * sigma

    sample = np.zeros_like(noise)

    mean_0 = initial_mean if initial_mean.ndim == 1 else initial_mean[..., 0]
    sd_0 = sigma if sigma.ndim == 1 else sigma[..., 0]

    prior_sample = rng.normal(loc=mean_0, scale=sd_0)
    sample[..., 0] = prior_sample

    activation = ActivationFunction.from_numpy_number(activation_function_number_f)
    for t in np.arange(1, num_time_steps):
        # Previous sample from a different individual
        D = sample[..., prev_time_diff_subject[t]]

        if prev_diff_subject_mask[t] == 1 and np.random.rand() <= coordination[t]:
            mean = D
        else:
            if self_dependent:
                # Previous sample from the same individual
                S = sample[..., prev_time_same_subject[t]] * prev_same_subject_mask[t]

                # If there's no previous value for the same subject, we just use the initial mean
                mean = S * prev_same_subject_mask[t] + mean_0 * (1 - prev_same_subject_mask[t])
            else:
                # Fixed mean value per subject
                if sigma.shape[1] == 1:
                    S = initial_mean[..., 0]
                else:
                    S = initial_mean[..., t]
                mean = S

        if sigma.shape[1] == 1:
            # Parameter sharing across subjects
            transition_sample = rng.normal(loc=mean, scale=sigma[..., 0])
        else:
            transition_sample = rng.normal(loc=mean, scale=sigma[..., t])

        sample[..., t] = transition_sample

    return sample + noise


class SerializedComponentParameters:

    def __init__(self, mean_mean_a0: np.ndarray, sd_mean_a0: np.ndarray, sd_sd_aa: np.ndarray):
        self.mean_a0 = Parameter(NormalParameterPrior(mean_mean_a0, sd_mean_a0))
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

    def __init__(self, uuid: str, num_subjects: int, dim_value: int, self_dependent: bool, mean_mean_a0: np.ndarray,
                 sd_mean_a0: np.ndarray, sd_sd_aa: np.ndarray, share_mean_a0_across_subjects: bool,
                 share_mean_a0_across_features: bool, share_sd_aa_across_subjects: bool,
                 share_sd_aa_across_features: bool, mode: Mode = Mode.BLENDING):

        # Check dimensionality of the parameters priors
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
        self.mode = mode

        self.parameters = SerializedComponentParameters(mean_mean_a0=mean_mean_a0,
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

    def draw_samples(self, num_series: int, time_scale_density: float,
                     coordination: np.ndarray, can_repeat_subject: bool,
                     seed: Optional[int] = None) -> SerializedComponentSamples:

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

        # Adjust parameters according to sharing options
        if self.share_mean_a0_across_subjects:
            mean_a0 = self.parameters.mean_a0.value[None, :].repeat(self.num_subjects, axis=0)
        else:
            mean_a0 = self.parameters.mean_a0.value

        if self.share_sd_aa_across_subjects:
            sd_aa = self.parameters.sd_aa.value[None, :].repeat(self.num_subjects, axis=0)
        else:
            sd_aa = self.parameters.sd_aa.value

        set_random_seed(seed)

        samples = SerializedComponentSamples()
        samples.values = []
        for s in range(num_series):
            sparse_subjects = self._draw_random_subjects(num_series, coordination.shape[-1], time_scale_density,
                                                         can_repeat_subject)
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
                samples.prev_time_same_subject[s][t] = prev_time_per_subject.get(samples.subjects[s][t], -1)

                for subject, time in prev_time_per_subject.items():
                    if subject == samples.subjects[s][t]:
                        continue

                    # Most recent time from a different subject
                    samples.prev_time_diff_subject[s][t] = time if samples.prev_time_diff_subject[s][t] == -1 else max(
                        samples.prev_time_diff_subject[s][t], time)

                prev_time_per_subject[samples.subjects[s][t]] = t

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

    def _draw_from_system_dynamics(self, time_steps_in_coordination_scale: np.ndarray, sampled_coordination: np.ndarray,
                                   subjects_in_time: np.ndarray, prev_time_same_subject: np.ndarray,
                                   prev_time_diff_subject: np.ndarray, mean_a0: np.ndarray,
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

            if prev_time_same_subject[t] < 0:
                # It is not only when t == 0 because the first utterance of a speaker can be later in the future.
                # t_0 is the initial utterance of one of the subjects only.

                mean = mean_a0[subject_idx_mean_a0]
                sd = sd_aa[subject_idx_sd_aa]

                values[:, t] = norm(loc=mean, scale=sd).rvs(size=self.dim_value)
            else:
                C = sampled_coordination[time_steps_in_coordination_scale[t]]

                if self.self_dependent:
                    # When there's self dependency, the component either depends on the previous value of another subject,
                    # or the previous value of the same subject.
                    S = values[..., prev_time_same_subject[t]]
                else:
                    # When there's no self dependency, the component either depends on the previous value of another subject,
                    # or it is samples around a fixed mean.
                    S = mean_a0[subject_idx_mean_a0]

                prev_diff_mask = (prev_time_diff_subject[t] != -1).astype(int)
                D = values[..., prev_time_diff_subject[t]]

                if self.mode == Mode.BLENDING:
                    mean = (D - S) * C * prev_diff_mask + S
                else:
                    if prev_diff_mask == 1 and np.random.rand() <= C:
                        mean = D
                    else:
                        mean = S
                sd = sd_aa[subject_idx_sd_aa]

                values[:, t] = norm(loc=mean, scale=sd).rvs()

        return values

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

        # Map 0 to -1
        subjects -= 1
        return subjects

    def _create_random_parameters(self, subjects: np.ndarray, mean_a0: Optional[Any] = None,
                                  sd_aa: Optional[Any] = None):
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

    def update_pymc_model(self, coordination: Any, prev_time_same_subject: np.ndarray,
                          prev_time_diff_subject: np.ndarray, prev_same_subject_mask: np.ndarray,
                          prev_diff_subject_mask: np.ndarray, subjects: np.ndarray, feature_dimension: str,
                          time_dimension: str, observed_values: Optional[Any] = None, mean_a0: Optional[Any] = None,
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
        serialized_component = pm.DensityDist(self.uuid, *logp_params,
                                              logp=logp_fn,
                                              random=random_fn,
                                              dims=[feature_dimension, time_dimension],
                                              observed=observed_values)

        return serialized_component, mean_a0, sd_aa

    def _get_extra_logp_params(self, subjects_in_time: np.ndarray):
        """
        Child classes can pass extra parameters to the logp and random functions
        """
        return ()

    def _get_logp_fn(self):
        return blending_logp if self.mode == Mode.BLENDING else mixture_logp

    def _get_random_fn(self):
        return blending_random if self.mode == Mode.BLENDING else mixture_random
