from typing import Any, Callable, Dict, List, Optional, Tuple

from enum import Enum
import numpy as np
import pymc as pm
import pytensor as pt
import pytensor.tensor as ptt
from scipy.stats import norm

from coordination.common.functions import one_hot_encode
from coordination.common.activation_function import ActivationFunction
from coordination.common.utils import set_random_seed
from coordination.model.parametrization import Parameter, HalfNormalParameterPrior, NormalParameterPrior


class Mode(Enum):
    BLENDING = 0
    MIXTURE = 1


def logp_f_feed_forward(X: Any,
                        nn_layers: Any,
                        activation_function_number: Any,
                        prev_same_subjects: ptt.TensorConstant,
                        prev_diff_subjects: ptt.TensorConstant):
    def forward(W, X, A, B, act_number):
        activation = ActivationFunction.from_number(act_number.eval())
        z = pm.math.dot(W.transpose(), ptt.concatenate([X, A, B, ptt.ones((1, X.shape[-1]))], axis=0))
        return activation(z)

    if nn_layers.shape.prod().eval() == 0:
        # If there are no layers, return original value
        return X

    res, updates = pt.scan(forward,
                           outputs_info=X,
                           sequences=[nn_layers],
                           non_sequences=[prev_same_subjects, prev_diff_subjects, activation_function_number])

    return res[-1]


def logp_f_feed_forward_no_self_dependency(X: Any,
                                           nn_layers: Any,
                                           activation_function_number: Any,
                                           prev_diff_subjects: ptt.TensorConstant):
    def forward(W, X, B, act_number):
        activation = ActivationFunction.from_number(act_number.eval())
        z = pm.math.dot(W.transpose(), ptt.concatenate([X, B, ptt.ones((1, X.shape[-1]))], axis=0))
        return activation(z)

    if np.any(np.array(nn_layers.shape) == 0):
        # If there are no layers, return original value
        return X

    res, updates = pt.scan(forward,
                           outputs_info=X,
                           sequences=[nn_layers],
                           non_sequences=[prev_diff_subjects, activation_function_number])
    return res[-1]


def blending_logp(serialized_component: Any,
                  initial_mean: Any,
                  sigma: Any,
                  coordination: Any,
                  f_nn_weights: Any,
                  f_activation_function_number: ptt.TensorConstant,
                  prev_time_same_subject: ptt.TensorConstant,
                  prev_time_diff_subject: ptt.TensorConstant,
                  prev_same_subject_mask: ptt.TensorConstant,
                  prev_diff_subject_mask: ptt.TensorConstant,
                  prev_same_subjects: ptt.TensorConstant,
                  prev_diff_subjects: ptt.TensorConstant):
    C = coordination[None, :]  # 1 x t
    S = serialized_component[..., prev_time_same_subject]  # d x t
    D = serialized_component[..., prev_time_diff_subject]  # d x t

    # Features + 2 one-hot-encode representation of the subject id + bias term
    input_dim = serialized_component.shape[0] + 2 * prev_same_subjects.shape[0] + 1
    num_layers = ptt.cast(f_nn_weights.shape[0] / input_dim, "int32")
    f_nn_layers = f_nn_weights.reshape((num_layers, input_dim, f_nn_weights.shape[-1]))

    D = logp_f_feed_forward(D, f_nn_layers, f_activation_function_number, prev_same_subjects, prev_diff_subjects)

    SM = prev_same_subject_mask[None, :]  # 1 x t
    DM = prev_diff_subject_mask[None, :]  # 1 x t

    # Coordination only affects the mean in time steps where there are previous observations from a different subject.
    # If there's no previous observation from the same subject, we use the initial mean.
    mean = D * C * DM + (1 - C * DM) * (S * SM + (1 - SM) * initial_mean)

    total_logp = pm.logp(pm.Normal.dist(mu=mean, sigma=sigma, shape=D.shape), serialized_component).sum()

    return total_logp


def blending_logp_no_self_dependency(serialized_component: Any,
                                     initial_mean: Any,
                                     sigma: Any,
                                     coordination: Any,
                                     f_nn_weights: Any,
                                     f_activation_function_number: ptt.TensorConstant,
                                     prev_time_diff_subject: ptt.TensorConstant,
                                     prev_diff_subject_mask: ptt.TensorConstant,
                                     prev_diff_subjects: ptt.TensorConstant):
    C = coordination[None, :]  # 1 x t
    D = serialized_component[..., prev_time_diff_subject]  # d x t

    # Features + 1 one-hot-encode representation of the subject id + bias term
    input_dim = serialized_component.shape[0] + prev_diff_subjects.shape[0] + 1
    num_layers = ptt.cast(f_nn_weights.shape[0] / input_dim, "int32")
    f_nn_layers = f_nn_weights.reshape((num_layers, input_dim, f_nn_weights.shape[-1]))
    D = logp_f_feed_forward_no_self_dependency(D, f_nn_layers, f_activation_function_number, prev_diff_subjects)

    DM = prev_diff_subject_mask[None, :]  # 1 x t

    # Coordination only affects the mean in time steps where there are previous observations from a different subject.
    mean = D * C * DM + (1 - C * DM) * initial_mean

    total_logp = pm.logp(pm.Normal.dist(mu=mean, sigma=sigma, shape=D.shape), serialized_component).sum()

    return total_logp


def blending_random(initial_mean: np.ndarray,
                    sigma: np.ndarray,
                    coordination: np.ndarray,
                    f_nn_weights: np.ndarray,
                    f_activation_function_number: int,
                    prev_time_same_subject: np.ndarray,
                    prev_time_diff_subject: np.ndarray,
                    prev_same_subject_mask: np.ndarray,
                    prev_diff_subject_mask: np.ndarray,
                    prev_same_subjects: np.ndarray,
                    prev_diff_subjects: np.ndarray,
                    rng: Optional[np.random.Generator] = None,
                    size: Optional[Tuple[int]] = None) -> np.ndarray:
    num_time_steps = coordination.shape[-1]

    noise = rng.normal(loc=0, scale=1, size=size) * sigma

    sample = np.zeros_like(noise)

    mean_0 = initial_mean if initial_mean.ndim == 1 else initial_mean[..., 0]
    sd_0 = sigma if sigma.ndim == 1 else sigma[..., 0]

    prior_sample = rng.normal(loc=mean_0, scale=sd_0)
    sample[..., 0] = prior_sample

    # Features + 1 one-hot-encode representation of the subject id + bias term
    input_dim = mean_0.shape[0] + 2 * prev_diff_subjects.shape[0] + 1
    num_layers = int(f_nn_weights.shape[0] / input_dim)
    f_nn_layers = f_nn_weights.reshape((num_layers, input_dim, f_nn_weights.shape[-1]))
    activation = ActivationFunction.from_number(f_activation_function_number)

    for t in np.arange(1, num_time_steps):
        # Previous sample from a different individual
        D = sample[..., prev_time_diff_subject[t]]  # d-vector

        for W in f_nn_layers:
            # Transform D with a function that depends on the previous value of a different speaker, that speaker and
            # the current one's identity.
            z = np.dot(np.concatenate([D, prev_same_subjects[:, t], prev_diff_subjects[:, t], [1]]), W)
            D = activation(z)

        # Previous sample from the same individual
        if prev_same_subject_mask[t] == 1:
            S = sample[..., prev_time_same_subject[t]]
        else:
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


def blending_random_no_self_dependency(initial_mean: np.ndarray,
                                       sigma: np.ndarray,
                                       coordination: np.ndarray,
                                       f_nn_weights: np.ndarray,
                                       f_activation_function_number: int,
                                       prev_time_diff_subject: np.ndarray,
                                       prev_diff_subject_mask: np.ndarray,
                                       prev_diff_subjects: np.ndarray,
                                       rng: Optional[np.random.Generator] = None,
                                       size: Optional[Tuple[int]] = None) -> np.ndarray:
    num_time_steps = coordination.shape[-1]

    noise = rng.normal(loc=0, scale=1, size=size) * sigma

    sample = np.zeros_like(noise)

    # Features + 1 one-hot-encode representation of the subject id + bias term
    input_dim = sample.shape[0] + prev_diff_subjects.shape[0] + 1
    num_layers = int(f_nn_weights.shape[0] / input_dim)
    f_nn_layers = f_nn_weights.reshape((num_layers, input_dim, f_nn_weights.shape[-1]))
    activation = ActivationFunction.from_number(f_activation_function_number)

    for t in np.arange(1, num_time_steps):
        # Previous sample from a different individual
        D = sample[..., prev_time_diff_subject[t]]

        for W in f_nn_layers:
            # Transform D with a function that depends on the previous value of a different speaker, and its identify
            z = np.dot(np.concatenate([D, prev_diff_subjects[:, t], [1]]), W)
            D = activation(z)

        # No self-dependency. The transition distribution is a blending between the previous value from another individual,
        # and a fixed mean.
        if sigma.shape[1] == 1:
            # Parameter sharing across subjects
            S = initial_mean[..., 0]
            mean = ((D - S) * coordination[t] * prev_diff_subject_mask[t] + S)
            transition_sample = rng.normal(loc=mean, scale=sigma[..., 0])
        else:
            S = initial_mean[..., t]
            mean = ((D - S) * coordination[t] * prev_diff_subject_mask[t] + S)
            transition_sample = rng.normal(loc=mean, scale=sigma[..., t])

        sample[..., t] = transition_sample

    return sample + noise


def mixture_logp(serialized_component: Any,
                 initial_mean: Any,
                 sigma: Any,
                 coordination: Any,
                 f_nn_weights: Any,
                 f_activation_function_number: ptt.TensorConstant,
                 prev_time_same_subject: ptt.TensorConstant,
                 prev_time_diff_subject: ptt.TensorConstant,
                 prev_same_subject_mask: ptt.TensorConstant,
                 prev_diff_subject_mask: ptt.TensorConstant,
                 prev_same_subjects: ptt.TensorConstant,
                 prev_diff_subjects: ptt.TensorConstant):
    C = coordination[None, :]  # 1 x t
    S = serialized_component[..., prev_time_same_subject]  # d x t
    D = serialized_component[..., prev_time_diff_subject]  # d x t

    # Features + 2 one-hot-encode representation of the subject id + bias term
    input_dim = serialized_component.shape[0] + 2 * prev_same_subjects.shape[0] + 1
    num_layers = ptt.cast(f_nn_weights.shape[0] / input_dim, "int32")
    f_nn_layers = f_nn_weights.reshape((num_layers, input_dim, f_nn_weights.shape[-1]))
    D = logp_f_feed_forward(D, f_nn_layers, f_activation_function_number, prev_same_subjects, prev_diff_subjects)

    SM = prev_same_subject_mask[None, :]  # 1 x t
    DM = prev_diff_subject_mask[None, :]  # 1 x t

    # Coordination only affects the mean in time steps where there are previous observations from a different subject.
    # If there's no previous observation from the same subject, we use the initial mean.
    logp_same = pm.logp(pm.Normal.dist(mu=(S * SM + (1 - SM) * initial_mean), sigma=sigma, shape=D.shape),
                        serialized_component)
    logp_diff = pm.logp(pm.Normal.dist(mu=D, sigma=sigma, shape=D.shape), serialized_component)

    total_logp = pm.math.log(C * DM * pm.math.exp(logp_diff) + (1 - C * DM) * pm.math.exp(logp_same)).sum()

    return total_logp


def mixture_logp_no_self_dependency(serialized_component: Any,
                                    initial_mean: Any,
                                    sigma: Any,
                                    coordination: Any,
                                    f_nn_weights: Any,
                                    f_activation_function_number: ptt.TensorConstant,
                                    prev_time_diff_subject: ptt.TensorConstant,
                                    prev_diff_subject_mask: ptt.TensorConstant,

                                    prev_diff_subjects: ptt.TensorConstant):
    C = coordination[None, :]  # 1 x t
    D = serialized_component[..., prev_time_diff_subject]  # d x t

    # Features + 1 one-hot-encode representation of the subject id + bias term
    input_dim = serialized_component.shape[0] + prev_diff_subjects.shape[0] + 1
    num_layers = ptt.cast(f_nn_weights.shape[0] / input_dim, "int32")
    f_nn_layers = f_nn_weights.reshape((num_layers, input_dim, f_nn_weights.shape[-1]))
    D = logp_f_feed_forward_no_self_dependency(D, f_nn_layers, f_activation_function_number, prev_diff_subjects)

    DM = prev_diff_subject_mask[None, :]  # 1 x t

    # Coordination only affects the mean in time steps where there are previous observations from a different subject.
    logp_same = pm.logp(pm.Normal.dist(mu=initial_mean, sigma=sigma, shape=D.shape), serialized_component)
    logp_diff = pm.logp(pm.Normal.dist(mu=D, sigma=sigma, shape=D.shape), serialized_component)

    total_logp = pm.math.log(C * DM * pm.math.exp(logp_diff) + (1 - C * DM) * pm.math.exp(logp_same)).sum()

    return total_logp


def mixture_random(initial_mean: np.ndarray,
                   sigma: np.ndarray,
                   coordination: np.ndarray,
                   f_nn_weights: np.ndarray,
                   f_activation_function_number: int,
                   prev_time_same_subject: np.ndarray,
                   prev_time_diff_subject: np.ndarray,
                   prev_same_subject_mask: np.ndarray,
                   prev_diff_subject_mask: np.ndarray,
                   prev_same_subjects: np.ndarray,
                   prev_diff_subjects: np.ndarray,
                   rng: Optional[np.random.Generator] = None,
                   size: Optional[Tuple[int]] = None) -> np.ndarray:
    num_time_steps = coordination.shape[-1]

    noise = rng.normal(loc=0, scale=1, size=size) * sigma

    sample = np.zeros_like(noise)

    mean_0 = initial_mean if initial_mean.ndim == 1 else initial_mean[..., 0]
    sd_0 = sigma if sigma.ndim == 1 else sigma[..., 0]

    prior_sample = rng.normal(loc=mean_0, scale=sd_0)
    sample[..., 0] = prior_sample

    # Features + 1 one-hot-encode representation of the subject id + bias term
    input_dim = mean_0.shape[0] + 2 * prev_diff_subjects.shape[0] + 1
    num_layers = int(f_nn_weights.shape[0] / input_dim)
    f_nn_layers = f_nn_weights.reshape((num_layers, input_dim, f_nn_weights.shape[-1]))
    activation = ActivationFunction.from_number(f_activation_function_number)

    for t in np.arange(1, num_time_steps):
        # Previous sample from a different individual
        D = sample[..., prev_time_diff_subject[t]]

        for W in f_nn_layers:
            # Transform D with a function that depends on the previous value of a different speaker, that speaker and
            # the current one's identity.
            z = np.dot(np.concatenate([D, prev_same_subjects[:, t], prev_diff_subjects[:, t], [1]]), W)
            D = activation(z)

        # Previous sample from the same individual
        S = sample[..., prev_time_same_subject[t]] * prev_same_subject_mask[t]

        if prev_diff_subject_mask[t] == 1 and np.random.rand() <= coordination[t]:
            mean = D
        else:
            mean = S * prev_same_subject_mask[t] + mean_0 * (1 - prev_same_subject_mask[t])

        if sigma.shape[1] == 1:
            # Parameter sharing across subjects
            transition_sample = rng.normal(loc=mean, scale=sigma[..., 0])
        else:
            transition_sample = rng.normal(loc=mean, scale=sigma[..., t])

        sample[..., t] = transition_sample

    return sample + noise


def mixture_random_no_self_dependency(initial_mean: np.ndarray,
                                      sigma: np.ndarray,
                                      coordination: np.ndarray,
                                      f_nn_weights: np.ndarray,
                                      f_activation_function_number: int,
                                      prev_time_diff_subject: np.ndarray,
                                      prev_diff_subject_mask: np.ndarray,
                                      prev_diff_subjects: np.ndarray,
                                      rng: Optional[np.random.Generator] = None,
                                      size: Optional[Tuple[int]] = None) -> np.ndarray:
    num_time_steps = coordination.shape[-1]

    noise = rng.normal(loc=0, scale=1, size=size) * sigma

    sample = np.zeros_like(noise)

    # Features + 1 one-hot-encode representation of the subject id + bias term
    input_dim = sample.shape[0] + prev_diff_subjects.shape[0] + 1
    num_layers = int(f_nn_weights.shape[0] / input_dim)
    f_nn_layers = f_nn_weights.reshape((num_layers, input_dim, f_nn_weights.shape[-1]))
    activation = ActivationFunction.from_number(f_activation_function_number)

    for t in np.arange(1, num_time_steps):
        # Previous sample from a different individual
        D = sample[..., prev_time_diff_subject[t]]

        for W in f_nn_layers:
            # Transform D with a function that depends on the previous value of a different speaker, that speaker and
            # the current one's identity.
            z = np.dot(np.concatenate([D, prev_diff_subjects[:, t], [1]]), W)
            D = activation(z)

        if sigma.shape[1] == 1:
            # Parameter sharing across subjects
            S = initial_mean[..., 0]
        else:
            S = initial_mean[..., t]

        if prev_diff_subject_mask[t] == 1 and np.random.rand() <= coordination[t]:
            mean = D
        else:
            mean = S

        # No self-dependency. The transition distribution is a blending between the previous value from another individual,
        # and a fixed mean.
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

        # Map between subjects and their genders
        self.gender_map: Dict[int, int] = {}

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
                 sd_mean_a0: np.ndarray, sd_sd_aa: np.ndarray, share_params_across_subjects: bool,
                 share_params_across_genders: bool, share_params_across_features: bool, mode: Mode = Mode.BLENDING,
                 f: Optional[Callable] = None):
        assert not (share_params_across_subjects and share_params_across_genders)

        dim = 1 if share_params_across_features else dim_value
        if share_params_across_subjects:
            assert (dim,) == mean_mean_a0.shape
            assert (dim,) == sd_mean_a0.shape
            assert (dim,) == sd_sd_aa.shape
        elif share_params_across_genders:
            # 2 genders: Male or Female
            assert (2, dim) == mean_mean_a0.shape
            assert (2, dim) == sd_mean_a0.shape
            assert (2, dim) == sd_sd_aa.shape
        else:
            assert (num_subjects, dim) == mean_mean_a0.shape
            assert (num_subjects, dim) == sd_mean_a0.shape
            assert (num_subjects, dim) == sd_sd_aa.shape

        self.uuid = uuid
        self.num_subjects = num_subjects
        self.dim_value = dim_value
        self.self_dependent = self_dependent
        self.share_params_across_subjects = share_params_across_subjects
        self.share_params_across_genders = share_params_across_genders
        self.share_params_across_features = share_params_across_features
        self.mode = mode
        self.f = f

        self.parameters = SerializedComponentParameters(mean_mean_a0=mean_mean_a0,
                                                        sd_mean_a0=sd_mean_a0,
                                                        sd_sd_aa=sd_sd_aa)

    @property
    def parameter_names(self) -> List[str]:
        return [
            self.mean_a0_name,
            self.sd_aa_name
        ]

    @property
    def mean_a0_name(self) -> str:
        return f"mean_a0_{self.uuid}"

    @property
    def sd_aa_name(self) -> str:
        return f"sd_aa_{self.uuid}"

    @property
    def f_nn_weights_name(self) -> str:
        return f"f_nn_weights_{self.uuid}"

    def draw_samples(self, num_series: int, time_scale_density: float,
                     coordination: np.ndarray, can_repeat_subject: bool,
                     seed: Optional[int] = None) -> SerializedComponentSamples:

        dim = 1 if self.share_params_across_features else self.dim_value
        if self.share_params_across_subjects:
            assert (dim,) == self.parameters.mean_a0.value.shape
            assert (dim,) == self.parameters.sd_aa.value.shape
        elif self.share_params_across_genders:
            assert (2, dim) == self.parameters.mean_a0.value.shape
            assert (2, dim) == self.parameters.sd_aa.value.shape
        else:
            assert (self.num_subjects, dim) == self.parameters.mean_a0.value.shape
            assert (self.num_subjects, dim) == self.parameters.sd_aa.value.shape

        assert 0 <= time_scale_density <= 1

        set_random_seed(seed)

        samples = SerializedComponentSamples()

        if self.share_params_across_subjects:
            mean_a0 = self.parameters.mean_a0.value[None, :].repeat(self.num_subjects, axis=0)
            sd_aa = self.parameters.sd_aa.value[None, :].repeat(self.num_subjects, axis=0)
        else:
            mean_a0 = self.parameters.mean_a0.value
            sd_aa = self.parameters.sd_aa.value

        for s in range(num_series):
            sparse_subjects = self._draw_random_subjects(num_series, coordination.shape[-1], time_scale_density,
                                                         can_repeat_subject)
            samples.subjects.append(np.array([s for s in sparse_subjects[s] if s >= 0], dtype=int))
            samples.time_steps_in_coordination_scale.append(
                np.array([t for t, s in enumerate(sparse_subjects[s]) if s >= 0], dtype=int))

            # Make it simple for gender. Even subjects are Male and odd Female.
            samples.gender_map = {idx: idx % 2 for idx in range(self.num_subjects)}

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

                curr_subject = samples.subjects[s][t]
                if self.share_params_across_genders:
                    curr_subject = samples.gender_map[curr_subject]
                elif self.share_params_across_subjects:
                    curr_subject = 0

                if samples.prev_time_same_subject[s][t] < 0:
                    # It is not only when t == 0 because the first utterance of a speaker can be later in the future.
                    # t_0 is the initial utterance of one of the subjects only.

                    mean = mean_a0[curr_subject]
                    sd = sd_aa[curr_subject]

                    samples.values[s][:, t] = norm(loc=mean, scale=sd).rvs(size=self.dim_value)
                else:
                    C = coordination[s, samples.time_steps_in_coordination_scale[s][t]]

                    if self.self_dependent:
                        # When there's self dependency, the component either depends on the previous value of another subject,
                        # or the previous value of the same subject.
                        S = samples.values[s][..., samples.prev_time_same_subject[s][t]]
                    else:
                        # When there's no self dependency, the component either depends on the previous value of another subject,
                        # or it is samples around a fixed mean.
                        S = mean_a0[curr_subject]

                    prev_diff_mask = (samples.prev_time_diff_subject[s][t] != -1).astype(int)
                    D = samples.values[s][..., samples.prev_time_diff_subject[s][t]]

                    if self.f is not None:
                        source_subject = samples.subjects[s][samples.prev_time_diff_subject[s][t]]

                        if self.self_dependent:
                            target_subject = samples.subjects[s][t]
                            D = self.f(D, source_subject, target_subject)
                        else:
                            D = self.f(D, source_subject)

                    if self.mode == Mode.BLENDING:
                        mean = (D - S) * C * prev_diff_mask + S
                    else:
                        if prev_diff_mask == 1 and np.random.rand() <= C:
                            mean = D
                        else:
                            mean = S
                    sd = sd_aa[curr_subject]

                    samples.values[s][:, t] = norm(loc=mean, scale=sd).rvs()

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

        # Map 0 to -1
        subjects -= 1
        return subjects

    def update_pymc_model(self, coordination: Any, prev_time_same_subject: np.ndarray,
                          prev_time_diff_subject: np.ndarray, prev_same_subject_mask: np.ndarray,
                          prev_diff_subject_mask: np.ndarray, subjects: np.ndarray, gender_map: Dict[int, int],
                          feature_dimension: str, time_dimension: str, observed_values: Optional[Any] = None,
                          num_hidden_layers_f: int = 0, activation_function_f: str = "linear") -> Any:

        dim = 1 if self.share_params_across_features else self.dim_value
        if self.share_params_across_subjects:
            mean_a0 = pm.Normal(name=self.mean_a0_name, mu=self.parameters.mean_a0.prior.mean,
                                sigma=self.parameters.mean_a0.prior.sd, size=dim,
                                observed=self.parameters.mean_a0.value)
            sd_aa = pm.HalfNormal(name=self.sd_aa_name, sigma=self.parameters.sd_aa.prior.sd,
                                  size=dim, observed=self.parameters.sd_aa.value)

            # Resulting dimension: (features, 1). The last dimension will be broadcasted across time.
            mean = mean_a0[:, None]
            sd = sd_aa[:, None]
        elif self.share_params_across_genders:
            mean_a0 = pm.Normal(name=self.mean_a0_name, mu=self.parameters.mean_a0.prior.mean,
                                sigma=self.parameters.mean_a0.prior.sd, size=(2, dim),
                                observed=self.parameters.mean_a0.value)
            sd_aa = pm.HalfNormal(name=self.sd_aa_name, sigma=self.parameters.sd_aa.prior.sd,
                                  size=(2, dim), observed=self.parameters.sd_aa.value)

            # One mean and sd per time step matching their subjects' genders. The indexing below results in a matrix of
            # dimensions: (features, time)
            genders = np.array([gender_map[subject] for subject in subjects], dtype=int)
            mean = mean_a0[genders].transpose()
            sd = sd_aa[genders].transpose()
        else:
            mean_a0 = pm.Normal(name=self.mean_a0_name, mu=self.parameters.mean_a0.prior.mean,
                                sigma=self.parameters.mean_a0.prior.sd, size=(self.num_subjects, dim),
                                observed=self.parameters.mean_a0.value)
            sd_aa = pm.HalfNormal(name=self.sd_aa_name, sigma=self.parameters.sd_aa.prior.sd,
                                  size=(self.num_subjects, dim), observed=self.parameters.sd_aa.value)

            # One mean and sd per time step matching their subjects. The indexing below results in a matrix of
            # dimensions: (features, time)
            mean = mean_a0[subjects].transpose()
            sd = sd_aa[subjects].transpose()

        if self.share_params_across_features:
            mean = mean.repeat(self.dim_value, axis=0)
            sd = sd.repeat(self.dim_value, axis=0)

        # We use a neural network to fit the function f
        weights = []
        weights_reshaped = []
        if num_hidden_layers_f > 0:
            # Extra dimensions for one-hot-encode of the previous subject ids
            # If there's self-dependency then we need two subjects, the id of the current subject and the id of the
            # previous different subject. If there's no self-dependency, we only need the id of the latter.
            one_hot_encode_size = self.num_subjects
            extra_dim = 2 * one_hot_encode_size if self.self_dependent else one_hot_encode_size

            # The bias term goes in the last row
            extra_dim += 1

            weight_dims = (num_hidden_layers_f, self.dim_value + extra_dim, self.dim_value)

            weights = pm.Normal(f"{self.f_nn_weights_name}", mu=0, sigma=1, size=weight_dims)

            # There's a bug in PyMC 5.0.2 that we cannot pass an argument with more dimensions than the
            # dimension of CustomDist. To work around it, I join will the layer dimension with the input one.
            # Inside the logp function, I will reshape the weights variable back to its original 3 dimensions:
            # #layers x #input x #output, so we can perform the feed-forward step.
            weights_reshaped = pm.Deterministic('weights_reshaped', weights.reshape(
                (num_hidden_layers_f * (self.dim_value + extra_dim), self.dim_value)))

        # f_nn_layers = []
        # f_bias = []
        # if num_hidden_layers_f > 0:
        #     # Extra dimensions for one-hot-encode of the previous subject ids
        #     # If there's self-dependency then we need two subjects, the id of the current subject and the id of the
        #     # previous different subject. If there's no self-dependency, we only need the id of the latter.
        #     one_hot_encode_size = self.num_subjects
        #     extra_dim = 2 * one_hot_encode_size if self.self_dependent else one_hot_encode_size
        #     weight_dims = (self.dim_value + extra_dim, self.dim_value)
        #
        #     # Each weight receives a vector of features plus 2 unique numbers identifying the previous and current
        #     # speaker
        #     f_nn_layers = pm.Normal(self.f_nn_weights_name, mu=0, sigma=1, size=weight_dims)
        #     f_bias = pm.Normal(f"{self.f_nn_weights_name}_bias", mu=0, sigma=1, size=self.dim_value)

        f_activation_function_number = ActivationFunction.NAME_TO_NUMBER[activation_function_f]

        if self.self_dependent:
            logp_params = (mean,
                           sd,
                           coordination,
                           weights_reshaped,
                           f_activation_function_number,
                           ptt.constant(prev_time_same_subject),
                           ptt.constant(prev_time_diff_subject),
                           ptt.constant(prev_same_subject_mask),
                           ptt.constant(prev_diff_subject_mask),
                           ptt.constant(one_hot_encode(subjects[prev_time_same_subject], self.num_subjects)),
                           ptt.constant(one_hot_encode(subjects[prev_time_diff_subject], self.num_subjects)))
            logp_fn = blending_logp if self.mode == Mode.BLENDING else mixture_logp
            random_fn = blending_random if self.mode == Mode.BLENDING else mixture_random
            serialized_component = pm.DensityDist(self.uuid, *logp_params, logp=logp_fn, random=random_fn,
                                                  dims=[feature_dimension, time_dimension],
                                                  observed=observed_values)
        else:
            logp_params = (mean,
                           sd,
                           coordination,
                           weights_reshaped,
                           f_activation_function_number,
                           ptt.constant(prev_time_diff_subject),
                           ptt.constant(prev_diff_subject_mask),
                           ptt.constant(one_hot_encode(subjects[prev_time_diff_subject], self.num_subjects)))
            logp_fn = blending_logp_no_self_dependency if self.mode == Mode.BLENDING else mixture_logp_no_self_dependency
            random_fn = blending_random_no_self_dependency if self.mode == Mode.BLENDING else mixture_random_no_self_dependency
            serialized_component = pm.DensityDist(self.uuid, *logp_params, logp=logp_fn, random=random_fn,
                                                  dims=[feature_dimension, time_dimension],
                                                  observed=observed_values)

        return serialized_component, mean_a0, sd_aa, weights
