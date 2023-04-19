from typing import Any, Callable, Dict, List, Optional, Tuple

from enum import Enum
import math
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


def add_bias(X: Any):
    if isinstance(X, np.ndarray):
        return np.concatenate([X, np.ones((1, X.shape[-1]))], axis=0)
    else:
        return ptt.concatenate([X, ptt.ones((1, X.shape[-1]))], axis=0)


def feed_forward_logp_f(input_data: Any,
                        input_layer_f: Any,
                        hidden_layers_f: Any,
                        output_layer_f: Any,
                        activation_function_number_f: ptt.TensorConstant,
                        pairs: Any):
    def forward(W, X, act_number):
        fn = ActivationFunction.from_pytensor_number(act_number.eval())
        z = pm.math.dot(W.transpose(), add_bias(X))
        return fn(z)

    if input_layer_f.shape.prod().eval() == 0:
        # Only transform the input data if a NN was specified
        return input_data

    hidden_dim = input_layer_f.shape[1]  # == f_nn_output_layer.shape[0]

    # Concatenate the pair IDs over time to the input data with the features over time
    input_data = ptt.concatenate([input_data, pairs], axis=0)

    # Input layer activations
    activation = ActivationFunction.from_pytensor_number(activation_function_number_f.eval())
    a0 = activation(pm.math.dot(input_layer_f.transpose(), add_bias(input_data)))

    # Reconstruct hidden layers as a 3 dimensional tensor, where the first dimension represents the number of layers.
    num_hidden_layers = ptt.cast(hidden_layers_f.shape[0] / (hidden_dim + 1), "int32")
    hidden_layers_f = hidden_layers_f.reshape((num_hidden_layers, hidden_dim + 1, hidden_dim))

    # Feed-Forward through the hidden layers
    res, updates = pt.scan(forward,
                           outputs_info=a0,
                           sequences=[hidden_layers_f],
                           non_sequences=[activation_function_number_f])

    h = res[-1]

    # Output layer activation
    out = activation(pm.math.dot(output_layer_f.transpose(), add_bias(h)))

    return out


def feed_forward_random_f(input_data: np.ndarray,
                          input_layer_f: np.ndarray,
                          hidden_layers_f: np.ndarray,
                          output_layer_f: np.ndarray,
                          activation: Callable,
                          pairs: np.ndarray):
    if len(input_layer_f) == 0:
        return input_data

    hidden_dim = input_layer_f.shape[1]  # == f_nn_output_layer.shape[0]

    # Concatenate the pair IDs to the input data with the features.
    input_data = np.concatenate([input_data, pairs], axis=0)

    # Input layer activations
    a0 = activation(np.dot(input_layer_f.transpose(), add_bias(input_data)))

    # Reconstruct hidden layers as a 3 dimensional tensor, where the first dimension represents the number of layers.
    num_hidden_layers = int(hidden_layers_f.shape[0] / (hidden_dim + 1))
    hidden_layers_f = hidden_layers_f.reshape((num_hidden_layers, hidden_dim + 1, hidden_dim))

    # Feed-Forward through the hidden layers
    h = a0
    for W in hidden_layers_f:
        h = activation(np.dot(W.transpose(), add_bias(h)))

    # Output layer activation.
    out = activation(np.dot(output_layer_f.transpose(), add_bias(h)))

    return out


def blending_logp(serialized_component: Any,
                  initial_mean: Any,
                  sigma: Any,
                  coordination: Any,
                  input_layer_f: Any,
                  hidden_layers_f: Any,
                  output_layer_f: Any,
                  activation_function_number_f: ptt.TensorConstant,
                  prev_time_same_subject: ptt.TensorConstant,
                  prev_time_diff_subject: ptt.TensorConstant,
                  prev_same_subject_mask: ptt.TensorConstant,
                  prev_diff_subject_mask: ptt.TensorConstant,
                  pairs: ptt.TensorConstant):
    C = coordination[None, :]  # 1 x t

    # We reshape to guarantee we don't create dimensions with unknown size in case the first dimension of
    # the serialized component is one
    S = serialized_component[..., prev_time_same_subject].reshape(serialized_component.shape)  # d x t
    D = serialized_component[..., prev_time_diff_subject].reshape(serialized_component.shape)  # d x t

    D = feed_forward_logp_f(input_data=D,
                            input_layer_f=input_layer_f,
                            hidden_layers_f=hidden_layers_f,
                            output_layer_f=output_layer_f,
                            activation_function_number_f=activation_function_number_f,
                            pairs=pairs)

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
                                     input_layer_f: Any,
                                     hidden_layers_f: Any,
                                     output_layer_f: Any,
                                     activation_function_number_f: ptt.TensorConstant,
                                     prev_time_diff_subject: ptt.TensorConstant,
                                     prev_diff_subject_mask: ptt.TensorConstant,
                                     pairs: ptt.TensorConstant):
    C = coordination[None, :]  # 1 x t

    # We reshape to guarantee we don't create dimensions with unknown size in case the first dimension of
    # the serialized component is one
    D = serialized_component[..., prev_time_diff_subject].reshape(serialized_component.shape)  # d x t

    D = feed_forward_logp_f(input_data=D,
                            input_layer_f=input_layer_f,
                            hidden_layers_f=hidden_layers_f,
                            output_layer_f=output_layer_f,
                            activation_function_number_f=activation_function_number_f,
                            pairs=pairs)

    DM = prev_diff_subject_mask[None, :]  # 1 x t

    # Coordination only affects the mean in time steps where there are previous observations from a different subject.
    mean = D * C * DM + (1 - C * DM) * initial_mean

    total_logp = pm.logp(pm.Normal.dist(mu=mean, sigma=sigma, shape=D.shape), serialized_component).sum()

    return total_logp


def blending_random(initial_mean: np.ndarray,
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
        D = sample[..., prev_time_diff_subject[t]]  # d-vector

        # Preserve the time dimension in the input_data and pair passed to the function for correct feed-forward pass
        D = feed_forward_random_f(input_data=D[:, None],
                                  input_layer_f=input_layer_f,
                                  hidden_layers_f=hidden_layers_f,
                                  output_layer_f=output_layer_f,
                                  activation=activation,
                                  pairs=pairs[:, t][:, None])[:, 0]

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
                                       input_layer_f: np.ndarray,
                                       hidden_layers_f: np.ndarray,
                                       output_layer_f: np.ndarray,
                                       activation_function_number_f: int,
                                       prev_time_diff_subject: np.ndarray,
                                       prev_diff_subject_mask: np.ndarray,
                                       pairs: np.ndarray,
                                       rng: Optional[np.random.Generator] = None,
                                       size: Optional[Tuple[int]] = None) -> np.ndarray:
    num_time_steps = coordination.shape[-1]

    noise = rng.normal(loc=0, scale=1, size=size) * sigma

    sample = np.zeros_like(noise)

    activation = ActivationFunction.from_numpy_number(activation_function_number_f)
    for t in np.arange(1, num_time_steps):
        # Previous sample from a different individual
        D = sample[..., prev_time_diff_subject[t]]

        # Preserve the time dimension in the input_data and pair passed to the function for correct feed-forward pass
        D = feed_forward_random_f(input_data=D[:, None],
                                  input_layer_f=input_layer_f,
                                  hidden_layers_f=hidden_layers_f,
                                  output_layer_f=output_layer_f,
                                  activation=activation,
                                  pairs=pairs[:, t][:, None])[:, 0]

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
                 input_layer_f: Any,
                 hidden_layers_f: Any,
                 output_layer_f: Any,
                 activation_function_number_f: ptt.TensorConstant,
                 prev_time_same_subject: ptt.TensorConstant,
                 prev_time_diff_subject: ptt.TensorConstant,
                 prev_same_subject_mask: ptt.TensorConstant,
                 prev_diff_subject_mask: ptt.TensorConstant,
                 pairs: ptt.TensorConstant):
    C = coordination[None, :]  # 1 x t

    # We reshape to guarantee we don't create dimensions with unknown size in case the first dimension of
    # the serialized component is one
    S = serialized_component[..., prev_time_same_subject].reshape(serialized_component.shape)  # d x t
    D = serialized_component[..., prev_time_diff_subject].reshape(serialized_component.shape)  # d x t

    D = feed_forward_logp_f(input_data=D,
                            input_layer_f=input_layer_f,
                            hidden_layers_f=hidden_layers_f,
                            output_layer_f=output_layer_f,
                            activation_function_number_f=activation_function_number_f,
                            pairs=pairs)

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
                                    input_layer_f: Any,
                                    hidden_layers_f: Any,
                                    output_layer_f: Any,
                                    activation_function_number_f: ptt.TensorConstant,
                                    prev_time_diff_subject: ptt.TensorConstant,
                                    prev_diff_subject_mask: ptt.TensorConstant,
                                    pairs: ptt.TensorConstant):
    C = coordination[None, :]  # 1 x t

    # We reshape to guarantee we don't create dimensions with unknown size in case the first dimension of
    # the serialized component is one
    D = serialized_component[..., prev_time_diff_subject].reshape(serialized_component.shape)  # d x t

    D = feed_forward_logp_f(input_data=D,
                            input_layer_f=input_layer_f,
                            hidden_layers_f=hidden_layers_f,
                            output_layer_f=output_layer_f,
                            activation_function_number_f=activation_function_number_f,
                            pairs=pairs)

    DM = prev_diff_subject_mask[None, :]  # 1 x t

    # Coordination only affects the mean in time steps where there are previous observations from a different subject.
    logp_same = pm.logp(pm.Normal.dist(mu=initial_mean, sigma=sigma, shape=D.shape), serialized_component)
    logp_diff = pm.logp(pm.Normal.dist(mu=D, sigma=sigma, shape=D.shape), serialized_component)

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

        # Preserve the time dimension in the input_data and pair passed to the function for correct feed-forward pass
        D = feed_forward_random_f(input_data=D[:, None],
                                  input_layer_f=input_layer_f,
                                  hidden_layers_f=hidden_layers_f,
                                  output_layer_f=output_layer_f,
                                  activation=activation,
                                  pairs=pairs[:, t][:, None])[:, 0]

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
                                      input_layer_f: np.ndarray,
                                      hidden_layers_f: np.ndarray,
                                      output_layer_f: np.ndarray,
                                      activation_function_number_f: int,
                                      prev_time_diff_subject: np.ndarray,
                                      prev_diff_subject_mask: np.ndarray,
                                      pairs: np.ndarray,
                                      rng: Optional[np.random.Generator] = None,
                                      size: Optional[Tuple[int]] = None) -> np.ndarray:
    num_time_steps = coordination.shape[-1]

    noise = rng.normal(loc=0, scale=1, size=size) * sigma

    sample = np.zeros_like(noise)

    activation = ActivationFunction.from_numpy_number(activation_function_number_f)
    for t in np.arange(1, num_time_steps):
        # Previous sample from a different individual
        D = sample[..., prev_time_diff_subject[t]]

        # Preserve the time dimension in the input_data and pair passed to the function for correct feed-forward pass
        D = feed_forward_random_f(input_data=D[:, None],
                                  input_layer_f=input_layer_f,
                                  hidden_layers_f=hidden_layers_f,
                                  output_layer_f=output_layer_f,
                                  activation=activation,
                                  pairs=pairs[:, t][:, None])[:, 0]

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
        # A list containing 3 tensors. One with the weights of the first layer, the second with the weights for the
        # hidden layers (3 dimensions, with first dimension indicating the number of layers), and the third containing
        # the weights of the output layer.
        self.weights_f: Optional[List[np.ndarray]] = None

    def clear_values(self):
        self.mean_a0.value = None
        self.sd_aa.value = None
        self.weights_f = None


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
                        target_subject = samples.subjects[s][t]

                        D = self.f(D, source_subject, target_subject)

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

    def _create_random_parameters(self, subjects: np.ndarray, gender_map: Dict[int, int]):
        """
        This function creates the initial mean and standard deviation of the serialized component distribution as
        random variables.
        """
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

        return mean, sd, mean_a0, sd_aa

    def _create_random_weights_f(self, num_hidden_layers: int, dim_hidden_layer: int, activation_function_name: str):
        """
        This function creates the weights used to fit the function f(.) as random variables. Because the serialized
        component uses a CustomDist, all the arguments of the logp function we pass must be tensors. So, we cannot
        pass a list of tensors from different sizes, otherwise the program will crash when it tries to convert that
        to a single tensor. Therefore, the strategy is to have 3 sets of weights, the first one represents the weights
        in the input layer, the second will be a list of weights with the same dimensions, which represent the weights
        in the hidden layers, and the last one will be weights in the last (output) layer.
        """

        # Gather observations from each layer. If some weights are pre-set, we don't need to infer them.
        if self.parameters.weights_f is None:
            observed_weights_f = [None] * 3
        else:
            observed_weights_f = self.parameters.weights_f

        # All possible pairs of subjects. We will pass the combination of previous subject ID
        # (different than the current one) and current subject id as a one-hot-encode (OHE) vector to the NN so it can
        # learn how to establish different patterns for dependencies across different subjects.
        one_hot_encode_size = math.comb(self.num_subjects, 2)

        # Features + subject pair ID + bias term
        input_layer_dim_in = self.dim_value + one_hot_encode_size + 1
        input_layer_dim_out = dim_hidden_layer

        hidden_layer_dim_in = dim_hidden_layer + 1
        hidden_layer_dim_out = dim_hidden_layer

        output_layer_dim_in = dim_hidden_layer + 1
        output_layer_dim_out = self.dim_value

        input_layer = pm.Normal(f"{self.f_nn_weights_name}_in", size=(input_layer_dim_in, input_layer_dim_out),
                                observed=observed_weights_f[0])

        hidden_layers = pm.Normal(f"{self.f_nn_weights_name}_hidden", mu=0, sigma=1,
                                  size=(num_hidden_layers, hidden_layer_dim_in, hidden_layer_dim_out),
                                  observed=observed_weights_f[1])

        # There's a bug in PyMC 5.0.2 that we cannot pass an argument with more dimensions than the
        # dimension of CustomDist. To work around it, I will join the layer dimension with the input dimension for
        # the hidden layers. Inside the logp function, I will reshape the layers back to their original 3 dimensions:
        # num_layers x in_dim x out_dim, so we can perform the feed-forward step.
        hidden_layers = pm.Deterministic(f"{self.f_nn_weights_name}_hidden_reshaped", hidden_layers.reshape(
            (num_hidden_layers * hidden_layer_dim_in, hidden_layer_dim_out)))

        output_layer = pm.Normal(f"{self.f_nn_weights_name}_out", size=(output_layer_dim_in, output_layer_dim_out),
                                 observed=observed_weights_f[2])

        # Because we cannot pass a string or a function to CustomDist, we will identify a function by a number and
        # we will retrieve it's implementation in the feed-forward function.
        activation_function_number = ActivationFunction.NAME_TO_NUMBER[activation_function_name]

        return input_layer, hidden_layers, output_layer, activation_function_number

    def update_pymc_model(self, coordination: Any, prev_time_same_subject: np.ndarray,
                          prev_time_diff_subject: np.ndarray, prev_same_subject_mask: np.ndarray,
                          prev_diff_subject_mask: np.ndarray, subjects: np.ndarray, gender_map: Dict[int, int],
                          feature_dimension: str, time_dimension: str, observed_values: Optional[Any] = None,
                          num_hidden_layers_f: int = 0, activation_function_name_f: str = "linear",
                          dim_hidden_layer_f: int = 0) -> Any:

        mean, sd, mean_a0, sd_aa = self._create_random_parameters(subjects, gender_map)

        if num_hidden_layers_f > 0:
            input_layer_f, hidden_layers_f, output_layer_f, activation_function_number_f = self._create_random_weights_f(
                num_hidden_layers=num_hidden_layers_f, dim_hidden_layer=dim_hidden_layer_f,
                activation_function_name=activation_function_name_f)
        else:
            input_layer_f = []
            hidden_layers_f = []
            output_layer_f = []
            activation_function_number_f = 0

        # Fill a dictionary with an ID for each pair of subjects.
        pairs_dict = {}
        pair_id = 0
        for i in range(self.num_subjects):
            for j in range(i + 1, self.num_subjects):
                pairs_dict[f"{i}#{j}"] = pair_id
                pair_id += 1

        # Create one-hot-encode (OHE) vectors for the different pairs
        num_time_steps = len(subjects)
        pairs = np.zeros((len(pairs_dict), num_time_steps))
        for t in range(num_time_steps):
            source_subject = subjects[prev_time_diff_subject[t]]
            target_subject = subjects[t]
            pair_key = f"{min(source_subject, target_subject)}#{max(source_subject, target_subject)}"
            pair_id = pairs_dict[pair_key]

            # Mark the index as 1 to create a OHE representation for that pair at time step t.
            pairs[pair_id, t] = 1

        if self.self_dependent:
            logp_params = (mean,
                           sd,
                           coordination,
                           input_layer_f,
                           hidden_layers_f,
                           output_layer_f,
                           activation_function_number_f,
                           ptt.constant(prev_time_same_subject),
                           ptt.constant(prev_time_diff_subject),
                           ptt.constant(prev_same_subject_mask),
                           ptt.constant(prev_diff_subject_mask),
                           ptt.constant(pairs))
            logp_fn = blending_logp if self.mode == Mode.BLENDING else mixture_logp
            random_fn = blending_random if self.mode == Mode.BLENDING else mixture_random
            serialized_component = pm.DensityDist(self.uuid, *logp_params, logp=logp_fn, random=random_fn,
                                                  dims=[feature_dimension, time_dimension],
                                                  observed=observed_values)
        else:
            logp_params = (mean,
                           sd,
                           coordination,
                           input_layer_f,
                           hidden_layers_f,
                           output_layer_f,
                           activation_function_number_f,
                           ptt.constant(prev_time_diff_subject),
                           ptt.constant(prev_diff_subject_mask),
                           ptt.constant(pairs))
            logp_fn = blending_logp_no_self_dependency if self.mode == Mode.BLENDING else mixture_logp_no_self_dependency
            random_fn = blending_random_no_self_dependency if self.mode == Mode.BLENDING else mixture_random_no_self_dependency
            serialized_component = pm.DensityDist(self.uuid, *logp_params, logp=logp_fn, random=random_fn,
                                                  dims=[feature_dimension, time_dimension],
                                                  observed=observed_values)

        return serialized_component, mean_a0, sd_aa, (input_layer_f, hidden_layers_f, output_layer_f)
