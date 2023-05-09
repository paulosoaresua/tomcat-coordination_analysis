import itertools
from typing import Any, Callable, Dict, List, Optional, Tuple

from enum import Enum
import math
import numpy as np
import pymc as pm
import pytensor.tensor as ptt
from scipy.stats import norm

from coordination.common.activation_function import ActivationFunction
from coordination.common.utils import set_random_seed
from coordination.model.parametrization import Parameter, HalfNormalParameterPrior, NormalParameterPrior
from coordination.component.utils import feed_forward_logp_f, feed_forward_random_f
from coordination.component.lag import Lag


class Mode(Enum):
    BLENDING = 0
    MIXTURE = 1


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
                  prev_same_subject_mask: Any,
                  prev_diff_subject_mask: Any,
                  pairs: ptt.TensorConstant,
                  self_dependent: ptt.TensorConstant):
    # We reshape to guarantee we don't create dimensions with unknown size in case the first dimension of
    # the serialized component is one
    D = serialized_component[..., prev_time_diff_subject].reshape(serialized_component.shape)  # d x t

    if input_layer_f.shape.prod().eval() != 0:
        # Only transform the input data if a NN was specified. There's a similar check inside the feed_forward_logp_f
        # function, but we duplicate it here because if input_layer_f is empty, we want to use D in its original
        # form instead of concatenated with the pairs matrix.
        D = feed_forward_logp_f(input_data=ptt.concatenate([D, pairs], axis=0),
                                input_layer_f=input_layer_f,
                                hidden_layers_f=hidden_layers_f,
                                output_layer_f=output_layer_f,
                                activation_function_number_f=activation_function_number_f)

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
        D = sample[..., prev_time_diff_subject[t]]  # d-vector

        if len(input_layer_f) != 0:
            # Only transform the input data if a NN was specified. There's a similar check inside the feed_forward_logp_f
            # function, but we duplicate it here because if input_layer_f is empty, we want to use D in its original
            # form instead of concatenated with the pairs matrix.
            D = feed_forward_random_f(input_data=np.concatenate([D[:, None], pairs[:, t][:, None]]),
                                      input_layer_f=input_layer_f,
                                      hidden_layers_f=hidden_layers_f,
                                      output_layer_f=output_layer_f,
                                      activation=activation)[:, 0]

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

    if input_layer_f.shape.prod().eval() != 0:
        # Only transform the input data if a NN was specified. There's a similar check inside the feed_forward_logp_f
        # function, but we duplicate it here because if input_layer_f is empty, we want to use D in its original
        # form instead of concatenated with the pairs matrix.
        D = feed_forward_logp_f(input_data=ptt.concatenate([D, pairs], axis=0),
                                input_layer_f=input_layer_f,
                                hidden_layers_f=hidden_layers_f,
                                output_layer_f=output_layer_f,
                                activation_function_number_f=activation_function_number_f)

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

        if len(input_layer_f) != 0:
            # Only transform the input data if a NN was specified. There's a similar check inside the feed_forward_logp_f
            # function, but we duplicate it here because if input_layer_f is empty, we want to use D in its original
            # form instead of concatenated with the pairs matrix.
            D = feed_forward_random_f(input_data=np.concatenate([D[:, None], pairs[:, t][:, None]]),
                                      input_layer_f=input_layer_f,
                                      hidden_layers_f=hidden_layers_f,
                                      output_layer_f=output_layer_f,
                                      activation=activation)[:, 0]

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

    def __init__(self, mean_mean_a0: np.ndarray, sd_mean_a0: np.ndarray, sd_sd_aa: np.ndarray, mean_weights_f: float,
                 sd_weights_f: float):
        self.mean_a0 = Parameter(NormalParameterPrior(mean_mean_a0, sd_mean_a0))
        self.sd_aa = Parameter(HalfNormalParameterPrior(sd_sd_aa))
        self.weights_f = Parameter(NormalParameterPrior(np.array(mean_weights_f), np.array([sd_weights_f])))

    def clear_values(self):
        self.mean_a0.value = None
        self.sd_aa.value = None
        self.weights_f.value = None


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
                 share_sd_aa_across_features: bool, mode: Mode = Mode.BLENDING,
                 f: Optional[Callable] = None, mean_weights_f: float = 0, sd_weights_f: float = 1, max_lag: int = 0):

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
        self.f = f

        if max_lag > 0:
            self.lag_cpn = Lag(f"{uuid}_lag", max_lag=max_lag)
        else:
            self.lag_cpn = None

        self.parameters = SerializedComponentParameters(mean_mean_a0=mean_mean_a0,
                                                        sd_mean_a0=sd_mean_a0,
                                                        sd_sd_aa=sd_sd_aa,
                                                        mean_weights_f=mean_weights_f,
                                                        sd_weights_f=sd_weights_f)

    @property
    def parameter_names(self) -> List[str]:
        names = [
            self.mean_a0_name,
            self.sd_aa_name
        ]
        if self.lag_cpn is not None:
            names.append(self.lag_cpn.uuid)

        return names

    @property
    def mean_a0_name(self) -> str:
        return f"mean_a0_{self.uuid}"

    @property
    def sd_aa_name(self) -> str:
        return f"sd_aa_{self.uuid}"

    @property
    def f_nn_weights_name(self) -> str:
        return f"f_nn_weights_{self.uuid}"

    def clear_parameter_values(self):
        self.parameters.clear_values()
        if self.lag_cpn is not None:
            self.lag_cpn.clear_parameter_values()

    def draw_samples(self, num_series: int, time_scale_density: float,
                     coordination: np.ndarray, can_repeat_subject: bool,
                     seed: Optional[int] = None) -> SerializedComponentSamples:
        # TODO - add support to generate samples with lag. Currently, this has to be done after the samples are
        #  generated by shifting the data over time.

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
        for s in range(num_series):
            sparse_subjects = self._draw_random_subjects(num_series, coordination.shape[-1], time_scale_density,
                                                         can_repeat_subject)
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

                subject_idx_mean_a0 = samples.subjects[s][t]
                subject_idx_sd_aa = samples.subjects[s][t]
                if self.share_mean_a0_across_subjects:
                    subject_idx_mean_a0 = 0
                if self.share_sd_aa_across_subjects:
                    subject_idx_sd_aa = 0

                if samples.prev_time_same_subject[s][t] < 0:
                    # It is not only when t == 0 because the first utterance of a speaker can be later in the future.
                    # t_0 is the initial utterance of one of the subjects only.

                    mean = mean_a0[subject_idx_mean_a0]
                    sd = sd_aa[subject_idx_sd_aa]

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
                        S = mean_a0[subject_idx_mean_a0]

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
                    sd = sd_aa[subject_idx_sd_aa]

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

    def _create_random_weights_f(self, num_layers: int, dim_hidden_layer: int, activation_function_name: str):
        """
        This function creates the weights used to fit the function f(.) as random variables. Because the serialized
        component uses a CustomDist, all the arguments of the logp function we pass must be tensors. So, we cannot
        pass a list of tensors from different sizes, otherwise the program will crash when it tries to convert that
        to a single tensor. Therefore, the strategy is to have 3 sets of weights, the first one represents the weights
        in the input layer, the second will be a list of weights with the same dimensions, which represent the weights
        in the hidden layers, and the last one will be weights in the last (output) layer.
        """

        if num_layers == 0:
            return [], [], [], 0

        # Gather observations from each layer. If some weights are pre-set, we don't need to infer them.
        if self.parameters.weights_f.value is None:
            observed_weights_f = [None] * 3
        else:
            observed_weights_f = self.parameters.weights_f.value

        # All possible pairs of subjects. We will pass the combination of previous subject ID
        # (different than the current one) and current subject id as a one-hot-encode (OHE) vector to the NN so it can
        # learn how to establish different patterns for dependencies across different subjects.
        one_hot_encode_size = math.comb(self.num_subjects, 2)

        # Features + subject pair ID + bias term
        input_layer_dim_in = self.dim_value + one_hot_encode_size + 1
        input_layer_dim_out = dim_hidden_layer if num_layers > 1 else self.dim_value

        input_layer = pm.Normal(f"{self.f_nn_weights_name}_in",
                                mu=self.parameters.weights_f.prior.mean,
                                sigma=self.parameters.weights_f.prior.sd,
                                size=(input_layer_dim_in, input_layer_dim_out),
                                observed=observed_weights_f[0])

        # Input and Output layers count as 2. The remaining is hidden.
        num_hidden_layers = num_layers - 2
        if num_hidden_layers > 0:
            hidden_layer_dim_in = dim_hidden_layer + 1
            hidden_layer_dim_out = dim_hidden_layer
            hidden_layers = pm.Normal(f"{self.f_nn_weights_name}_hidden",
                                      mu=self.parameters.weights_f.prior.mean,
                                      sigma=self.parameters.weights_f.prior.sd,
                                      size=(num_hidden_layers, hidden_layer_dim_in, hidden_layer_dim_out),
                                      observed=observed_weights_f[1])

            # There's a bug in PyMC 5.0.2 that we cannot pass an argument with more dimensions than the
            # dimension of CustomDist. To work around it, I will join the layer dimension with the input dimension for
            # the hidden layers. Inside the logp function, I will reshape the layers back to their original 3 dimensions:
            # num_layers x in_dim x out_dim, so we can perform the feed-forward step.
            hidden_layers = pm.Deterministic(f"{self.f_nn_weights_name}_hidden_reshaped", hidden_layers.reshape(
                (num_hidden_layers * hidden_layer_dim_in, hidden_layer_dim_out)))
        else:
            hidden_layers = []

        if num_layers > 1:
            output_layer_dim_in = dim_hidden_layer + 1
            output_layer_dim_out = self.dim_value
            output_layer = pm.Normal(f"{self.f_nn_weights_name}_out",
                                     mu=self.parameters.weights_f.prior.mean,
                                     sigma=self.parameters.weights_f.prior.sd,
                                     size=(output_layer_dim_in, output_layer_dim_out),
                                     observed=observed_weights_f[2])
        else:
            output_layer = []

        # Because we cannot pass a string or a function to CustomDist, we will identify a function by a number and
        # we will retrieve its implementation in the feed-forward function.
        activation_function_number = ActivationFunction.NAME_TO_NUMBER[activation_function_name]

        return input_layer, hidden_layers, output_layer, activation_function_number

    def _encode_subject_pairs(self, subjects_in_time: np.ndarray, prev_time_diff_subject: np.ndarray):
        # Creates a series (over time) of pair ids in a one-hot-encode representation. This will be used in f(.) to
        # identify the influencer and influencee such that the correct transformation can be learned for each distinct
        # pair interaction.

        # First we construct a dictionary to attribute an ID to each distinct pair of subjects.
        pair_id_to_idx = {}
        pair_idx = 0
        for influencer, influencee in itertools.combinations(range(self.num_subjects), 2):
            pair_id_to_idx[f"{influencer}#{influencee}"] = pair_idx
            pair_id_to_idx[f"{influencee}#{influencer}"] = pair_idx
            pair_idx += 1

        # Now for each time step we check influencer and influencee, and use the pair ID to create a one-hot-encoded
        # vector for the interacting pair of subjects.
        num_pairs = math.comb(self.num_subjects, 2)
        num_time_steps = len(subjects_in_time)
        encoded_pairs = np.zeros((num_pairs, num_time_steps))
        for t in range(num_time_steps):
            # if prev_time_diff_subject < 0, there's no dependency with another subject at the time step.
            if prev_time_diff_subject[t] >= 0:
                influencer = subjects_in_time[prev_time_diff_subject[t]]
                influencee = subjects_in_time[t]
                pair_id = f"{influencer}#{influencee}"
                pair_idx = pair_id_to_idx[pair_id]
                encoded_pairs[pair_idx, t] = 1

        return encoded_pairs

    def _create_lag_table(self, subjects_in_time: np.ndarray, prev_time_diff_subject: np.ndarray):
        # A lag table contain time step indices of dependencies on different subjects over time for different
        # lag values.
        # For instance, lag_table[k + l, t] gives us the index of the value from a previous subject different than the
        # subject in time t, considering a lag of l for a maximum lag of k.
        # We will use this table to dynamically select prev_time_diff_subject for different lag values.
        num_time_steps = len(subjects_in_time)
        lag_table = np.full(shape=(2 * self.lag_cpn.max_lag + 1, num_time_steps), fill_value=-1)
        subject_times = [[t for t, _ in enumerate(subjects_in_time) if subjects_in_time[t] == s] for s in
                         range(self.num_subjects)]
        for t in range(num_time_steps):
            if prev_time_diff_subject[t] >= 0:
                # Previous subject different than the current one
                influencer = subjects_in_time[prev_time_diff_subject[t]]

                # In the array containing all time steps when the influencer had observations, find
                # the index of the time step associated with prev_time_diff_subject[t]. This will be the index for lag
                # zero.
                lag_zero_idx = np.searchsorted(subject_times[influencer], prev_time_diff_subject[t])

                # How much further back we have values for the influencer
                min_local_lag = -min(lag_zero_idx, self.lag_cpn.max_lag)

                # How much further ahead we have values for the influencer
                max_local_lag = min(len(subject_times[influencer]) - lag_zero_idx - 1, self.lag_cpn.max_lag)

                # The rows range from -K to K, where K is the max lag size. So, to access an element with lag l, we do
                # lat_table[K + l, :].
                lb = min_local_lag + self.lag_cpn.max_lag
                ub = max_local_lag + self.lag_cpn.max_lag + 1
                entry = subject_times[influencer][lag_zero_idx + min_local_lag:lag_zero_idx + max_local_lag + 1]
                lag_table[lb:ub, t] = entry

        return lag_table

    def update_pymc_model(self, coordination: Any, prev_time_same_subject: np.ndarray,
                          prev_time_diff_subject: np.ndarray, prev_same_subject_mask: np.ndarray,
                          prev_diff_subject_mask: np.ndarray, subjects: np.ndarray, feature_dimension: str,
                          time_dimension: str, observed_values: Optional[Any] = None, mean_a0: Optional[Any] = None,
                          sd_aa: Optional[Any] = None, num_layers_f: int = 0,
                          activation_function_name_f: str = "linear", dim_hidden_layer_f: int = 0) -> Any:

        mean_a0, sd_aa = self._create_random_parameters(subjects, mean_a0, sd_aa)

        input_layer_f, hidden_layers_f, output_layer_f, activation_function_number_f = self._create_random_weights_f(
            num_layers=num_layers_f, dim_hidden_layer=dim_hidden_layer_f,
            activation_function_name=activation_function_name_f)

        encoded_pairs = self._encode_subject_pairs(subjects, prev_time_diff_subject)

        if self.lag_cpn is not None:
            influencers = subjects[prev_time_diff_subject]
            influencees = subjects

            lag = self.lag_cpn.update_pymc_model(num_lags=self.num_subjects)
            lag_influencers = lag[influencers]
            lag_influencees = lag[influencees]

            # We need to clip the lag because invalid samples can be generated by the MCMC procedure on PyMC.
            dlags = ptt.clip(lag_influencees - lag_influencers, -self.lag_cpn.max_lag, self.lag_cpn.max_lag)

            lag_table = self._create_lag_table(subjects, prev_time_diff_subject)
            lag_idx = self.lag_cpn.max_lag + dlags
            prev_time_diff_subject = ptt.take_along_axis(lag_table, lag_idx[None, :], 0)[0]

            # We multiply at mask at lag zero because that's the ground truth for the time steps when we have
            # dependencies on influencers. This is because -1 is a valid index so, we need to make sure we are not
            # introducing spurious dependencies when we update the dependencies from the lag table.
            mask_at_lag_zero = ptt.where(lag_table[self.lag_cpn.max_lag] >= 0, 1, 0)
            prev_diff_subject_mask = ptt.where(prev_time_diff_subject >= 0, 1, 0) * mask_at_lag_zero

        logp_params = (mean_a0,
                       sd_aa,
                       coordination,
                       input_layer_f,
                       hidden_layers_f,
                       output_layer_f,
                       activation_function_number_f,
                       prev_time_same_subject,
                       prev_time_diff_subject,
                       prev_same_subject_mask,
                       prev_diff_subject_mask,
                       encoded_pairs,
                       np.array(self.self_dependent))
        logp_fn = blending_logp if self.mode == Mode.BLENDING else mixture_logp
        random_fn = blending_random if self.mode == Mode.BLENDING else mixture_random
        serialized_component = pm.DensityDist(self.uuid, *logp_params, logp=logp_fn, random=random_fn,
                                              dims=[feature_dimension, time_dimension],
                                              observed=observed_values)

        return serialized_component, mean_a0, sd_aa, (input_layer_f, hidden_layers_f, output_layer_f)
