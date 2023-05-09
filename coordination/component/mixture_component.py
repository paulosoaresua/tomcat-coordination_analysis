from typing import Any, Callable, List, Optional, Tuple

from functools import partial
import numpy as np
import pymc as pm
import pytensor.tensor as ptt
from scipy.stats import norm

from coordination.common.activation_function import ActivationFunction
from coordination.common.utils import set_random_seed
from coordination.model.parametrization import Parameter, HalfNormalParameterPrior, DirichletParameterPrior, \
    NormalParameterPrior
from coordination.component.utils import feed_forward_logp_f, feed_forward_random_f
from coordination.component.lag import Lag


def mixture_logp(mixture_component: Any,
                 initial_mean: Any,
                 sigma: Any,
                 mixture_weights: Any,
                 coordination: Any,
                 input_layer_f: Any,
                 hidden_layers_f: Any,
                 output_layer_f: Any,
                 activation_function_number_f: ptt.TensorConstant,
                 expander_aux_mask_matrix: ptt.TensorConstant,
                 prev_time_diff_subject: Any,
                 prev_diff_subject_mask: Any,
                 self_dependent: ptt.TensorConstant):
    num_subjects = mixture_component.shape[0]
    num_features = mixture_component.shape[1]

    # Log probability due to the initial time step in the component's scale.
    total_logp = pm.logp(pm.Normal.dist(mu=initial_mean, sigma=sigma, shape=(num_subjects, num_features)),
                         mixture_component[..., 0]).sum()

    # Fit a function f(.) to correct anti-symmetry.
    X = feed_forward_logp_f(input_data=mixture_component.reshape(
        (mixture_component.shape[0] * mixture_component.shape[1], mixture_component.shape[2])),
        input_layer_f=input_layer_f,
        hidden_layers_f=hidden_layers_f,
        output_layer_f=output_layer_f,
        activation_function_number_f=activation_function_number_f).reshape(mixture_component.shape)

    # D contains the values from other individuals for each individual
    D = ptt.tensordot(expander_aux_mask_matrix, X, axes=(1, 0))  # s * (s-1) x d x t

    # Discard last time step because it is not previous to any other time step.
    # D = pt.printing.Print("D")(ptt.take_along_axis(D, prev_time_diff_subject[:, None, :], axis=2)[..., 1:])
    D = ptt.take_along_axis(D, prev_time_diff_subject[:, None, :], axis=2)[..., 1:]

    # Current values from each subject. We extend S and point such that they match the dimensions of D.
    point_extended = ptt.repeat(mixture_component[..., 1:], repeats=(num_subjects - 1), axis=0)

    if self_dependent.eval():
        # Previous values from every subject
        P = mixture_component[..., :-1]  # s x d x t-1

        # Previous values from the same subjects
        # S_extended = pt.printing.Print("S")(ptt.repeat(P, repeats=(num_subjects - 1), axis=0))
        S_extended = ptt.repeat(P, repeats=(num_subjects - 1), axis=0)
    else:
        # Fixed value given by the initial mean for each subject. No self-dependency.
        S_extended = ptt.repeat(initial_mean[:, :, None], repeats=(num_subjects - 1), axis=0)

    # The mask will zero out dependencies on D if we have shifts caused by latent lags. In that case, we cannot infer
    # coordination if the values do not exist on all the subjects because of gaps introduced by the shift. So we can
    # only infer the next value of the latent value from its previous one on the same subject,
    C = coordination[None, None, 1:]  # 1 x 1 x t-1
    mean = (D - S_extended) * C * prev_diff_subject_mask[:, None, 1:] + S_extended

    sd = ptt.repeat(sigma, repeats=(num_subjects - 1), axis=0)[:, :, None]

    logp_extended = pm.logp(pm.Normal.dist(mu=mean, sigma=sd, shape=D.shape), point_extended)
    logp_tmp = logp_extended.reshape((num_subjects, num_subjects - 1, num_features, logp_extended.shape[-1]))
    total_logp += pm.math.logsumexp(logp_tmp + pm.math.log(mixture_weights[:, :, None, None]), axis=1).sum()

    return total_logp


def mixture_random(initial_mean: np.ndarray,
                   sigma: np.ndarray,
                   mixture_weights: np.ndarray,
                   coordination: np.ndarray,
                   input_layer_f: np.ndarray,
                   hidden_layers_f: np.ndarray,
                   output_layer_f: np.ndarray,
                   activation_function_number_f: int,
                   expander_aux_mask_matrix: np.ndarray,
                   prev_time_diff_subject: Any,
                   prev_diff_subject_mask: Any,
                   self_dependent: bool,
                   num_subjects: int,
                   dim_value: int,
                   rng: Optional[np.random.Generator] = None,
                   size: Optional[Tuple[int]] = None) -> np.ndarray:
    num_time_steps = coordination.shape[-1]

    noise = rng.normal(loc=0, scale=1, size=size) * sigma[:, :, None]

    # We sample the influencers in each time step using the mixture weights
    influencers = []
    for subject in range(num_subjects):
        probs = np.insert(mixture_weights[subject], subject, 0)
        influencer = rng.choice(a=np.arange(num_subjects), p=probs, size=num_time_steps)
        # We will use the influencer to index a matrix with 6 columns. One for each pair influencer -> influenced
        influencers.append(subject * (num_subjects - 1) + np.minimum(influencer, num_subjects - 2))
    influencers = np.array(influencers)

    sample = np.zeros_like(noise)
    prior_sample = rng.normal(loc=initial_mean, scale=sigma, size=(num_subjects, dim_value))
    sample[..., 0] = prior_sample

    # TODO - Add treatment for lag
    activation = ActivationFunction.from_numpy_number(activation_function_number_f)
    for t in np.arange(1, num_time_steps):
        D = np.einsum("ij,jlk->ilk", expander_aux_mask_matrix, sample[..., t - 1][..., None])  # s * (s-1) x d x 1

        # Previous sample from a different individual
        D = feed_forward_random_f(input_data=D.reshape((D.shape[0] * D.shape[1], D.shape[2])),
                                  input_layer_f=input_layer_f,
                                  hidden_layers_f=hidden_layers_f,
                                  output_layer_f=output_layer_f,
                                  activation=activation).reshape(D.shape)[..., 0]  # s * (s-1) x d x 1

        D = D[influencers[..., t]]  # s x d

        # Previous sample from the same individual
        if self_dependent:
            S = sample[..., t - 1]
        else:
            S = initial_mean

        mean = ((D - S) * coordination[t] + S)

        transition_sample = rng.normal(loc=mean, scale=sigma)

        sample[..., t] = transition_sample

    return sample + noise


class MixtureComponentParameters:

    def __init__(self, mean_mean_a0: np.ndarray, sd_mean_a0: np.ndarray, sd_sd_aa: np.ndarray,
                 a_mixture_weights: np.ndarray, mean_weights_f: float, sd_weights_f: float):
        self.mean_a0 = Parameter(NormalParameterPrior(mean_mean_a0, sd_mean_a0))
        self.sd_aa = Parameter(HalfNormalParameterPrior(sd_sd_aa))
        self.mixture_weights = Parameter(DirichletParameterPrior(a_mixture_weights))
        self.weights_f = Parameter(NormalParameterPrior(np.array(mean_weights_f), np.array([sd_weights_f])))

    def clear_values(self):
        self.mean_a0.value = None
        self.sd_aa.value = None
        self.mixture_weights.value = None
        self.weights_f.value = None


class MixtureComponentSamples:

    def __init__(self):
        self.values = np.array([])

        # For each time step in the component's scale, it contains the time step in the coordination scale
        self.time_steps_in_coordination_scale = np.array([])

    @property
    def num_time_steps(self):
        return self.values.shape[-1]


class MixtureComponent:

    def __init__(self, uuid: str, num_subjects: int, dim_value: int, self_dependent: bool, mean_mean_a0: np.ndarray,
                 sd_mean_a0: np.ndarray, sd_sd_aa: np.ndarray, a_mixture_weights: np.ndarray,
                 share_mean_a0_across_subjects: bool, share_mean_a0_across_features: bool,
                 share_sd_aa_across_subjects: bool, share_sd_aa_across_features: bool, f: Optional[Callable] = None,
                 mean_weights_f: float = 0, sd_weights_f: float = 1, max_lag: int = 0):

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

        assert (num_subjects, num_subjects - 1) == a_mixture_weights.shape

        self.uuid = uuid
        self.num_subjects = num_subjects
        self.dim_value = dim_value
        self.self_dependent = self_dependent
        self.share_mean_a0_across_subjects = share_mean_a0_across_subjects
        self.share_mean_a0_across_features = share_mean_a0_across_features
        self.share_sd_aa_across_subjects = share_sd_aa_across_subjects
        self.share_sd_aa_across_features = share_sd_aa_across_features
        self.f = f

        if max_lag > 0:
            self.lag_cpn = Lag(f"{uuid}_lag", max_lag=max_lag)
        else:
            self.lag_cpn = None

        self.parameters = MixtureComponentParameters(mean_mean_a0=mean_mean_a0,
                                                     sd_mean_a0=sd_mean_a0,
                                                     sd_sd_aa=sd_sd_aa,
                                                     a_mixture_weights=a_mixture_weights,
                                                     mean_weights_f=mean_weights_f,
                                                     sd_weights_f=sd_weights_f)

    @property
    def parameter_names(self) -> List[str]:
        names = [
            self.mean_a0_name,
            self.sd_aa_name,
            self.mixture_weights_name,
            f"{self.f_nn_weights_name}_in",
            f"{self.f_nn_weights_name}_hidden",
            f"{self.f_nn_weights_name}_out"
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
    def mixture_weights_name(self) -> str:
        return f"mixture_weights_{self.uuid}"

    @property
    def f_nn_weights_name(self) -> str:
        return f"f_nn_weights_{self.uuid}"

    def clear_parameter_values(self):
        self.parameters.clear_values()
        if self.lag_cpn is not None:
            self.lag_cpn.clear_parameter_values()

    def draw_samples(self, num_series: int, relative_frequency: float,
                     coordination: np.ndarray, seed: Optional[int] = None) -> MixtureComponentSamples:
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

        assert relative_frequency >= 1
        assert (self.num_subjects, self.num_subjects - 1) == self.parameters.mixture_weights.value.shape

        set_random_seed(seed)

        samples = MixtureComponentSamples()

        # Number of time steps in the component's scale
        num_time_steps_in_cpn_scale = int(coordination.shape[-1] / relative_frequency)
        samples.time_steps_in_coordination_scale = (np.arange(num_time_steps_in_cpn_scale) * relative_frequency).astype(
            int)

        # Sample influencers in each time step
        influencers = []
        for subject in range(self.num_subjects):
            probs = np.insert(self.parameters.mixture_weights.value[subject], subject, 0)
            influencers.append(
                np.random.choice(a=np.arange(self.num_subjects), p=probs,
                                 size=(num_series, num_time_steps_in_cpn_scale)))
        influencers = np.array(influencers).swapaxes(0, 1)

        if self.share_mean_a0_across_subjects:
            mean_a0 = self.parameters.mean_a0.value[None, None, :]
        else:
            mean_a0 = self.parameters.mean_a0.value[None, :]

        if self.share_sd_aa_across_subjects:
            sd_aa = self.parameters.sd_aa.value[None, None, :]
        else:
            sd_aa = self.parameters.sd_aa.value[None, :]

        # draw values from the system dynamics. The default model generates samples by following a Gaussian random walk
        # with blended values from different subjects according to the coordination levels over time. Child classes
        # can implement their own dynamics, like spring-mass-damping systems for instance.
        samples.values = self._draw_from_system_dynamics(
            time_steps_in_coordination_scale=samples.time_steps_in_coordination_scale,
            sampled_coordination=coordination,
            sampled_influencers=influencers,
            mean_a0=mean_a0,
            sd_aa=sd_aa)

        return samples

    def _draw_from_system_dynamics(self, time_steps_in_coordination_scale: np.ndarray, sampled_coordination: np.ndarray,
                                   sampled_influencers: np.ndarray, mean_a0: np.ndarray,
                                   sd_aa: np.ndarray) -> np.ndarray:

        num_series = sampled_coordination.shape[0]
        num_time_steps = len(time_steps_in_coordination_scale)
        values = np.zeros((num_series, self.num_subjects, self.dim_value, num_time_steps))

        for t in range(num_time_steps):
            if t == 0:
                values[..., 0] = norm(loc=mean_a0, scale=sd_aa).rvs(
                    size=(num_series, self.num_subjects, self.dim_value))
            else:
                C = sampled_coordination[:, time_steps_in_coordination_scale[t]][:, None]
                P = values[..., t - 1]

                if self.f is not None:
                    D = self.f(values[..., t - 1], sampled_influencers[..., t])
                else:
                    D = P

                D = D[:, sampled_influencers[..., t]][0]

                if self.self_dependent:
                    S = P
                else:
                    S = mean_a0

                mean = (D - S) * C + S

                values[..., t] = norm(loc=mean, scale=sd_aa).rvs()

        return values

    def _create_random_parameters(self, mean_a0: Optional[Any] = None, sd_aa: Optional[Any] = None,
                                  mixture_weights: Optional[Any] = None):
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

        if mixture_weights is None:
            mixture_weights = pm.Dirichlet(name=self.mixture_weights_name,
                                           a=self.parameters.mixture_weights.prior.a,
                                           observed=self.parameters.mixture_weights.value)

        return mean_a0, sd_aa, mixture_weights

    def _create_random_weights_f(self, num_layers: int, dim_hidden_layer: int, activation_function_name: str):
        """
        This function creates the weights used to fit the function f(.) as random variables. Because the mixture
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

        # Features * number of subjects + bias term
        input_layer_dim_in = self.dim_value * self.num_subjects + 1
        input_layer_dim_out = dim_hidden_layer if num_layers > 1 else self.dim_value * self.num_subjects

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
            output_layer_dim_out = self.dim_value * self.num_subjects
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

    @staticmethod
    def _create_lag_mask(num_time_steps: Any, lag: Any):
        # There should be one lag per subject. After we shift the samples applying the lags, we create a
        # mask to indicate the portion of the data that intersects. Coordination can only be inferred on that part.
        #
        # e.g.    Subject A: |-----------|             Subject A: |-----------| (-2 lag)
        #         Subject B: |-----------|  after lag  Subject B:       |-----------| (+4 lag)
        #         Subject C: |-----------|             Subject C:   |-----------|
        #                                                               |-----| = Intersection. We can only infer
        #                                                                         coordination on this part.
        #
        # The mask will be passed to the logp function to prevent gradient propagation in parts where coordination
        # should not interfere.

        max_diff = ptt.max(lag) - ptt.min(lag)

        # If all lags are negative, we start from the first time step because we will shift all values to the left.
        # Similar reasoning is used for positive lags and the upper_idx.
        lower_idx = ptt.maximum(max_diff, 0)
        upper_idx = num_time_steps + ptt.minimum(-max_diff, 0)  # Index not included.

        lag_mask = ptt.zeros((1, num_time_steps))
        lag_mask = ptt.set_subtensor(lag_mask[..., lower_idx:upper_idx], 1)

        return ptt.cast(lag_mask, "int32")

    def update_pymc_model(self, coordination: Any, subject_dimension: str, feature_dimension: str, time_dimension: str,
                          num_time_steps: int, observed_values: Optional[Any] = None, mean_a0: Optional[Any] = None,
                          sd_aa: Optional[Any] = None, mixture_weights: Optional[Any] = None,
                          num_layers_f: int = 0, activation_function_name_f: str = "linear",
                          dim_hidden_layer_f: int = 0) -> Any:

        mean_a0, sd_aa, mixture_weights = self._create_random_parameters(mean_a0, sd_aa, mixture_weights)

        input_layer_f, hidden_layers_f, output_layer_f, activation_function_number_f = self._create_random_weights_f(
            num_layers=num_layers_f, dim_hidden_layer=dim_hidden_layer_f,
            activation_function_name=activation_function_name_f)

        # Auxiliary matricx to compute logp in a vectorized manner without having to loop over the individuals.
        # The expander matrix transforms a s x f x t tensor to a s * (s-1) x f x t tensor where the rows contain
        # values of other subjects for each subject in the set.
        expander_aux_mask_matrix = []
        for subject in range(self.num_subjects):
            expander_aux_mask_matrix.append(np.delete(np.eye(self.num_subjects), subject, axis=0))
            aux = np.zeros((self.num_subjects, self.num_subjects - 1))
            aux[subject] = 1

        expander_aux_mask_matrix = np.concatenate(expander_aux_mask_matrix, axis=0)

        # We fit one lag per pair, so the number of lags is C_s_2, where s is the number of subjects.
        if self.lag_cpn is None:
            lag_mask = np.ones((1, num_time_steps), dtype=int)
            prev_time_diff_subject = ptt.arange(num_time_steps)[None, :].repeat(
                self.num_subjects * (self.num_subjects - 1), axis=0) - 1
        else:
            # We fix a lag zero for the first subject and move the others relative to the fixed one.
            # lag = ptt.concatenate([ptt.zeros(1, dtype=int), self.lag_cpn.update_pymc_model(self.num_subjects - 1)])
            lag = self.lag_cpn.update_pymc_model(self.num_subjects)

            # The difference between the influencee and influencer's lags will tell us which time step we need to look
            # at the influencer for each time step in the influencee.
            influencer_lag = ptt.dot(expander_aux_mask_matrix, lag)
            influencee_lag = ptt.repeat(lag, repeats=(self.num_subjects - 1))
            dlag = ptt.cast(influencee_lag - influencer_lag, "int32")

            lag_mask = MixtureComponent._create_lag_mask(num_time_steps, lag)

            prev_time_diff_subject = ptt.arange(num_time_steps, dtype=int)[None, :] + dlag[:, None] - 1
            prev_time_diff_subject *= lag_mask

        logp_params = (mean_a0,
                       sd_aa,
                       mixture_weights,
                       coordination,
                       input_layer_f,
                       hidden_layers_f,
                       output_layer_f,
                       activation_function_number_f,
                       expander_aux_mask_matrix,
                       prev_time_diff_subject,
                       lag_mask,
                       np.array(self.self_dependent))
        random_fn = partial(mixture_random, num_subjects=self.num_subjects, dim_value=self.dim_value)
        mixture_component = pm.CustomDist(self.uuid, *logp_params, logp=mixture_logp,
                                          random=random_fn,
                                          dims=[subject_dimension, feature_dimension, time_dimension],
                                          observed=observed_values)

        return mixture_component, mean_a0, sd_aa, mixture_weights
