import itertools
from typing import Any, Callable, Dict, List, Optional, Tuple

from enum import Enum
import math
import numpy as np
import pymc as pm
import pytensor.tensor as ptt
from scipy.linalg import expm
from scipy.stats import norm

from coordination.common.activation_function import ActivationFunction
from coordination.common.utils import set_random_seed
from coordination.model.parametrization import Parameter, HalfNormalParameterPrior, NormalParameterPrior
from coordination.component.utils import feed_forward_logp_f, feed_forward_random_f
from coordination.component.lag import Lag

from coordination.component.serialized_component import SerializedComponent


class SerializedMassSpringDamperComponent(SerializedComponent):

    def __init__(self,uuid: str,
                 num_subjects: int,
                 spring_constant: float,
                 mass: float,
                 damping_coefficient: float,
                 dt: float,
                 self_dependent: bool,
                 mean_mean_a0: np.ndarray,
                 sd_mean_a0: np.ndarray,
                 sd_sd_aa: np.ndarray,
                 share_mean_a0_across_subjects: bool,
                 share_mean_a0_across_features: bool,
                 share_sd_aa_across_subjects: bool,
                 share_sd_aa_across_features: bool,
                 f: Optional[Callable] = None,
                 mean_weights_f: float = 0,
                 sd_weights_f: float = 1,
                 max_lag: int = 0):
        """
        Generates a time series of latent states formed by position and velocity in a mass-spring-damper system. We do
        not consider external force in this implementation but it can be easily added if necessary.
        """
        super().__init__(uuid=uuid,
                         num_subjects=num_subjects,
                         dim_value=2,  # 2 dimensions: position and velocity
                         self_dependent=self_dependent,
                         mean_mean_a0=mean_mean_a0,
                         sd_mean_a0=sd_mean_a0,
                         sd_sd_aa=sd_sd_aa,
                         share_mean_a0_across_subjects=share_mean_a0_across_subjects,
                         share_mean_a0_across_features=share_mean_a0_across_features,
                         share_sd_aa_across_subjects=share_sd_aa_across_subjects,
                         share_sd_aa_across_features=share_sd_aa_across_features,
                         f=f,
                         mean_weights_f=mean_weights_f,
                         sd_weights_f=sd_weights_f,
                         max_lag=max_lag)

        self.spring_constant = spring_constant
        self.mass = mass
        self.damping_coefficient = damping_coefficient
        self.dt = dt  # size of the time step

        # Systems dynamics matrix
        A = np.array([
            [0, 1],
            [-self.spring_constant / self.mass, -self.damping_coefficient / self.mass]
        ])
        self.F = expm(A * self.dt)  # Fundamental matrix
        self.F_inv = expm(-A * self.dt)  # Fundamental matrix inverse to estimate backward dynamics

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

            sd = sd_aa[subject_idx_sd_aa]

            if prev_time_same_subject[t] < 0:
                # It is not only when t == 0 because the first utterance of a speaker can be later in the future.
                # t_0 is the initial utterance of one of the subjects only.
                mean = mean_a0[subject_idx_mean_a0]

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

                if self.f is not None:
                    source_subject = subjects_in_time[prev_time_diff_subject[t]]
                    target_subject = subjects_in_time[t]

                    D = self.f(D, source_subject, target_subject)

                blended_state = (D - S) * C * prev_diff_mask + S

                mean = np.dot(self.F[subjects_in_time[t]], blended_state[:, None])

                values[:, t] = norm(loc=mean, scale=sd).rvs()

        return values

    def update_pymc_model(self,
                          coordination: Any,
                          subject_dimension: str,
                          feature_dimension: str,
                          time_dimension: str,
                          num_time_steps: int,
                          observed_values: Optional[Any] = None,
                          mean_a0: Optional[Any] = None,
                          sd_aa: Optional[Any] = None,
                          mixture_weights: Optional[Any] = None,
                          num_layers_f: int = 0,
                          activation_function_name_f: str = "linear",
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
                       np.array(self.self_dependent),
                       self.F_inv)
        # random_fn = partial(mixture_random, num_subjects=self.num_subjects, dim_value=self.dim_value)
        mixture_component = pm.CustomDist(self.uuid, *logp_params, logp=mixture_logp,
                                          # random=random_fn,
                                          dims=[subject_dimension, feature_dimension, time_dimension],
                                          observed=observed_values)

        # mixture_logp(observed_values, *logp_params)

        return mixture_component, mean_a0, sd_aa, mixture_weights