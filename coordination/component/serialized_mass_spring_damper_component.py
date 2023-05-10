
from coordination.component.serialized_component import SerializedComponent


class SerializedMassSpringDamperComponent(MixtureComponent):

    def __init__(self, uuid: str,
                 num_subjects: int,
                 spring_constant: float,
                 mass: float,
                 damping_coefficient: float,
                 dt: float,
                 self_dependent: bool,
                 mean_mean_a0: np.ndarray,
                 sd_mean_a0: np.ndarray,
                 sd_sd_aa: np.ndarray,
                 a_mixture_weights: np.ndarray,
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
                         a_mixture_weights=a_mixture_weights,
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
                                   sampled_influencers: np.ndarray, mean_a0: np.ndarray,
                                   sd_aa: np.ndarray) -> np.ndarray:
        num_series = sampled_coordination.shape[0]
        num_time_steps = len(time_steps_in_coordination_scale)

        if self.lag_cpn is not None:
            max_lag = self.lag_cpn.max_lag
            extra_time_steps = 2 * self.lag_cpn.max_lag
        else:
            max_lag = 0
            extra_time_steps = 0

        values = np.zeros((num_series, self.num_subjects, self.dim_value, num_time_steps + extra_time_steps))

        for t in range(max_lag, -1, -1):
            if t == max_lag:
                values[..., t] = norm(loc=mean_a0, scale=sd_aa).rvs(
                    size=(num_series, self.num_subjects, self.dim_value))
            else:
                # Backwards dynamics. This is to guarantee when we apply the lag to each subject, we have mean_a0 as
                # the initial value
                mean = np.einsum("ij,klj->kli", self.F_inv, values[..., t + 1])
                values[..., t] = norm(loc=mean, scale=sd_aa).rvs()

        for t in range(max_lag + 1, num_time_steps + extra_time_steps):
            if t >= num_time_steps + extra_time_steps - max_lag:
                # There's no coordination for these time steps. They are extra and we use them to fill in the gaps
                # when we roll the samples by each subject's lag in the end.
                mean = np.einsum("ij,klj->kli", self.F, values[..., t - 1])
            else:
                C = sampled_coordination[:, time_steps_in_coordination_scale[t - max_lag]][:, None]

                P = values[..., t - 1]

                if self.f is not None:
                    D = self.f(values[..., t - 1], sampled_influencers[..., t - max_lag])
                else:
                    D = P

                D = D[:, sampled_influencers[..., t - max_lag]][0]

                if self.self_dependent:
                    S = P
                else:
                    S = mean_a0

                blended_state = (D - S) * C + S

                mean = np.einsum("ij,klj->kli", self.F, blended_state)

            # Add some noise
            values[..., t] = norm(loc=mean, scale=sd_aa).rvs()

        # Apply lags if any
        if self.lag_cpn is not None:
            truncated_values_per_subject = []
            for subject in range(self.num_subjects):
                lag = -self.lag_cpn.parameters.lag.value[subject] # lag contains the correction, so we use -lag
                truncated_values_per_subject.append(values[:, subject, :, (max_lag + lag):(num_time_steps + max_lag + lag)][:, None, :, :])

            values = np.concatenate(truncated_values_per_subject, axis=1)

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