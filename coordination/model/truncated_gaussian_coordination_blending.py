from typing import Callable, List

import numpy as np

from coordination.common.dataset import Dataset, SeriesData
from coordination.model.coordination_model import CoordinationModel
from coordination.model.lds import apply_marginal_property, apply_conditional_property, pdf_projection


class TruncatedGaussianCoordinationBlendingInference(CoordinationModel):

    def __init__(self,
                 mean_prior_coordination: float,
                 std_prior_coordination: float,
                 std_coordination_drifting: float,
                 mean_prior_vocalics: np.array,
                 std_prior_vocalics: np.array,
                 std_coordinated_vocalics: np.ndarray,
                 f: Callable = lambda x, s: x,
                 fix_coordination_on_second_half: bool = True,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._mean_prior_coordination = mean_prior_coordination
        self._std_prior_coordination = std_prior_coordination
        self._std_coordination_drifting = std_coordination_drifting
        self._mean_prior_vocalics = mean_prior_vocalics
        self._std_prior_vocalics = std_prior_vocalics
        self._std_coordinated_vocalics = std_coordinated_vocalics
        self._f = f
        self._fix_coordination_on_second_half = fix_coordination_on_second_half

    def fit(self, input_features: Dataset, num_particles: int = 0, num_iter: int = 0, discard_first: int = 0, *args,
            **kwargs):
        # MCMC to train parameters? We start by choosing with cross validation instead.
        return self

    def predict(self, input_features: Dataset, apply_rts_smooth: bool = True, *args, **kwargs) -> List[np.ndarray]:
        if input_features.num_trials > 0:
            assert len(self._mean_prior_vocalics) == input_features.series[0].vocalics.num_features
            assert len(self._std_prior_vocalics) == input_features.series[0].vocalics.num_features
            assert len(self._std_coordinated_vocalics) == input_features.series[0].vocalics.num_features

        result = []
        for d in range(input_features.num_trials):
            series = input_features.series[d]

            params = self._kalman_filter(series)

            if apply_rts_smooth:
                params = self._rts_smoother(params)

            result.append(params)

        return result

    def _kalman_filter(self, series: SeriesData) -> np.ndarray:
        var_C = self._std_coordination_drifting ** 2
        var_V = self._std_coordinated_vocalics ** 2

        M = int(series.num_time_steps / 2)
        num_time_steps = M + 1 if self._fix_coordination_on_second_half else series.num_time_steps

        params = np.zeros((2, num_time_steps))
        params[:, 0] = np.array([self._mean_prior_coordination, self._std_prior_coordination ** 2])
        mean, var = params[:, 0]
        for t in range(1, series.num_time_steps):
            if t < M or not self._fix_coordination_on_second_half:
                # No coordination drift from M beyond if coordination in the second half is fixed
                m_previous, v_previous = params[:, t - 1]

                """
                Coordination moves because of drifting:
                N(C_t | m_t, s_t) sim Integral [N(C_t | C_{t-1}, var_C) * N(C_{t-1}, m_{t-1}, v_{t-1})] dC_{t-1} 
                                    = N(C_t | m_{t-1}, v_{t-1} + var_C)
    
                """
                mean, var = apply_marginal_property(1, 0, var_C, m_previous, v_previous)
                mean, var = pdf_projection(mean, var, 0, 1)

            """
            Coordination moves because of vocalics:

            N(C_t | m_t, s_t) sim Integral [
                                    N(C_t | C_{t-1}, var_C) * N(C_{t-1}, m_{t-1}, v_{t-1}) *
                                    p(A_t | A_{t-1}, B_{t-1}, C_t, f) * p(A_t | A_{t-1}, B_{t-1}, C_t, f)
                                    ] dC_{t-1}

                                = p(A_t | A_{t-1}, B_{t-1}, C_t, f) * p(A_t | A_{t-1}, B_{t-1}, C_t, f) *
                                    Integral [N(C_t | C_{t-1}, var_C) * N(C_{t-1}, m_{t-1}, v_{t-1})] dC_{t-1} 

                                = N(A_t | D_{t-1} * C_t + A_{t-1}, var_AB) *
                                    N(B_t | -D_{t-1} * C_t + B_{t-1}, var_BA) *
                                    N(C_t | m_{t-1}, v_{t-1} + var_C) -- Obtained by the integral property previously

                                We can apply the conditional property recursively to obtain the mean and 
                                variances of the final distribution. We must apply the conditional property
                                to each dimension of A and B.
            """

            A_t = self._f(series.vocalics.values[:, t], 0)
            A_prev = None if series.vocalics.previous_from_self[t] is None else self._f(
                series.vocalics.values[:, series.vocalics.previous_from_self[t]], 0)
            B_prev = None if series.vocalics.previous_from_other[t] is None else self._f(
                series.vocalics.values[:, series.vocalics.previous_from_other[t]], 1)

            if series.vocalics.mask[t] == 1 and B_prev is not None:
                if A_prev is None:
                    A_prev = self._mean_prior_vocalics
                D = B_prev - A_prev
                for i in range(series.vocalics.num_features):
                    mean, var = apply_conditional_property(A_t[i], D[i], A_prev[i], var_V[i], mean, var)

            if self._fix_coordination_on_second_half:
                params[:, min(t, M)] = [mean, var]
            else:
                params[:, t] = [mean, var]

        return params

    def _rts_smoother(self, filter_params: np.ndarray) -> np.ndarray:
        var_C = self._std_coordination_drifting ** 2

        params = np.zeros((2, filter_params.shape[1]))  # Coordination mean and variance over time

        last_time_step = filter_params.shape[1] - 1
        params[:, last_time_step] = filter_params[:, -1]
        for t in range(last_time_step - 1, -1, -1):
            K = (filter_params[1, t] ** 2) / (filter_params[1, t] ** 2 + var_C)

            mean_filter, var_filter = filter_params[:, t]
            mean, var = apply_marginal_property(K, (1 - K) * mean_filter, (1 - K) * var_filter, *params[:, t + 1])

            params[:, t] = [mean, var]

        return params
