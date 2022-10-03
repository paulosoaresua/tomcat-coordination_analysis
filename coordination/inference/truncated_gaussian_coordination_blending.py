from typing import Callable

import numpy as np

from coordination.component.speech.common import VocalicsSparseSeries
from coordination.inference.inference_engine import InferenceEngine
from coordination.inference.lds import apply_marginal_property, apply_conditional_property, pdf_projection


class TruncatedGaussianCoordinationBlendingInference(InferenceEngine):

    def __init__(self, vocalic_series: VocalicsSparseSeries, mean_prior_coordination: float,
                 std_prior_coordination: float, std_coordination_drifting: float, mean_prior_vocalics: np.array,
                 std_prior_vocalics: np.array, std_coordinated_vocalics: np.ndarray, f: Callable = lambda x, s: x,
                 fix_coordination_on_second_half: bool = True):

        assert len(mean_prior_vocalics) == vocalic_series.num_series
        assert len(std_prior_vocalics) == vocalic_series.num_series
        assert len(std_coordinated_vocalics) == vocalic_series.num_series

        self._vocalic_series = vocalic_series
        self._mean_prior_coordination = mean_prior_coordination
        self._std_prior_coordination = std_prior_coordination
        self._std_coordination_drifting = std_coordination_drifting
        self._mean_prior_vocalics = mean_prior_vocalics
        self._std_prior_vocalics = std_prior_vocalics
        self._std_coordinated_vocalics = std_coordinated_vocalics
        self._f = f
        self._fix_coordination_on_second_half = fix_coordination_on_second_half

        self._num_features, self._time_steps = vocalic_series.values.shape  # n and T

    def estimate_means_and_variances(self) -> np.ndarray:
        filter_params = self._kalman_filter()
        return self._rts_smoother(filter_params)

    def filter(self) -> np.ndarray:
        return self._kalman_filter()

    def _kalman_filter(self) -> np.ndarray:
        var_C = self._std_coordination_drifting ** 2
        var_V = self._std_coordinated_vocalics ** 2

        M = int(self._time_steps / 2)
        num_time_steps = M + 1 if self._fix_coordination_on_second_half else self._time_steps

        params = np.zeros((2, num_time_steps))
        params[:, 0] = np.array([self._mean_prior_coordination, self._std_prior_coordination ** 2])
        mean, var = params[:, 0]
        for t in range(1, self._time_steps):
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

            A_t = self._f(self._vocalic_series.values[:, t], 0)
            A_prev = None if self._vocalic_series.previous_from_self[t] is None else self._f(
                self._vocalic_series.values[:, self._vocalic_series.previous_from_self[t]], 0)
            B_prev = None if self._vocalic_series.previous_from_other[t] is None else self._f(
                self._vocalic_series.values[:, self._vocalic_series.previous_from_other[t]], 1)

            if self._vocalic_series.mask[t] == 1 and B_prev is not None:
                if A_prev is None:
                    A_prev = self._mean_prior_vocalics
                D = B_prev - A_prev
                for i in range(self._num_features):
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
