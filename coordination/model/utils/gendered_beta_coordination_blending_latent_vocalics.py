from __future__ import annotations

from typing import Optional

import numpy as np

from coordination.model.utils.beta_coordination_blending_latent_vocalics import \
    BetaCoordinationLatentVocalicsTrainingHyperParameters, BetaCoordinationLatentVocalicsModelParameters


class GenderedBetaCoordinationLatentVocalicsTrainingHyperParameters(
    BetaCoordinationLatentVocalicsTrainingHyperParameters):

    def __init__(self, a_vu: float, b_vu: float, a_va: float, b_va: float, a_vaa: float, b_vaa: float,
                 mu_mo_male: np.ndarray, nu_mo_male: np.ndarray, a_vo_male: np.ndarray, b_vo_male: np.ndarray,
                 mu_mo_female: np.ndarray, nu_mo_female: np.ndarray, a_vo_female: np.ndarray,
                 b_vo_female: np.ndarray, vu0: float, vc0: float, va0: float, vaa0: float, mo0_male: np.ndarray,
                 vo0_male: np.ndarray, mo0_female: np.ndarray, vo0_female: np.ndarray, vu_mcmc_prop: float,
                 vc_mcmc_prop: float, u_mcmc_iter: int, c_mcmc_iter: int):
        """
        @param a_vu: 1st parameter of vu prior (inv. gamma)
        @param b_vu: 2nd parameter of vu prior
        @param a_va: 1st parameter of va prior (inv. gamma)
        @param b_va: 2nd parameter of va prior
        @param a_vaa: 1st parameter of vaa prior (inv. gamma)
        @param b_vaa: 2nd parameter of vaa prior
        @param mu_mo_male: 1st parameter of mo and vo prior (Normal-Inverse-Gamma w/ params mu, nu, a, b)
        @param nu_mo_male: 2nd parameter of mo and vo prior
        @param a_vo_male: 3rd parameter of mo and vo prior
        @param b_vo_male: 4rd parameter of mo and vo prior
        @param mu_mo_female: 1st parameter of mo and vo prior (Normal-Inverse-Gamma w/ params mu, nu, a, b)
        @param nu_mo_female: 2nd parameter of mo and vo prior
        @param a_vo_female: 3rd parameter of mo and vo prior
        @param b_vo_female: 4rd parameter of mo and vo prior
        @param vu0: Initial vu
        @param vc0: Initial vc
        @param va0: Initial va
        @param vaa0: Initial vaa
        @param vaa0: Initial vaa
        @param vaa0: Initial vaa
        @param vu_mcmc_prop: Variance of the proposal distribution for unbounded coordination
        @param vc_mcmc_prop: Variance of the proposal distribution for coordination
        @param u_mcmc_iter: Number of MCMC samples to discard when sampling unbounded coordination
        @param c_mcmc_iter: Number of MCMC samples to discard when sampling coordination
        """

        # var_c has uniform prior so, we can set a_vc and b_vc to 0
        super().__init__(a_vu, b_vu, a_va, b_va, a_vaa, b_vaa, 0, 0, vu0, vc0, va0, vaa0, 0, vu_mcmc_prop,
                         vc_mcmc_prop, u_mcmc_iter, c_mcmc_iter)

        # Prior parameters
        self.mu_mo_male = mu_mo_male
        self.nu_mo_male = nu_mo_male
        self.a_vo_male = a_vo_male
        self.b_vo_male = b_vo_male
        self.mu_mo_female = mu_mo_female
        self.nu_mo_female = nu_mo_female
        self.a_vo_female = a_vo_female
        self.b_vo_female = b_vo_female

        # Initial values
        self.mo0_male = mo0_male
        self.vo0_male = vo0_male
        self.mo0_female = mo0_female
        self.vo0_female = vo0_female


class GenderedBetaCoordinationLatentVocalicsModelParameters(BetaCoordinationLatentVocalicsModelParameters):

    def __init__(self):
        super().__init__()

        # We do not fit var_o. Now we fit a mean and a variance for O, difference for each gender.
        self.set_var_o(0, True)

        # Mean and variance are fit jointly thus we control freezing with one parameter only.
        # From a conjugate Gaussian-Inverse-Gamma distribution.
        # We also have one means and variance per vocalic feature (so the array notation) in each gender.
        self._mean_o_male: Optional[np.ndarray] = None
        self._var_o_male: Optional[np.ndarray] = None
        self._mean_var_o_male_frozen = False

        self._mean_o_female: Optional[np.ndarray] = None
        self._var_o_female: Optional[np.ndarray] = None
        self._mean_var_o_female_frozen = False

    def freeze(self):
        super().freeze()
        self._mean_var_o_male_frozen = True
        self._mean_var_o_female_frozen = True

    def reset(self):
        super().reset()
        self.set_var_o(0, True)

        self._mean_o_male = None
        self._var_o_male = None
        self._mean_var_o_male_frozen = False

        self._mean_o_female = None
        self._var_o_female = None
        self._mean_var_o_female_frozen = False

    def set_mean_var_male(self, mean_o: np.ndarray, var_o: np.ndarray, freeze: bool = True):
        self._mean_o_male = mean_o
        self._var_o_male = var_o
        self._mean_var_o_male_frozen = freeze

    def set_mean_var_female(self, mean_o: np.ndarray, var_o: np.ndarray, freeze: bool = True):
        self._mean_o_female = mean_o
        self._var_o_female = var_o
        self._mean_var_o_female_frozen = freeze

    @property
    def mean_o_male(self):
        return self._mean_o_male

    @property
    def var_o_male(self):
        return self._var_o_male

    @property
    def mean_var_o_male_frozen(self):
        return self._mean_var_o_male_frozen

    @property
    def mean_o_female(self):
        return self._mean_o_female

    @property
    def var_o_female(self):
        return self._var_o_female

    @property
    def mean_var_o_female_frozen(self):
        return self._mean_var_o_female_frozen
