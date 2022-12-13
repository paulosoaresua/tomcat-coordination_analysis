from __future__ import annotations

from typing import List, Optional

import numpy as np

from coordination.component.speech.common import VocalicsSparseSeries
from coordination.model.utils.coordination_blending_latent_vocalics import LatentVocalicsDataset, \
    LatentVocalicsDataSeries, LatentVocalicsParticles, LatentVocalicsSamples, \
    LatentVocalicsParticlesSummary, LatentVocalicsModelParameters, LatentVocalicsTrainingHyperParameters


class BetaCoordinationLatentVocalicsParticles(LatentVocalicsParticles):
    unbounded_coordination: np.ndarray

    def _keep_particles_at(self, indices: np.ndarray):
        super()._keep_particles_at(indices)

        if isinstance(self.unbounded_coordination, np.ndarray):
            self.unbounded_coordination = self.unbounded_coordination[indices]


class BetaCoordinationLatentVocalicsParticlesSummary(LatentVocalicsParticlesSummary):
    unbounded_coordination_mean: np.ndarray
    unbounded_coordination_var: np.ndarray

    @classmethod
    def from_latent_vocalics_particles_summary(cls, summary: LatentVocalicsParticlesSummary):
        new_summary = cls()
        new_summary.coordination_mean = summary.coordination_mean
        new_summary.coordination_var = summary.coordination_var
        new_summary.latent_vocalics_mean = summary.latent_vocalics_mean
        new_summary.latent_vocalics_var = summary.latent_vocalics_var

        return new_summary


class BetaCoordinationLatentVocalicsSamples(LatentVocalicsSamples):
    unbounded_coordination: np.ndarray


class BetaCoordinationLatentVocalicsDataSeries(LatentVocalicsDataSeries):

    def __init__(self, uuid: str, observed_vocalics: VocalicsSparseSeries, team_score: float,
                 team_process_surveys: np.ndarray, team_satisfaction_surveys: np.ndarray, genders: np.ndarray,
                 ages: np.ndarray, features: List[str], unbounded_coordination: Optional[np.ndarray] = None,
                 coordination: Optional[np.ndarray] = None, latent_vocalics: VocalicsSparseSeries = None):
        super().__init__(uuid, observed_vocalics, team_score, team_process_surveys, team_satisfaction_surveys, genders,
                         ages, features, coordination, latent_vocalics)
        self.unbounded_coordination = unbounded_coordination

    @property
    def is_complete(self) -> bool:
        return super().is_complete and self.unbounded_coordination is not None

    @classmethod
    def from_latent_vocalics_data_series(cls,
                                         series: LatentVocalicsDataSeries) -> BetaCoordinationLatentVocalicsDataSeries:
        return cls(
            uuid=series.uuid,
            coordination=series.coordination,
            latent_vocalics=series.latent_vocalics,
            observed_vocalics=series.observed_vocalics,
            team_score=series.team_score,
            team_process_surveys=series.team_process_surveys,
            team_satisfaction_surveys=series.team_satisfaction_surveys,
            genders=series.genders,
            ages=series.ages,
            features=series.features
        )


class BetaCoordinationLatentVocalicsDataset(LatentVocalicsDataset):

    def __init__(self, series: List[BetaCoordinationLatentVocalicsDataSeries]):
        super().__init__(series)

        self.series: List[BetaCoordinationLatentVocalicsDataSeries] = series

        # Extra tensor
        self.unbounded_coordination = None if series[0].unbounded_coordination is None else np.array(
            [s.unbounded_coordination for s in series])

    @classmethod
    def from_samples(cls, samples: BetaCoordinationLatentVocalicsSamples) -> BetaCoordinationLatentVocalicsDataset:
        series = []
        for i in range(samples.size):
            unbounded_coordination = samples.unbounded_coordination[
                i] if samples.unbounded_coordination is not None else None
            coordination = samples.coordination[i] if samples.coordination is not None else None
            latent_vocalics = samples.latent_vocalics[i] if samples.latent_vocalics is not None else None

            # We fix male gender for even speaker and female for odd ones
            genders = [-1 if speaker is None else speaker % 2 for speaker in samples.observed_vocalics[i].subjects]

            s = BetaCoordinationLatentVocalicsDataSeries(
                uuid=f"{i}",
                unbounded_coordination=unbounded_coordination,
                coordination=coordination,
                latent_vocalics=latent_vocalics,
                observed_vocalics=samples.observed_vocalics[i],
                team_score=0,
                team_process_surveys=np.array([]),
                team_satisfaction_surveys=np.array([]),
                ages=np.array([]),
                genders=genders
            )

            series.append(s)

        evidence = cls(series)

        return evidence

    @classmethod
    def from_latent_vocalics_dataset(cls, dataset: LatentVocalicsDataset):
        series = []
        for i in range(dataset.num_trials):
            series.append(BetaCoordinationLatentVocalicsDataSeries.from_latent_vocalics_data_series(dataset.series[i]))

        return cls(series=series)


class BetaCoordinationLatentVocalicsTrainingHyperParameters(LatentVocalicsTrainingHyperParameters):

    def __init__(self, a_vu: float, b_vu: float, a_va: float, b_va: float, a_vaa: float,
                 b_vaa: float, a_vo: float, b_vo: float, vu0: float, vc0: float, va0: float, vaa0: float, vo0: float,
                 vu_mcmc_prop: float, vc_mcmc_prop: float, u_mcmc_iter: int, c_mcmc_iter: int):
        """
        @param a_vu: 1st parameter of vu prior (inv. gamma)
        @param b_vu: 2nd parameter of vu prior
        @param a_va: 1st parameter of va prior (inv. gamma)
        @param b_va: 2nd parameter of va prior
        @param a_vaa: 1st parameter of vaa prior (inv. gamma)
        @param b_vaa: 2nd parameter of vaa prior
        @param a_vo: 1st parameter of vo prior (inv. gamma)
        @param b_vo: 2nd parameter of vo prior
        @param vu0: Initial vu
        @param vc0: Initial vc
        @param va0: Initial va
        @param vaa0: Initial vaa
        @param vo0: Initial vo
        @param vu_mcmc_prop: Variance of the proposal distribution for unbounded coordination
        @param vc_mcmc_prop: Variance of the proposal distribution for coordination
        @param u_mcmc_iter: Number of MCMC samples to discard when sampling unbounded coordination
        @param c_mcmc_iter: Number of MCMC samples to discard when sampling coordination
        """

        # var_c has uniform prior so, we can set a_vc and b_vc to 0
        super().__init__(0, 0, a_va, b_va, a_vaa, b_vaa, a_vo, b_vo, vc0, va0, vaa0, vo0)

        self.a_vu = a_vu
        self.b_vu = b_vu
        self.vu0 = vu0
        self.vu_mcmc_prop = vu_mcmc_prop
        self.vc_mcmc_prop = vc_mcmc_prop
        self.u_mcmc_iter = u_mcmc_iter
        self.c_mcmc_iter = c_mcmc_iter


class BetaCoordinationLatentVocalicsModelParameters(LatentVocalicsModelParameters):

    def __init__(self):
        super().__init__()

        self._var_u: Optional[float] = None
        self._var_u_frozen = False

    def freeze(self):
        super().freeze()
        self._var_u_frozen = True

    def reset(self):
        super().reset()
        self._var_u = None
        self._var_u_frozen = False

    def set_var_u(self, var_u: float, freeze: bool = True):
        self._var_u = var_u
        self._var_u_frozen = freeze

    @property
    def var_u(self):
        return self._var_u

    @property
    def var_u_frozen(self):
        return self._var_u_frozen
