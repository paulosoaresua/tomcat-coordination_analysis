from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from coordination.common.dataset import EvidenceDataset, EvidenceDataSeries
from coordination.component.speech.common import VocalicsSparseSeries
from coordination.model.particle_filter import Particles
from coordination.model.pgm import ParticlesSummary, Samples, TrainingHyperParameters


def clip_coordination(coordination: np.ndarray) -> np.ndarray:
    return np.clip(coordination, a_min=0, a_max=1)


class LatentVocalicsParticles(Particles):
    coordination: np.ndarray
    latent_vocalics: Dict[str, np.ndarray]

    def _keep_particles_at(self, indices: np.ndarray):
        if isinstance(self.coordination, np.ndarray):
            # otherwise, coordination is given and it will be a single number
            self.coordination = self.coordination[indices]

        for speaker, latent_vocalics in self.latent_vocalics.items():
            if latent_vocalics is not None:
                if np.ndim(latent_vocalics) > 1:
                    # otherwise, coordination is given and it will be a single number
                    self.latent_vocalics[speaker] = latent_vocalics[indices, :]


class LatentVocalicsParticlesSummary(ParticlesSummary):
    coordination_mean: np.ndarray
    coordination_var: np.ndarray
    latent_vocalics_mean: np.ndarray
    latent_vocalics_var: np.ndarray


class LatentVocalicsSamples(Samples):
    coordination: np.ndarray
    latent_vocalics: List[VocalicsSparseSeries]
    observed_vocalics: List[VocalicsSparseSeries]

    @property
    def size(self):
        return len(self.observed_vocalics)


class LatentVocalicsDataSeries(EvidenceDataSeries):

    def __init__(self, uuid: str, observed_vocalics: VocalicsSparseSeries, coordination: Optional[np.ndarray] = None,
                 latent_vocalics: VocalicsSparseSeries = None):
        super().__init__(uuid)
        self.coordination = coordination
        self.latent_vocalics = latent_vocalics
        self.observed_vocalics = observed_vocalics

    @property
    def is_complete(self) -> bool:
        return self.coordination is not None and self.latent_vocalics is not None

    @property
    def num_time_steps(self):
        return self.observed_vocalics.num_time_steps

    @property
    def num_vocalic_features(self):
        return self.observed_vocalics.num_features


class LatentVocalicsDataset(EvidenceDataset):

    def __init__(self, series: List[LatentVocalicsDataSeries], team_scores: np.ndarray,
                 team_process_surveys: np.ndarray, team_satisfaction_surveys: np.ndarray, genders: np.ndarray,
                 ages: np.ndarray):
        super().__init__(series)

        self.series: List[LatentVocalicsDataSeries] = series

        self.team_scores = team_scores
        self.team_process_surveys = team_process_surveys
        self.team_satisfaction_surveys = team_satisfaction_surveys
        self.genders = genders
        self.ages = ages

        # Keep a matrix representation of the data for fast processing during training
        self.coordination = None if series[0].coordination is None else np.zeros(
            (len(series), series[0].num_time_steps))

        # n (num samples) x k (num features) x T (num time steps)
        self.latent_vocalics = None if series[0].latent_vocalics is None else np.zeros(
            (len(series), series[0].num_vocalic_features, series[0].num_time_steps))

        self.observed_vocalics = np.zeros((len(series), series[0].num_vocalic_features, series[0].num_time_steps))
        self.vocalics_mask = np.zeros((len(series), series[0].num_time_steps))
        self.previous_vocalics_from_self = np.zeros((len(series), series[0].num_time_steps)).astype(np.int)
        self.previous_vocalics_from_other = np.zeros((len(series), series[0].num_time_steps)).astype(np.int)
        self.next_vocalics_from_self = np.ones((len(series), series[0].num_time_steps)).astype(np.int) * -1
        self.next_vocalics_from_other = np.ones((len(series), series[0].num_time_steps)).astype(np.int) * -1

        for i, series in enumerate(series):
            if series.coordination is not None:
                self.coordination[i] = series.coordination

            if series.latent_vocalics is not None:
                self.latent_vocalics[i] = series.latent_vocalics.values

            self.observed_vocalics[i] = series.observed_vocalics.values
            self.vocalics_mask[i] = series.observed_vocalics.mask
            self.previous_vocalics_from_self[i] = np.array(
                [-1 if t is None else t for t in series.observed_vocalics.previous_from_self])
            self.previous_vocalics_from_other[i] = np.array(
                [-1 if t is None else t for t in series.observed_vocalics.previous_from_other])
            self.previous_vocalics_from_self_mask = np.where(self.previous_vocalics_from_self >= 0, 1, 0)
            self.previous_vocalics_from_other_mask = np.where(self.previous_vocalics_from_other >= 0, 1, 0)

            for t in range(series.num_time_steps):
                if self.previous_vocalics_from_self[i, t] >= 0:
                    self.next_vocalics_from_self[i, self.previous_vocalics_from_self[i, t]] = t

                if self.previous_vocalics_from_other[i, t] >= 0:
                    self.next_vocalics_from_other[i, self.previous_vocalics_from_other[i, t]] = t

    def get_subset(self, indices: List[int]) -> LatentVocalicsDataset:
        return self.__class__(
            series=[self.series[i] for i in indices],
            team_scores=self.team_scores[indices],
            team_process_surveys=np.take_along_axis(self.team_process_surveys, indices, axis=0),
            team_satisfaction_surveys=np.take_along_axis(self.team_satisfaction_surveys, indices, axis=0),
            genders=np.take_along_axis(self.genders, indices, axis=0),
            ages=np.take_along_axis(self.ages, indices, axis=0)
        )

    def merge(self, dataset2: LatentVocalicsDataset) -> EvidenceDataset:
        return self.__class__(
            series=self.series + dataset2.series,
            team_scores=np.concatenate([self.team_scores, dataset2.team_scores]),
            team_process_surveys=np.concatenate([self.team_process_surveys, dataset2.team_process_surveys], axis=0),
            team_satisfaction_surveys=np.concatenate(
                [self.team_satisfaction_surveys, dataset2.team_satisfaction_surveys], axis=0),
            genders=np.concatenate([self.genders, dataset2.genders], axis=0),
            ages=np.concatenate([self.ages, dataset2.ages], axis=0),
        )


class BaseF:

    def __call__(self, latent_vocalics: np.ndarray, speaker_mask: int) -> np.ndarray:
        return latent_vocalics

    def __repr__(self):
        return "Identity"


class BaseG:

    def __call__(self, latent_vocalics: np.ndarray) -> np.ndarray:
        return latent_vocalics

    def __repr__(self):
        return "Identity"


class LatentVocalicsTrainingHyperParameters(TrainingHyperParameters):

    def __init__(self, a_vc: float, b_vc: float, a_va: float, b_va: float, a_vaa: float, b_vaa: float, a_vo: float,
                 b_vo: float, vc0: float, va0: float, vaa0: float, vo0: float):
        """
        @param a_vc: 1st parameter of vc prior (inv. gamma)
        @param b_vc: 2nd parameter of vc prior
        @param a_va: 1st parameter of va prior (inv. gamma)
        @param b_va: 2nd parameter of va prior
        @param a_vaa: 1st parameter of vaa prior (inv. gamma)
        @param b_vaa: 2nd parameter of vaa prior
        @param a_vo: 1st parameter of vo prior (inv. gamma)
        @param b_vo: 2nd parameter of vo prior
        @param vc0: Initial vc
        @param va0: Initial va
        @param vaa0: Initial vaa
        @param vo0: Initial vo
        """

        self.a_vc = a_vc
        self.b_vc = b_vc
        self.a_va = a_va
        self.b_va = b_va
        self.a_vaa = a_vaa
        self.b_vaa = b_vaa
        self.a_vo = a_vo
        self.b_vo = b_vo
        self.vc0 = vc0
        self.va0 = va0
        self.vaa0 = vaa0
        self.vo0 = vo0

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__


class LatentVocalicsModelParameters:

    def __init__(self):
        self._var_c: Optional[float] = None
        self._var_a: Optional[np.ndarray] = None
        self._var_aa: Optional[np.ndarray] = None
        self._var_o: Optional[np.ndarray] = None

        self._var_c_frozen = False
        self._var_a_frozen = False
        self._var_aa_frozen = False
        self._var_o_frozen = False

    def reset(self):
        self._var_c = None
        self._var_a = None
        self._var_aa = None
        self._var_o = None

        self._var_c_frozen = False
        self._var_a_frozen = False
        self._var_aa_frozen = False
        self._var_o_frozen = False

    def set_var_c(self, var_c: float, freeze: bool = True):
        self._var_c = var_c
        self._var_c_frozen = freeze

    def set_var_a(self, var_a: float, freeze: bool = True):
        self._var_a = var_a
        self._var_a_frozen = freeze

    def set_var_aa(self, var_aa: float, freeze: bool = True):
        self._var_aa = var_aa
        self._var_aa_frozen = freeze

    def set_var_o(self, var_o: float, freeze: bool = True):
        self._var_o = var_o
        self._var_o_frozen = freeze

    @property
    def var_c(self):
        return self._var_c

    @property
    def var_a(self):
        return self._var_a

    @property
    def var_aa(self):
        return self._var_aa

    @property
    def var_o(self):
        return self._var_o

    @property
    def var_c_frozen(self):
        return self._var_c_frozen

    @property
    def var_a_frozen(self):
        return self._var_a_frozen

    @property
    def var_aa_frozen(self):
        return self._var_aa_frozen

    @property
    def var_o_frozen(self):
        return self._var_o_frozen
