from __future__ import annotations

from typing import Any, Dict, List, Optional

from copy import deepcopy

import numpy as np

from coordination.common.dataset import EvidenceDataset, EvidenceDataSeries
from coordination.component.speech.common import VocalicsSparseSeries
from coordination.model.particle_filter import Particles
from coordination.model.pgm import ParticlesSummary, Samples, TrainingHyperParameters, ModelParameters


class BrainBodyParticlesSummary(ParticlesSummary):
    unbounded_coordination_mean: np.ndarray
    unbounded_coordination_std: np.ndarray
    coordination_mean: np.ndarray
    coordination_std: np.ndarray
    latent_brain_mean: np.ndarray
    latent_brain_std: np.ndarray
    latent_body_mean: np.ndarray
    latent_body_std: np.ndarray


class BrainBodySamples(Samples):
    unbounded_coordination: np.ndarray
    coordination: np.ndarray
    latent_brain: np.ndarray
    latent_body: np.ndarray
    observed_brain: np.ndarray
    observed_body: np.ndarray

    def __init__(self, num_subjects: int):
        self.num_subjects = num_subjects

    @property
    def size(self):
        return len(self.observed_brain)


class BrainBodyDataSeries(EvidenceDataSeries):

    def __init__(self,
                 uuid: str,
                 observed_brain_signals: np.ndarray,
                 observed_body_movements: np.ndarray,
                 team_score: float,
                 unbounded_coordination: Optional[np.ndarray] = None,
                 coordination: Optional[np.ndarray] = None,
                 latent_brain_signals: Optional[np.ndarray] = None,
                 latent_body_movements: Optional[np.ndarray] = None):
        super().__init__(uuid)

        self.observed_brain_signals = observed_brain_signals
        self.observed_body_movements = observed_body_movements
        self.team_score = team_score
        self.unbounded_coordination = unbounded_coordination
        self.coordination = coordination
        self.latent_brain_signals = latent_brain_signals
        self.latent_body_movements = latent_body_movements

    @property
    def num_time_steps(self):
        # dims: time x subject x channels
        return self.observed_brain_signals.size(axis=0)

    @property
    def num_subjects(self):
        # dims: time x subject x channels
        return self.observed_brain_signals.size(axis=1)

    @property
    def num_fnirs_channels(self):
        # dims: time x subject x channels
        return self.observed_brain_signals.size(axis=2)

    def shuffle_per_subject(self):
        num_subjects = self.observed_brain_signals.size(axis=1)
        for i in range(num_subjects):
            shuffled_indices = np.arange(self.num_time_steps)
            np.random.shuffle(shuffled_indices)

            self.observed_brain_signals[:, i] = self.observed_brain_signals[shuffled_indices, i]
            self.observed_body_movements[:, i] = self.observed_body_movements[shuffled_indices, i]


class BrainBodyDataset(EvidenceDataset):

    def __init__(self, series: List[BrainBodyDataSeries]):
        super().__init__(series)

        self.series: List[BrainBodyDataSeries] = series

        # Below we store data from the list of series in this dataset in a set of tensors
        # for fast processing during training.
        self.coordination = np.array([])
        self.latent_vocalics = np.array([])
        self.observed_vocalics = np.array([])
        self.vocalics_mask = np.array([])
        self.speech_semantic_links = np.array([])

        # Dependency on vocalics from the same speaker
        self.previous_vocalics_from_self = np.array([])
        self.previous_vocalics_from_self_mask = np.array([])
        self.next_vocalics_from_self = np.array([])

        # Dependency on vocalics from a different speaker
        self.previous_vocalics_from_other = np.array([])
        self.previous_vocalics_from_self_mask = np.array([])
        self.next_vocalics_from_other = np.array([])

        # Table containing speaker's gender per trial and time step
        self.genders = np.array([])

        self._fill_tensors()

    def _fill_tensors(self):
        num_trials = len(self.series)
        num_vocalic_features = self.series[0].num_vocalic_features if num_trials > 0 else 0
        num_time_steps = self.series[0].num_time_steps if num_trials > 0 else 0

        self.coordination = None if self.series[0].coordination is None else np.zeros((num_trials, num_time_steps))
        self.latent_vocalics = None if self.series[0].latent_vocalics is None else np.zeros(
            (num_trials, num_vocalic_features, num_time_steps))
        self.observed_vocalics = np.zeros((num_trials, num_vocalic_features, num_time_steps))
        self.vocalics_mask = np.zeros((num_trials, num_time_steps))
        self.speech_semantic_links = np.zeros((num_trials, num_time_steps))

        self.previous_vocalics_from_self = np.ones((num_trials, num_time_steps)).astype(np.int) * -1
        self.previous_vocalics_from_self_mask = np.zeros_like(self.previous_vocalics_from_self)
        self.next_vocalics_from_self = np.ones((num_trials, num_time_steps)).astype(np.int) * -1
        self.enable_self_dependency()

        self.previous_vocalics_from_other = np.ones((num_trials, num_time_steps)).astype(np.int) * -1
        self.previous_vocalics_from_self_mask = np.zeros_like(self.previous_vocalics_from_other)
        self.next_vocalics_from_other = np.ones((num_trials, num_time_steps)).astype(np.int) * -1

        self.genders = np.ones((num_trials, num_time_steps)) * -1

        for i, s in enumerate(self.series):
            if s.coordination is not None:
                self.coordination[i] = s.coordination

            if s.latent_vocalics is not None:
                self.latent_vocalics[i] = s.latent_vocalics.values

            self.observed_vocalics[i] = s.observed_vocalics.values
            self.vocalics_mask[i] = s.observed_vocalics.mask

            # Vocalics from other speaker
            self.previous_vocalics_from_other[i] = np.array(
                [-1 if t is None else t for t in s.observed_vocalics.previous_from_other])

            self.speech_semantic_links[i] = s.speech_semantic_links

            for t in range(s.num_time_steps):
                if self.previous_vocalics_from_other[i, t] >= 0:
                    self.next_vocalics_from_other[i, self.previous_vocalics_from_other[i, t]] = t

                # Gender
                if s.observed_vocalics.mask[t] == 1:
                    self.genders[i, t] = s.get_speaker_gender(s.observed_vocalics.utterances[t].subject_id)

        self.previous_vocalics_from_other_mask = np.where(self.previous_vocalics_from_other >= 0, 1, 0)

    def disable_self_dependency(self):
        self.previous_vocalics_from_self = np.ones_like(self.previous_vocalics_from_self) * -1
        self.previous_vocalics_from_self_mask = np.zeros_like(self.previous_vocalics_from_self)
        self.next_vocalics_from_self = np.ones_like(self.previous_vocalics_from_self) * -1

    def enable_self_dependency(self):
        for i, series in enumerate(self.series):
            self.previous_vocalics_from_self[i] = np.array(
                [-1 if t is None else t for t in series.observed_vocalics.previous_from_self])

            for t in range(series.num_time_steps):
                if self.previous_vocalics_from_self[i, t] >= 0:
                    self.next_vocalics_from_self[i, self.previous_vocalics_from_self[i, t]] = t

        self.previous_vocalics_from_self_mask = np.where(self.previous_vocalics_from_self >= 0, 1, 0)

    def get_subset(self, indices: List[int]) -> LatentVocalicsDataset:
        return self.__class__(series=[self.series[i] for i in indices])

    def merge(self, dataset2: LatentVocalicsDataset) -> EvidenceDataset:
        return self.__class__(series=self.series + dataset2.series)

    def remove_vocalic_features(self, feature_indices: List[int]):
        self.latent_vocalics = None if self.latent_vocalics is None else np.delete(np.zeros_like(self.latent_vocalics),
                                                                                   feature_indices, axis=1)
        self.observed_vocalics = np.delete(np.zeros_like(self.observed_vocalics), feature_indices, axis=1)

        for i in range(self.num_trials):
            if self.series[i].latent_vocalics is not None:
                self.series[i].latent_vocalics.values = np.delete(self.series[i].latent_vocalics.values,
                                                                  feature_indices, axis=0)
                self.latent_vocalics[i] = self.series[i].latent_vocalics.values

            self.series[i].observed_vocalics.values = np.delete(self.series[i].observed_vocalics.values,
                                                                feature_indices, axis=0)
            self.observed_vocalics[i] = self.series[i].observed_vocalics.values

    def keep_vocalic_features(self, features: List[str]):
        if self.num_trials > 0:
            FEATURE_MAP = {feature_name: i for i, feature_name in enumerate(self.series[0].features)}
            features_to_remove = set(range(self.series[0].observed_vocalics.num_features))
            for f in features:
                features_to_remove.remove(FEATURE_MAP[f])

            if len(features_to_remove) > 0:
                # Remove feature from the dataset
                self.remove_vocalic_features(list(features_to_remove))

    def normalize_per_subject(self):
        for s in self.series:
            s.observed_vocalics.normalize_per_subject()
        self._fill_tensors()

    # Replaces genders that are different from male (0) and female(1) with a random sample with probabilities
    # proportional to the quantity of disclosed males and females.
    def normalize_gender(self):
        males = 0
        females = 0

        for s in self.series:
            for gender in s.genders.values():
                if gender == 0:
                    males += 1
                elif gender == 1:
                    females += 1

        p_male = males / (males + females)

        for s in self.series:
            for key, gender in s.genders.items():
                if gender != 0 and gender != 1:
                    u = np.random.rand()
                    if u <= p_male:
                        s.genders[key] = 0  # male
                    else:
                        s.genders[key] = 1  # female

        self._fill_tensors()

    def disable_speech_semantic_links(self):
        for s in self.series:
            s.speech_semantic_links = np.zeros_like(s.speech_semantic_links)
        self._fill_tensors()

    def shuffle(self):
        for s in self.series:
            s.shuffle_observed_vocalics_and_semantic_links_per_speaker()

        self._fill_tensors()


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


class LatentVocalicsModelParameters(ModelParameters):

    def __init__(self):
        self._var_c: Optional[float] = None
        self._var_a: Optional[np.ndarray] = None
        self._var_aa: Optional[np.ndarray] = None
        self._var_o: Optional[np.ndarray] = None

        self._var_c_frozen = False
        self._var_a_frozen = False
        self._var_aa_frozen = False
        self._var_o_frozen = False

    def freeze(self):
        self._var_c_frozen = True
        self._var_a_frozen = True
        self._var_aa_frozen = True
        self._var_o_frozen = True

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
