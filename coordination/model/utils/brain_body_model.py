from __future__ import annotations

from typing import Any, List, Optional

import numpy as np

from coordination.common.dataset import EvidenceDataset, EvidenceDataSeries
from coordination.model.pgm import ParticlesSummary, Samples


class BrainBodyParticlesSummary(ParticlesSummary):
    unbounded_coordination_mean: np.ndarray
    unbounded_coordination_std: np.ndarray
    coordination_mean: np.ndarray
    coordination_std: np.ndarray
    latent_brain_mean: np.ndarray
    latent_brain_std: np.ndarray
    latent_body_mean: np.ndarray
    latent_body_std: np.ndarray

    @classmethod
    def from_inference_data(cls, idata: Any, retain_every: int = 1) -> BrainBodyParticlesSummary:
        summary = cls()
        summary.unbounded_coordination_mean = idata.posterior["unbounded_coordination"][
                                              ::retain_every].mean(
            dim=["chain", "draw"]).to_numpy()
        summary.coordination_mean = idata.posterior["coordination"][::retain_every].mean(
            dim=["chain", "draw"]).to_numpy()
        summary.latent_brain_mean = idata.posterior["latent_brain"][::retain_every].mean(
            dim=["chain", "draw"]).to_numpy()
        # summary.latent_body_mean = idata.posterior["latent_body"][::retain_every].mean(
        #     dim=["chain", "draw"])

        summary.unbounded_coordination_std = idata.posterior["unbounded_coordination"][
                                             ::retain_every].std(
            dim=["chain", "draw"]).to_numpy()
        summary.coordination_std = idata.posterior["coordination"][::retain_every].std(
            dim=["chain", "draw"]).to_numpy()
        summary.latent_brain_std = idata.posterior["latent_brain"][::retain_every].std(
            dim=["chain", "draw"]).to_numpy()
        # summary.latent_body_std = idata.posterior["latent_body"][::retain_every].std(
        #     dim=["chain", "draw"])

        return summary


class BrainBodySamples(Samples):

    def __init__(self):
        self.unbounded_coordination: Optional[np.ndarray] = None
        self.coordination: Optional[np.ndarray] = None
        self.latent_brain: Optional[np.ndarray] = None
        self.latent_body: Optional[np.ndarray] = None
        self.observed_brain: Optional[np.ndarray] = None
        self.observed_body: Optional[np.ndarray] = None

    @property
    def size(self):
        if self.observed_brain is None:
            return 0

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
        # trial x speaker x channels/features x time
        return self.observed_brain_signals.shape[-1]

    @property
    def num_subjects(self):
        # trial x speaker x channels/features x time
        return self.observed_brain_signals.shape[1]

    @property
    def num_fnirs_channels(self):
        # trial x speaker x channels/features x time
        return self.observed_brain_signals.shape[2]

    def shuffle_per_subject(self):
        for i in range(self.num_subjects):
            shuffled_indices = np.arange(self.num_time_steps)
            np.random.shuffle(shuffled_indices)

            self.observed_brain_signals[:, i] = self.observed_brain_signals[shuffled_indices, i]
            self.observed_body_movements[:, i] = self.observed_body_movements[shuffled_indices, i]

    def normalize_per_subject(self):
        mean = self.observed_brain_signals.mean(axis=-1)[:, :, None]
        std = self.observed_brain_signals.std(axis=-1)[:, :, None]
        self.observed_brain_signals = (self.observed_brain_signals - mean) / std

        mean = self.observed_body_movements.mean(axis=-1)[:, :, None]
        std = self.observed_body_movements.std(axis=-1)[:, :, None]
        self.observed_body_movements = (self.observed_body_movements - mean) / std


class BrainBodyDataset(EvidenceDataset):

    def __init__(self, series: List[BrainBodyDataSeries]):
        super().__init__(series)

        self.series: List[BrainBodyDataSeries] = series

    @property
    def unbounded_coordination(self):
        if len(self.series) == 0 or self.series[0].unbounded_coordination is None:
            return None

        return np.array([s.unbounded_coordination for s in self.series])

    @property
    def coordination(self):
        if len(self.series) == 0 or self.series[0].coordination is None:
            return None

        return np.array([s.coordination for s in self.series])

    @property
    def latent_brain_signals(self):
        if len(self.series) == 0 or self.series[0].latent_brain_signals is None:
            return None

        return np.array([s.latent_brain_signals for s in self.series])

    @property
    def latent_body_movements(self):
        if len(self.series) == 0 or self.series[0].latent_body_movements is None:
            return None

        return np.array([s.latent_body_movements for s in self.series])

    @property
    def observed_brain_signals(self):
        if len(self.series) == 0 or self.series[0].observed_brain_signals is None:
            return None

        return np.array([s.observed_brain_signals for s in self.series])

    @property
    def observed_body_movements(self):
        if len(self.series) == 0 or self.series[0].observed_body_movements is None:
            return None

        return np.array([s.observed_body_movements for s in self.series])

    def get_subset(self, indices: List[int]) -> BrainBodyDataset:
        return self.__class__(series=[self.series[i] for i in indices])

    def merge(self, dataset2: BrainBodyDataset) -> BrainBodyDataset:
        return self.__class__(series=self.series + dataset2.series)

    def normalize_per_subject(self):
        for s in self.series:
            s.normalize_per_subject()

    def shuffle(self):
        for s in self.series:
            s.shuffle_per_subject()

    @classmethod
    def from_samples(cls, samples: BrainBodySamples) -> BrainBodyDataset:
        series = []

        for i in range(samples.size):
            unbounded_coordination = samples.unbounded_coordination[
                i] if samples.unbounded_coordination is not None else None
            coordination = samples.coordination[i] if samples.coordination is not None else None
            latent_brain = samples.latent_brain[i] if samples.latent_brain is not None else None
            latent_body = samples.latent_body[i] if samples.latent_body is not None else None
            observed_brain = samples.observed_brain[i] if samples.observed_brain is not None else None
            observed_body = samples.observed_body[i] if samples.observed_body is not None else None

            s = BrainBodyDataSeries(
                uuid=f"{i}",
                unbounded_coordination=unbounded_coordination,
                coordination=coordination,
                latent_brain_signals=latent_brain,
                latent_body_movements=latent_body,
                observed_brain_signals=observed_brain,
                observed_body_movements=observed_body,
                team_score=0
            )

            series.append(s)

        evidence = cls(series)

        return evidence


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


class BrainBodyModelParameters:

    def __init__(self):
        self.sd_uc = None
        self.sd_c = None
        self.sd_brain = None
        self.sd_body = None
        self.sd_obs_brain = None
        self.sd_obs_body = None

    def reset(self):
        self.sd_uc = None
        self.sd_c = None
        self.sd_brain = None
        self.sd_body = None
        self.sd_obs_brain = None
        self.sd_obs_body = None
