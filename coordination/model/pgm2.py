from __future__ import annotations
from typing import Any, Generic, List, Optional, TypeVar

import os
import pickle

from sklearn.base import BaseEstimator

from coordination.common.dataset import EvidenceDataset


class Samples:

    @property
    def size(self):
        raise NotImplementedError


class ParticlesSummary:
    pass


SP = TypeVar('SP')
S = TypeVar('S')


class PGM2(BaseEstimator, Generic[SP, S]):
    def __init__(self):
        super().__init__()

    def save(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)

        with open(f"{out_dir}/model.pkl", "wb") as f:
            pickle.dump(self, f)

    def sample(self, num_series: int, num_time_steps: int, seed: Optional[int]) -> SP:
        raise NotImplementedError

    def fit(self, evidence: EvidenceDataset, burn_in: int, num_samples: int, num_chains: int,
            seed: Optional[int], retain_every: int, num_jobs: int):
        raise NotImplementedError

    def predict(self, evidence: EvidenceDataset, num_samples: int, burn_in: int, num_chains: int, seed: Optional[int],
                retain_every: int, num_jobs: int) -> List[S]:
        raise NotImplementedError
