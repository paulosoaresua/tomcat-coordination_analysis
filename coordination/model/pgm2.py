from __future__ import annotations
from typing import Any, Dict, Generic, List, Optional, TypeVar

from multiprocessing import Pool
import os
import pickle

import numpy as np
from sklearn.base import BaseEstimator
from tqdm import tqdm

from coordination.callback.callback import Callback
from coordination.common.log import BaseLogger
from coordination.common.dataset import EvidenceDataset, EvidenceDataSeries
from coordination.common.parallelism import display_inner_progress_bar
from coordination.common.utils import set_seed
from coordination.model.particle_filter import Particles, ParticleFilter


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

    def reset_parameters(self):
        raise NotImplementedError

    def sample(self, num_series: int, num_time_steps: int, seed: Optional[int]) -> SP:
        raise NotImplementedError

    def fit(self, evidence: EvidenceDataset, num_samples: int, burn_in: int, num_chains: int,
            seed: Optional[int], num_jobs: int = 1):
        raise NotImplementedError

    def predict(self, evidence: EvidenceDataset, num_samples: int, burn_in: int, num_chains: int, seed: Optional[int],
                num_jobs: int = 1) -> List[S]:
        raise NotImplementedError
