from __future__ import annotations

from datetime import datetime
from typing import List, Optional, Dict

import arviz as az
import numpy as np
import pymc as pm
import pytensor.tensor as at
from scipy.stats import norm
from tqdm import tqdm

from coordination.common.distribution import beta
from coordination.common.utils import logit, sigmoid
from coordination.component.speech.common import SegmentedUtterance, VocalicsSparseSeries
from coordination.model.pgm2 import PGM2
from coordination.model.utils.beta_coordination_blending_latent_vocalics import \
    BetaCoordinationLatentVocalicsParticlesSummary, BetaCoordinationLatentVocalicsSamples, \
    BetaCoordinationLatentVocalicsDataSeries
from coordination.model.utils.coordination_blending_latent_vocalics import BaseF, BaseG
from coordination.model.utils.vocalics_semantic_model import VocalicsSemanticsModelParameters

# For numerical stability
EPSILON = 1e-6
MIN_COORDINATION = 2 * EPSILON
MAX_COORDINATION = 1 - MIN_COORDINATION

def serialized_logp(latent_variable: at.TensorVariable, coordination: at.TensorVariable,
                    sigma: at.TensorVariable, previous_vocalics_from_self: at.TensorVariable,
                    previous_vocalics_from_other: at.TensorVariable,
                    previous_vocalics_from_self_mask: at.TensorVariable,
                    previous_vocalics_from_other_mask: at.TensorVariable,
                    mask: at.TensorVariable):
    V = latent_variable

    A = V[:, previous_vocalics_from_self]
    B = V[:, previous_vocalics_from_other]

    M = mask[None, :]
    Ma = previous_vocalics_from_self_mask[None, :]
    Mb = previous_vocalics_from_other_mask[None, :]

    mean = ((B - A * Ma) * coordination[None, :] * Mb + A * Ma) * M * Mb

    return pm.logp(pm.Normal.dist(mu=mean, sigma=sigma, shape=V.shape), V).sum()


class VocalicsSemanticModel(
    PGM2[BetaCoordinationLatentVocalicsSamples, BetaCoordinationLatentVocalicsParticlesSummary]):

    def __init__(self,
                 initial_coordination: float,
                 num_vocalic_features: int,
                 num_subjects: int,
                 f: BaseF = BaseF(),
                 g: BaseG = BaseG(),
                 disable_self_dependency: bool = False):
        super().__init__()

        # Fix the number of subjects to 3 for now
        assert num_subjects == 3

        self.initial_coordination = initial_coordination
        self.num_vocalic_features = num_vocalic_features
        self.num_subjects = num_subjects
        self.f = f
        self.g = g
        self.disable_self_dependency = disable_self_dependency

        self.parameters = VocalicsSemanticsModelParameters()

        self._hyper_params = {
            "c0": initial_coordination,
            "#features": num_vocalic_features,
            "#speakers": num_subjects,
            "f": f.__repr__(),
            "g": g.__repr__(),
            "disable_self_dependency": disable_self_dependency
        }

    def sample(self, num_series: int, num_time_steps: int, seed: Optional[int], time_scale_density: float = 1,
               p_semantic_links: float = 0, *args, **kwargs) -> BetaCoordinationLatentVocalicsSamples:

        np.random.seed(seed)

        samples = BetaCoordinationLatentVocalicsSamples(self.num_subjects)
        self._generate_coordination_samples(num_series, num_time_steps, samples)
        samples.latent_vocalics = []
        samples.observed_vocalics = []
        samples.genders = np.ones((num_series, num_time_steps))
        samples.speech_semantic_links = np.zeros((num_series, num_time_steps))

        for i in tqdm(range(num_series), desc="Sampling Trial", position=0, leave=False):
            # Subjects A and B
            previous_self = [None] * num_time_steps
            previous_other = [None] * num_time_steps
            previous_time_per_speaker: Dict[int, int] = {}
            latent_vocalics_values = np.zeros((self.num_vocalic_features, num_time_steps))
            observed_vocalics_values = np.zeros((self.num_vocalic_features, num_time_steps))
            utterances: List[Optional[SegmentedUtterance]] = [None] * num_time_steps

            speakers = self._generate_random_speakers(num_time_steps, time_scale_density)
            mask = np.zeros(num_time_steps)

            for t in tqdm(range(num_time_steps), desc="Sampling Time Step", position=1, leave=False):
                current_coordination = samples.coordination[i, t]

                if speakers[t] is not None:
                    # Simple rule for gender. Male is even speakers and female odd ones.
                    samples.genders[i, t] = speakers[t] % 2

                    mask[t] = 1

                    previous_time_self = previous_time_per_speaker.get(speakers[t], None)
                    previous_time_other = None
                    for speaker, time in previous_time_per_speaker.items():
                        if speaker == speakers[t]:
                            continue

                        # Most recent vocalics from a different speaker
                        previous_time_other = time if previous_time_other is None else max(previous_time_other, time)

                    previous_value_self = None if previous_time_self is None else latent_vocalics_values[:,
                                                                                  previous_time_self]
                    previous_value_other = None if previous_time_other is None else latent_vocalics_values[:,
                                                                                    previous_time_other]

                    latent_vocalics_values[:, t] = self._sample_latent_vocalics(previous_value_self,
                                                                                previous_value_other,
                                                                                current_coordination)
                    observed_vocalics_values[:, t] = self._sample_observed_vocalics(latent_vocalics_values[:, t],
                                                                                    samples.genders[i, t])

                    previous_self[t] = previous_time_self
                    previous_other[t] = previous_time_other

                    # Dummy utterance
                    utterances[t] = SegmentedUtterance(f"Speaker {speakers[t]}", datetime.now(), datetime.now(), "")
                    previous_time_per_speaker[speakers[t]] = t

                    # Semantic link
                    u = np.random.rand()
                    if u <= p_semantic_links:
                        u = np.random.rand()
                        if u <= current_coordination:
                            samples.speech_semantic_links[i, t] = 1

            samples.latent_vocalics.append(VocalicsSparseSeries(utterances=utterances, previous_from_self=previous_self,
                                                                previous_from_other=previous_other,
                                                                values=latent_vocalics_values, mask=mask))
            samples.observed_vocalics.append(
                VocalicsSparseSeries(utterances=utterances, previous_from_self=previous_self,
                                     previous_from_other=previous_other,
                                     values=observed_vocalics_values, mask=mask))

        return samples

    def _generate_coordination_samples(self, num_samples: int, num_time_steps: int,
                                       samples: BetaCoordinationLatentVocalicsSamples):
        samples.unbounded_coordination = np.zeros((num_samples, num_time_steps))
        samples.coordination = np.zeros((num_samples, num_time_steps))

        for t in tqdm(range(num_time_steps), desc="Coordination", position=0, leave=False):
            if t == 0:
                samples.unbounded_coordination[:, 0] = logit(self.initial_coordination)
            else:
                samples.unbounded_coordination[:, t] = norm(loc=samples.unbounded_coordination[:, t - 1],
                                                            scale=self.parameters.sd_uc).rvs()

            # The variance of a beta distribution, cannot be bigger than m * (1 - m). Therefore, we
            # constrain the sampled from the unbounded distribution such that we cannot generate
            # beta distributions with impossible means when we compute coordination.
            clipped_uc = np.clip(sigmoid(samples.unbounded_coordination[:, t]), MIN_COORDINATION, MAX_COORDINATION)
            clipped_vc = np.minimum(SD_C ** 2, 0.5 * clipped_uc * (1 - clipped_uc))
            samples.coordination[:, t] = beta(clipped_uc, clipped_vc).rvs()

        # # TODO: Coordination is unbounded
        # samples.coordination = sigmoid(samples.unbounded_coordination)

    def _generate_random_speakers(self, num_time_steps: int, time_scale_density: float) -> List[Optional[int]]:
        # We always change speakers between time steps when generating vocalics
        transition_matrix = 1 - np.eye(self.num_subjects + 1)

        transition_matrix *= time_scale_density / (self.num_subjects - 1)
        transition_matrix[:, -1] = 1 - time_scale_density

        prior = np.ones(self.num_subjects + 1) * time_scale_density / self.num_subjects
        prior[-1] = 1 - time_scale_density
        transition_matrix[-1] = prior

        initial_speaker = np.random.choice(self.num_subjects + 1, 1, p=prior)[0]
        initial_speaker = None if initial_speaker == self.num_subjects else initial_speaker
        speakers = [initial_speaker]

        for t in range(1, num_time_steps):
            probabilities = transition_matrix[self.num_subjects] if speakers[t - 1] is None else transition_matrix[
                speakers[t - 1]]
            speaker = np.random.choice(self.num_subjects + 1, 1, p=probabilities)[0]
            speaker = None if speaker == self.num_subjects else speaker
            speakers.append(speaker)

        return speakers

    def _sample_latent_vocalics(self, previous_self: Optional[float], previous_other: Optional[float],
                                coordination: float) -> np.ndarray:
        if previous_other is None:
            if previous_self is None or self.disable_self_dependency:
                distribution = norm(loc=np.zeros(self.num_vocalic_features), scale=self.parameters.sd_vocalics)
            else:
                distribution = norm(loc=self.f(previous_self, 0), scale=self.parameters.sd_vocalics)
        else:
            if previous_self is None or self.disable_self_dependency:
                D = self.f(previous_other, 1)
                distribution = norm(loc=D * coordination, scale=self.parameters.sd_vocalics)
            else:
                D = self.f(previous_other, 1) - self.f(previous_self, 0)
                distribution = norm(loc=D * coordination + previous_self, scale=self.parameters.sd_vocalics)

        # # # TODO: temporary just to test the model inference
        # distribution = norm(loc=np.ones(self.num_vocalic_features) * coordination, scale=self.parameters.sd_vocalics)

        return distribution.rvs()

    def _sample_observed_vocalics(self, latent_vocalics: np.array, gender: int) -> np.ndarray:
        return norm(loc=self.g(latent_vocalics), scale=self.parameters.sd_obs_vocalics).rvs()

    def fit(self, evidence: BetaCoordinationLatentVocalicsDataSeries, burn_in: int, num_samples: int, num_chains: int,
            seed: Optional[int], num_jobs: int = 1) -> az.InferenceData:

        model = self._define_pymc_model(evidence)
        with model:
            idata = pm.sample(num_samples, init="adapt_diag", tune=burn_in, chains=num_chains, random_seed=seed,
                              cores=num_jobs)

            return idata

    def _define_pymc_model(self, evidence: BetaCoordinationLatentVocalicsDataSeries):
        coords = {"vocalic_feature": np.arange(self.num_vocalic_features), "time": np.arange(evidence.num_time_steps)}
        model = pm.Model(coords=coords)

        with model:
            # Parameters to be inferred and shared among time series of brain signal and body movement.
            sd_uc = pm.HalfNormal(name="sd_uc", sigma=1, size=1, observed=self.parameters.sd_uc)
            sd_c = pm.HalfNormal(name="sd_c", sigma=1, size=1, observed=self.parameters.sd_c)
            sd_vocalics = pm.HalfNormal(name="sd_vocalics", sigma=1, size=1, observed=self.parameters.sd_vocalics)
            sd_obs_vocalics = pm.HalfNormal(name="sd_obs_vocalics", sigma=1, size=1,
                                            observed=self.parameters.sd_obs_vocalics)

            prior = pm.Normal.dist(mu=logit(self.initial_coordination), sigma=sd_uc)
            unbounded_coordination = pm.GaussianRandomWalk("unbounded_coordination",
                                                           init_dist=prior,
                                                           sigma=sd_uc,
                                                           dims=["time"])

            mean_coordination = pm.Deterministic("mean_coordination", pm.math.sigmoid(unbounded_coordination),
                                                 dims=["time"])

            mean_coordination_clipped = pm.Deterministic(f"mean_coordination_clipped",
                                                         pm.math.clip(mean_coordination, MIN_COORDINATION,
                                                                      MAX_COORDINATION), dims=["time"])
            sd_c_clipped = pm.Deterministic("sd_c_clipped", pm.math.minimum(sd_c, 0.5 * mean_coordination_clipped * (
                    1 - mean_coordination_clipped)))

            coordination = pm.Beta(name="coordination", mu=mean_coordination_clipped, sigma=sd_c_clipped,
                                   dims=["time"])

            vocalics_params = (coordination,
                               sd_vocalics,
                               at.constant(evidence.previous_vocalics_from_self),
                               at.constant(evidence.previous_vocalics_from_other),
                               at.constant(evidence.previous_vocalics_from_self_mask),
                               at.constant(evidence.previous_vocalics_from_other_mask),
                               at.constant(evidence.vocalics_mask))

            latent_vocalics = pm.DensityDist("latent_vocalics", *vocalics_params,
                                             logp=serialized_logp,
                                             dims=["vocalic_feature", "time"])

            pm.Normal(name="observed_vocalics", mu=latent_vocalics, sigma=sd_obs_vocalics,
                      dims=("vocalic_feature", "time"), observed=evidence.observed_vocalics.values)

            pm.Bernoulli("semantic_links", p=coordination[evidence.speech_semantic_links_times],
                         observed=evidence.speech_semantic_links_vector_of_ones)

        return model
