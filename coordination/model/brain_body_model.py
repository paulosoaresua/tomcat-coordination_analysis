from __future__ import annotations
from typing import Any, List, Optional, Tuple

import numpy as np
import pymc as pm
import pytensor.tensor as at
from scipy.stats import norm

from coordination.common.utils import logit, sigmoid
from coordination.common.distribution import beta
from coordination.model.utils.coordination_blending_latent_vocalics import BaseF, BaseG

from coordination.model.pgm2 import PGM2
from coordination.model.utils.brain_body_model import BrainBodyDataset, BrainBodySamples, BrainBodyParticlesSummary, \
    BrainBodyModelParameters

# For numerical stability
EPSILON = 1e-6
MIN_COORDINATION = 2 * EPSILON
MAX_COORDINATION = 1 - MIN_COORDINATION

# Subjects influence others equally. This can be adapted to consider a drift in influence power in future models.
W = np.array([0.5, 0.5])

# We fix standard deviation for coordination. The unbounded coordination standard deviation will be fit to the data.
# OBS: I declare this variable as global because for some unknown reason, if this variable is declared as a local
# attribute of the model class, NUTS performs 3 to 4 times slower when I pass this variable to the pm.math.minimum
# function.
SD_C = 0.1


def multi_influencers_mixture_logp(latent_variable: at.TensorVariable, coordination: at.TensorVariable,
                                   sigma: at.TensorVariable):
    latent_prev = latent_variable[:, :, :, :-1]
    latent_curr = latent_variable[:, :, :, 1:]
    c_i = coordination[:, 1:][:, None, :]

    # mixture_weights = np.array([0.5, 0.5])
    # num_subjects = 3

    # num_trials, _, num_channels, num_time_steps = latent_variable.shape
    # shape = (num_trials, num_channels, num_time_steps - 1)
    num_trials, num_subjects, num_channels, num_time_steps = latent_variable.shape
    shape = (num_trials, num_channels, num_time_steps - 1)

    num_subjects = num_subjects.eval()
    total_logp = 0
    for s1 in range(num_subjects):
        w = 0
        likelihood_per_subject = 0
        for s2 in range(num_subjects):
            if s1 == s2:
                continue

            logp_s1_from_s2 = pm.logp(
                pm.Normal.dist(mu=latent_prev[:, s2] * c_i + (1 - c_i) * latent_prev[:, s1], sigma=sigma, shape=shape),
                latent_curr[:, s1])

            likelihood_per_subject += W[w] * pm.math.exp(logp_s1_from_s2)
            w += 1

        total_logp += pm.math.log(likelihood_per_subject).sum()

    init_dist = pm.Normal.dist(mu=0, sigma=1, shape=(
        num_trials, num_subjects, num_channels))

    return at.sum(pm.logp(init_dist, latent_variable[:, :, :, 0])) + total_logp


def unbounded_coordination_drift_logp(unbounded_coordination: at.TensorVariable, sigma: at.TensorVariable,
                                      initial_value: at.TensorVariable):
    """
    The GaussianRandomWalk class is issuing a warning when it's created for a RV. So I am encapsulating on a custom
    distribution to avoid the warning until the issue is fixed.

    Issue: https://github.com/pymc-devs/pymc/pull/6407
    """
    N, T = unbounded_coordination.shape

    prior = pm.Normal.dist(mu=initial_value, sigma=1, shape=N)
    return pm.logp(pm.GaussianRandomWalk.dist(init_dist=prior,
                                              sigma=sigma, shape=(N, T)), unbounded_coordination).sum()


class BrainBodyModel(PGM2[BrainBodySamples, BrainBodyParticlesSummary]):

    # TODO: I disabled body movement for now

    def __init__(self,
                 initial_coordination: float,
                 num_brain_channels: int,
                 num_subjects: int,
                 f: BaseF = BaseF(),
                 g: BaseG = BaseG(),
                 disable_self_dependency: bool = False):
        super().__init__()

        # Fix the number of subjects to 3 for now
        assert num_subjects == 3

        self.initial_coordination = initial_coordination
        self.num_brain_channels = num_brain_channels
        self.num_subjects = num_subjects
        self.f = f
        self.g = g
        self.disable_self_dependency = disable_self_dependency

        # We assume all subjects contribute with the same weight.
        # This parameter can be a random variable in future work.
        self.mixture_weigths = np.ones(num_subjects - 1) / (num_subjects - 1)

        self.parameters = BrainBodyModelParameters()

        self._hyper_params = {
            "c0": initial_coordination,
            "#features": num_brain_channels,
            "#speakers": num_subjects,
            "f": f.__repr__(),
            "g": g.__repr__(),
            "disable_self_dependency": disable_self_dependency
        }

    def sample(self, num_series: int, num_time_steps: int, seed: Optional[int], *args, **kwargs) -> BrainBodySamples:
        samples = BrainBodySamples()
        samples.unbounded_coordination = np.zeros((num_series, num_time_steps))
        samples.coordination = np.zeros((num_series, num_time_steps))
        samples.latent_brain = np.zeros((num_series, self.num_subjects, self.num_brain_channels, num_time_steps))
        samples.latent_body = np.zeros((num_series, self.num_subjects, 1, num_time_steps))
        samples.observed_brain = np.zeros((num_series, self.num_subjects, self.num_brain_channels, num_time_steps))
        samples.observed_body = np.zeros((num_series, self.num_subjects, 1, num_time_steps))

        for t in range(num_time_steps):
            if t == 0:
                samples.unbounded_coordination[:, t] = logit(self.initial_coordination)
            else:
                samples.unbounded_coordination[:, t] = norm(loc=samples.unbounded_coordination[:, t - 1],
                                                            scale=self.parameters.sd_uc).rvs()

            # We clip the mean and variance to make sure we have a proper Beta distribution from which we can sample
            # from.
            clipped_uc = np.clip(sigmoid(samples.unbounded_coordination[:, t]), MIN_COORDINATION, MAX_COORDINATION)
            clipped_vc = np.minimum(SD_C ** 2, 0.5 * clipped_uc * (1 - clipped_uc))
            samples.coordination[:, t] = beta(clipped_uc, clipped_vc).rvs()

            if t == 0:
                samples.latent_brain[:, :, :, 0] = norm(loc=0, scale=1).rvs(
                    size=(num_series, self.num_subjects, self.num_brain_channels))
                samples.latent_body[:, :, :, 0] = norm(loc=0, scale=1).rvs(
                    size=(num_series, self.num_subjects, 1))
            else:
                # Subject 0
                for subject1 in range(self.num_subjects):
                    brain_samples_from_mixture = []
                    body_samples_from_mixture = []
                    for subject2 in range(self.num_subjects):
                        if subject1 == subject2:
                            continue

                        mu = samples.latent_brain[:, subject2, :, t - 1] * samples.coordination[:, t][:, None] + \
                             samples.latent_brain[:, subject1, :, t - 1] * (1 - samples.coordination[:, t][:, None])
                        brain_samples_from_mixture.append(norm(loc=mu, scale=self.parameters.sd_brain).rvs())

                        mu = samples.latent_body[:, subject2, :, t - 1] * samples.coordination[:, t][:, None] + \
                             samples.latent_body[:, subject1, :, t - 1] * (1 - samples.coordination[:, t][:, None])
                        body_samples_from_mixture.append(norm(loc=mu, scale=self.parameters.sd_brain).rvs())

                    brain_influencer_indices = np.random.choice(a=np.arange(self.num_subjects - 1), size=num_series,
                                                                p=self.mixture_weigths)
                    body_influencer_indices = np.random.choice(a=np.arange(self.num_subjects - 1), size=num_series,
                                                               p=self.mixture_weigths)
                    for i in range(num_series):
                        influencer_idx = int(brain_influencer_indices[i])
                        if isinstance(brain_samples_from_mixture[influencer_idx], float):
                            samples.latent_brain[i, subject1, :, t] = brain_samples_from_mixture[influencer_idx]
                        else:
                            samples.latent_brain[i, subject1, :, t] = brain_samples_from_mixture[influencer_idx][i]

                        influencer_idx = int(body_influencer_indices[i])
                        if isinstance(body_samples_from_mixture[influencer_idx], float):
                            samples.latent_body[i, subject1, :, t] = body_samples_from_mixture[influencer_idx]
                        else:
                            samples.latent_body[i, subject1, :, t] = body_samples_from_mixture[influencer_idx][i]

            samples.observed_brain[:, :, :, t] = norm(loc=samples.latent_brain[:, :, :, t],
                                                      scale=self.parameters.sd_obs_brain).rvs()
            samples.observed_body[:, :, :, t] = norm(loc=samples.latent_body[:, :, :, t],
                                                     scale=self.parameters.sd_obs_body).rvs()

        return samples

    def fit(self, evidence: BrainBodyDataset, burn_in: int, num_samples: int, num_chains: int,
            seed: Optional[int], retain_every: int = 1, num_jobs: int = 1) -> Any:

        model = self._define_pymc_model(evidence)
        with model:
            idata = pm.sample(num_samples, init="adapt_diag", tune=burn_in, chains=num_chains, random_seed=seed,
                              cores=num_jobs)

            self.parameters.sd_uc = idata.posterior["sd_uc"][::retain_every].mean(dim=["chain", "draw"]).to_numpy()
            self.parameters.sd_brain = idata.posterior["sd_brain"][::retain_every].mean(
                dim=["chain", "draw"]).to_numpy()
            # self.parameters.sd_body = idata.posterior["sd_body"][::retain_every].mean(dim=["chain", "draw"]).to_numpy()
            self.parameters.sd_obs_brain = idata.posterior["sd_obs_brain"][::retain_every].mean(
                dim=["chain", "draw"]).to_numpy()
            # self.parameters.sd_obs_body = idata.posterior["sd_obs_body"][::retain_every].mean(
            #     dim=["chain", "draw"]).to_numpy()

            return idata

    def _define_pymc_model(self, evidence: BrainBodyDataset):
        coords = {"trial": np.arange(evidence.num_trials), "subject": np.arange(self.num_subjects),
                  "brain_channel": np.arange(self.num_brain_channels), "body_feature": np.arange(1),
                  "time": np.arange(evidence.num_time_steps)}
        model = pm.Model(coords=coords)

        with model:
            # Parameters to be inferred and shared among time series of brain signal and body movement.
            sd_uc = pm.HalfNormal(name="sd_uc", sigma=1, size=1, observed=self.parameters.sd_uc)
            sd_brain = pm.HalfNormal(name="sd_brain", sigma=1, size=1, observed=self.parameters.sd_brain)
            # sd_body = pm.HalfNormal(name="sd_body", sigma=1, size=1, observed=self.parameters.sd_body)
            sd_obs_brain = pm.HalfNormal(name="sd_obs_brain", sigma=1, size=1, observed=self.parameters.sd_obs_brain)
            # sd_obs_body = pm.HalfNormal(name="sd_obs_body", sigma=1, size=1, observed=self.parameters.sd_obs_body)

            N = evidence.num_trials
            T = evidence.num_time_steps

            unbounded_coordination_params = (at.as_tensor_variable(sd_uc),
                                             at.constant(logit(self.initial_coordination)))
            unbounded_coordination = pm.DensityDist("unbounded_coordination", *unbounded_coordination_params,
                                                    logp=unbounded_coordination_drift_logp,
                                                    initval=np.ones((N, T)) * logit(self.initial_coordination),
                                                    dims=["trial", "time"],
                                                    observed=evidence.unbounded_coordination)

            mean_coordination = pm.Deterministic("mean_coordination", pm.math.sigmoid(unbounded_coordination),
                                                 dims=["trial", "time"])
            mean_coordination_clipped = pm.Deterministic(f"mean_coordination_clipped",
                                                         pm.math.clip(mean_coordination, MIN_COORDINATION,
                                                                      MAX_COORDINATION), dims=["trial", "time"])
            sd_c_clipped = pm.Deterministic("sd_c_clipped", pm.math.minimum(SD_C, 0.5 * mean_coordination_clipped * (
                    1 - mean_coordination_clipped)))

            coordination = pm.Beta(name="coordination", mu=mean_coordination_clipped, sigma=sd_c_clipped,
                                   dims=["trial", "time"], observed=evidence.coordination)

            brain_params = (at.as_tensor_variable(coordination),
                            at.as_tensor_variable(sd_brain))
            latent_brain = pm.DensityDist("latent_brain", *brain_params,
                                          logp=multi_influencers_mixture_logp,
                                          dims=["trial", "subject", "brain_channel", "time"],
                                          observed=evidence.latent_brain_signals)

            # body_params = (at.as_tensor_variable(coordination),
            #                at.as_tensor_variable(sd_body))
            # latent_body = pm.DensityDist("latent_body", *body_params,
            #                              logp=multi_influencers_mixture_logp,
            #                              dims=["trial", "subject", "body_feature", "time"],
            #                              observed=evidence.latent_body_movements)

            pm.Normal(name="observed_brain", mu=latent_brain, sigma=sd_obs_brain,
                      dims=("trial", "subject", "brain_channel", "time"), observed=evidence.observed_brain_signals)
            # pm.Normal(name="observed_body", mu=latent_body, sigma=sd_obs_body,
            #           observed=evidence.observed_body_movements)

        return model

    def predict(self, evidence: BrainBodyDataset, num_samples: int, burn_in: int, num_chains: int, seed: Optional[int],
                retain_every: int = 1, num_jobs: int = 1) -> List[BrainBodyParticlesSummary]:

        summaries = []
        for i in range(evidence.num_trials):
            model = self._define_pymc_model(evidence.get_subset([i]))
            with model:
                idata = pm.sample(num_samples, init="adapt_diag", tune=burn_in, chains=num_chains, random_seed=seed,
                                  cores=num_jobs)

                summary = BrainBodyParticlesSummary.from_inference_data(idata, retain_every)

                summaries.append(summary)

        return summaries
