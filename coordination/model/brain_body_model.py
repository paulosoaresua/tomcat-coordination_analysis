from __future__ import annotations
from typing import List, Optional

import numpy as np
import pymc3 as pm
import theano
import arviz as az
from theano import tensor as tt
from scipy.stats import norm

from coordination.common.utils import logit, sigmoid
from coordination.common.distribution import beta
from coordination.model.utils.coordination_blending_latent_vocalics import BaseF, BaseG

from coordination.model.pgm2 import PGM2
from coordination.model.utils.brain_body_model import BrainBodyDataset, BrainBodySamples, BrainBodyParticlesSummary

# For numerical stability
EPSILON = 1e-6
MIN_COORDINATION = 2 * EPSILON
MAX_COORDINATION = 1 - MIN_COORDINATION


class LatentComponentRandomWalk(pm.GaussianRandomWalk):

    def __init__(self, tau=None, init=pm.Flat.dist(), sigma=None, mu=0.0, sd=None, coordination=None, num_time_steps=0,
                 num_channels=1, *args, **kwargs):
        super().__init__(tau=tau, init=init, sigma=sigma, mu=mu, sd=sd, *args, **kwargs)

        # Coordination
        self.num_time_steps = num_time_steps
        self.num_channels = num_channels
        self.c = tt.as_tensor_variable(coordination)

    def logp(self, x):
        x_im1 = x[:-1]
        x_i = x[1:]
        c_i = self.c[1:]
        mu, sigma = self._mu_and_sigma(self.mu, self.sigma)

        T = self.shape[0]
        C = self.shape[2]

        p1 = pm.Mixture.dist(w=np.array([0.5, 0.5]), comp_dists=[
            pm.Normal.dist(mu=x_im1[:, 1] * c_i + (1 - c_i) * x_im1[:, 0] + mu, sigma=sigma, shape=(T - 1, C)),
            pm.Normal.dist(mu=x_im1[:, 2] * c_i + (1 - c_i) * x_im1[:, 0] + mu, sigma=sigma, shape=(T - 1, C)),
        ], shape=(T - 1, C)).logp(x_i[:, 0])

        p2 = pm.Mixture.dist(w=np.array([0.5, 0.5]), comp_dists=[
            pm.Normal.dist(mu=x_im1[:, 0] * c_i + (1 - c_i) * x_im1[:, 1] + mu, sigma=sigma, shape=(T - 1, C)),
            pm.Normal.dist(mu=x_im1[:, 2] * c_i + (1 - c_i) * x_im1[:, 1] + mu, sigma=sigma, shape=(T - 1, C)),
        ], shape=(T - 1, C)).logp(x_i[:, 1])

        p3 = pm.Mixture.dist(w=np.array([0.5, 0.5]), comp_dists=[
            pm.Normal.dist(mu=x_im1[:, 0] * c_i + (1 - c_i) * x_im1[:, 2] + mu, sigma=sigma, shape=(T - 1, C)),
            pm.Normal.dist(mu=x_im1[:, 1] * c_i + (1 - c_i) * x_im1[:, 2] + mu, sigma=sigma, shape=(T - 1, C)),
        ], shape=(T - 1, C)).logp(x_i[:, 2])

        return tt.sum(self.init.logp(x[0])) + tt.sum(p1) + tt.sum(p2) + tt.sum(p3)


class BrainBodyModel(PGM2[BrainBodySamples, BrainBodyParticlesSummary]):

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

        self.sd_c = 1e-4

        self._hyper_params = {
            "c0": initial_coordination,
            "#features": num_brain_channels,
            "#speakers": num_subjects,
            "f": f.__repr__(),
            "g": g.__repr__(),
            "disable_self_dependency": disable_self_dependency
        }

        # Model & shared variable definition
        self._model = pm.Model()
        self._define_model_parameters()

    def reset_parameters(self):
        self.sd_uc = None
        self.sd_brain = None
        self.sd_body = None
        self.sd_obs_brain = None
        self.sd_obs_body = None

    def _define_model_parameters(self):
        with self._model:
            # Shared data. Used to set the value of the parameters of the model after it's trained, or for
            # synthetic data generation.
            self._sd_uc_value = pm.Data('sd_uc_value', None)
            self._sd_brain_value = pm.Data('sd_brain_value', None)
            self._sd_body_value = pm.Data('sd_body_value', None)
            self._sd_obs_brain_value = pm.Data('sd_obs_brain_value', None)
            self._sd_obs_body_value = pm.Data('sd_obs_body_value', None)

            # Variances to be inferred and shared among time series of brain signal and body movement
            self._sd_uc = pm.HalfNormal(name="sd_uc", sigma=1, observed=self._sd_uc_value)
            self._sd_brain = pm.HalfNormal(name="sd_brain", sigma=1, observed=self._sd_brain_value)
            self._sd_body = pm.HalfNormal(name="sd_body", sigma=1, observed=self._sd_body_value)
            self._sd_obs_brain = pm.HalfNormal(name="sd_obs_brain", sigma=1, observed=self._sd_obs_brain_value)
            self._sd_obs_body = pm.HalfNormal(name="sd_obs_body", sigma=1, observed=self._sd_obs_body_value)

    def _define_model_variables(self, num_series: int, num_time_steps: int):
        """
        This function depends on the evidence because we create one variable per observed time series.
        """

        with self._model:
            T = num_time_steps
            S = self.num_subjects
            C = self.num_brain_channels

            for i in range(num_series):
                unbounded_coordination_value = pm.Data(f"unbounded_coordination_value_{i}", None)
                coordination_value = pm.Data(f"coordination_value_{i}", None)
                latent_brain_value = pm.Data(f"latent_brain_value_{i}", None)
                latent_body_value = pm.Data(f"latent_body_value_{i}", None)
                observed_brain_value = pm.Data(f"observed_brain_value_{i}", None)
                observed_body_value = pm.Data(f"observed_body_value_{i}", None)

                unbounded_coordination = pm.GaussianRandomWalk(name=f"unbounded_coordination_{i}", sigma=self._sd_uc,
                                                               shape=(T, 1), testval=logit(self.initial_coordination),
                                                               observed=unbounded_coordination_value)

                mean_coordination = pm.Deterministic(f"mean_coordination_{i}", pm.math.sigmoid(unbounded_coordination))

                # Don't allow the mean to be zero or one to avoid issues with the beta distribution.
                mean_coordination_clipped = pm.Deterministic(f"mean_coordination_clipped_{i}",
                                                             pm.math.clip(mean_coordination, MIN_COORDINATION,
                                                                          MAX_COORDINATION))

                # If the mean of the beta distribution is too small, use a smaller variance to obtain a valid distribution.
                sd_c_clipped = pm.Deterministic(f"sd_c_clipped_{i}",
                                                pm.math.minimum(self.sd_c, 2 * mean_coordination_clipped * (
                                                        1 - mean_coordination_clipped)))
                coordination = pm.Beta(name=f"coordination_{i}", mu=mean_coordination_clipped, sigma=sd_c_clipped,
                                       shape=(T, 1), observed=coordination_value)

                latent_brain_prior = pm.Normal.dist(mu=0, sigma=1, shape=(S, C))
                latent_brain = LatentComponentRandomWalk(name=f"latent_brain_{i}", init=latent_brain_prior,
                                                         sigma=self._sd_brain,
                                                         shape=(T, S, C), coordination=coordination,
                                                         observed=latent_brain_value)

                latent_body_prior = pm.Normal.dist(mu=0, sigma=1, shape=(S, 1))
                latent_body = LatentComponentRandomWalk(name=f"latent_body_{i}", init=latent_body_prior,
                                                        sigma=self._sd_brain,
                                                        shape=(T, S, 1), coordination=coordination,
                                                        observed=latent_body_value)

                observed_brain = pm.Normal(name=f"observed_brain_{i}", mu=latent_brain, sigma=self._sd_obs_brain,
                                           shape=(T, S, C), observed=observed_brain_value)
                observed_body = pm.Normal(name=f"observed_body_{i}", mu=latent_body, sigma=self._sd_obs_body,
                                          shape=(T, S, 1), observed=observed_body_value)

    def sample(self, num_series: int, num_time_steps: int, seed: Optional[int]) -> BrainBodySamples:
        samples = BrainBodySamples(self.num_subjects)
        samples.unbounded_coordination = np.zeros((num_series, num_time_steps, 1))
        samples.coordination = np.zeros((num_series, num_time_steps, 1))
        samples.latent_brain = np.zeros((num_series, num_time_steps, self.num_subjects, self.num_brain_channels))
        samples.latent_body = np.zeros((num_series, num_time_steps, self.num_subjects, 1))
        samples.observed_brain = np.zeros((num_series, num_time_steps, self.num_subjects, self.num_brain_channels))
        samples.observed_body = np.zeros((num_series, num_time_steps, self.num_subjects, 1))

        for t in range(num_time_steps):
            if t == 0:
                samples.unbounded_coordination[:, t] = logit(self.initial_coordination)
            else:
                samples.unbounded_coordination[:, t] = norm(loc=samples.unbounded_coordination[:, t - 1],
                                                            scale=self.sd_uc).rvs()

            clipped_uc = np.clip(sigmoid(samples.unbounded_coordination[:, t]), MIN_COORDINATION, MAX_COORDINATION)
            clipped_vc = np.minimum(self.sd_c ** 2, 2 * clipped_uc * (1 - clipped_uc))
            samples.coordination[:, t] = beta(clipped_uc, clipped_vc).rvs()

            if t == 0:
                samples.latent_brain[:, 0] = norm(loc=0, scale=1).rvs(
                    size=(num_series, self.num_subjects, self.num_brain_channels))
                samples.latent_body[:, 0] = norm(loc=0, scale=1).rvs(
                    size=(num_series, self.num_subjects, 1))
            else:
                # Subject 0
                u = np.random.rand(num_series)
                m1 = samples.latent_brain[:, t - 1, 1] * samples.coordination[:, t] + \
                     samples.latent_brain[:, t - 1, 0] * (1 - samples.coordination[:, t])
                m2 = samples.latent_brain[:, t - 1, 2] * samples.coordination[:, t] + \
                     samples.latent_brain[:, t - 1, 0] * (1 - samples.coordination[:, t])
                comp1 = norm(loc=m1, scale=self.sd_brain).rvs()
                comp2 = norm(loc=m2, scale=self.sd_brain).rvs()
                samples.latent_brain[:, t, 0] = np.where(u <= 1 / 2, comp1, comp2)

                u = np.random.rand(num_series)
                m1 = samples.latent_body[:, t - 1, 1] * samples.coordination[:, t] + \
                     samples.latent_body[:, t - 1, 0] * (1 - samples.coordination[:, t])
                m2 = samples.latent_body[:, t - 1, 2] * samples.coordination[:, t] + \
                     samples.latent_body[:, t - 1, 0] * (1 - samples.coordination[:, t])
                comp1 = norm(loc=m1, scale=self.sd_body).rvs()
                comp2 = norm(loc=m2, scale=self.sd_body).rvs()
                samples.latent_body[:, t, 0] = np.where(u <= 1 / 2, comp1, comp2)

                # Subject 1
                u = np.random.rand(num_series)
                m1 = samples.latent_brain[:, t - 1, 0] * samples.coordination[:, t] + \
                     samples.latent_brain[:, t - 1, 1] * (1 - samples.coordination[:, t])
                m2 = samples.latent_brain[:, t - 1, 2] * samples.coordination[:, t] + \
                     samples.latent_brain[:, t - 1, 1] * (1 - samples.coordination[:, t])
                comp1 = norm(loc=m1, scale=self.sd_brain).rvs()
                comp2 = norm(loc=m2, scale=self.sd_brain).rvs()
                samples.latent_brain[:, t, 1] = np.where(u <= 1 / 2, comp1, comp2)

                u = np.random.rand(num_series)
                m1 = samples.latent_body[:, t - 1, 0] * samples.coordination[:, t] + \
                     samples.latent_body[:, t - 1, 1] * (1 - samples.coordination[:, t])
                m2 = samples.latent_body[:, t - 1, 2] * samples.coordination[:, t] + \
                     samples.latent_body[:, t - 1, 1] * (1 - samples.coordination[:, t])
                comp1 = norm(loc=m1, scale=self.sd_body).rvs()
                comp2 = norm(loc=m2, scale=self.sd_body).rvs()
                samples.latent_body[:, t, 1] = np.where(u <= 1 / 2, comp1, comp2)

                # Subject 2
                u = np.random.rand(num_series)
                m1 = samples.latent_brain[:, t - 1, 0] * samples.coordination[:, t] + \
                     samples.latent_brain[:, t - 1, 2] * (1 - samples.coordination[:, t])
                m2 = samples.latent_brain[:, t - 1, 1] * samples.coordination[:, t] + \
                     samples.latent_brain[:, t - 1, 2] * (1 - samples.coordination[:, t])
                comp1 = norm(loc=m1, scale=self.sd_brain).rvs()
                comp2 = norm(loc=m2, scale=self.sd_brain).rvs()
                samples.latent_brain[:, t, 2] = np.where(u <= 1 / 2, comp1, comp2)

                u = np.random.rand(num_series)
                m1 = samples.latent_body[:, t - 1, 0] * samples.coordination[:, t] + \
                     samples.latent_body[:, t - 1, 2] * (1 - samples.coordination[:, t])
                m2 = samples.latent_body[:, t - 1, 1] * samples.coordination[:, t] + \
                     samples.latent_body[:, t - 1, 2] * (1 - samples.coordination[:, t])
                comp1 = norm(loc=m1, scale=self.sd_body).rvs()
                comp2 = norm(loc=m2, scale=self.sd_body).rvs()
                samples.latent_body[:, t, 2] = np.where(u <= 1 / 2, comp1, comp2)

            samples.observed_brain[:, t] = norm(loc=samples.latent_brain[:, t], scale=self.sd_obs_brain).rvs()
            samples.observed_body[:, t] = norm(loc=samples.latent_body[:, t], scale=self.sd_obs_body).rvs()

        return samples

    def fit(self, evidence: BrainBodyDataset, num_samples: int, burn_in: int, num_chains: int,
            seed: Optional[int], num_jobs: int = 1):

        self._define_model_variables(evidence.num_trials, evidence.num_time_steps)

        with self._model:
            for i in range(evidence.num_trials):
                pm.set_data({
                    f"unbounded_coordination_value_{i}": evidence.series[i].unbounded_coordination,
                    f"coordination_value_{i}": evidence.series[i].coordination,
                    f"latent_brain_value_{i}": evidence.series[i].latent_brain_signals,
                    f"latent_body_value_{i}": evidence.series[i].latent_body_movements,
                    f"observed_brain_value_{i}": evidence.series[i].observed_brain_signals,
                    f"observed_body_value_{i}": evidence.series[i].observed_body_movements
                })

            theano.config.floatX = 'float64'
            trace = pm.sample(num_samples, init="adapt_diag", return_inferencedata=False, tune=burn_in,
                              chains=num_chains, random_seed=seed)

            self.sd_uc = trace["sd_uc"][::5].mean(axis=0)
            self.sd_brain = trace["sd_brain"][::5].mean(axis=0)
            self.sd_body = trace["sd_body"][::5].mean(axis=0)
            self.sd_obs_brain = trace["sd_obs_brain"][::5].mean(axis=0)
            self.sd_obs_body = trace["sd_obs_brain"][::5].mean(axis=0)

            az.plot_trace(trace, var_names=[
                "sd_uc",
                "sd_brain",
                "sd_body",
                "sd_obs_brain",
                "sd_obs_brain"
            ])

    def predict(self, evidence: BrainBodyDataset, num_samples: int, burn_in: int, num_chains: int, seed: Optional[int],
                num_jobs: int = 1) -> List[BrainBodyParticlesSummary]:
        summaries = []

        with self._model:
            for i in range(evidence.num_trials):
                self._define_model_variables(1, evidence.num_time_steps)

                pm.set_data({
                    "unbounded_coordination_value_0": evidence.series[i].unbounded_coordination,
                    "coordination_value_0": evidence.series[i].coordination,
                    "latent_brain_value_0": evidence.series[i].latent_brain_signals,
                    "latent_body_value_0": evidence.series[i].latent_body_movements,
                    "observed_brain_value_0": evidence.series[i].observed_brain_signals,
                    "observed_body_value_0": evidence.series[i].observed_body_movements
                })

                theano.config.floatX = 'float64'
                trace = pm.sample(num_samples, init="adapt_diag", return_inferencedata=False, tune=burn_in,
                                  chains=num_chains, random_seed=seed)

                summary = BrainBodyParticlesSummary()
                summary.unbounded_coordination_mean = trace["unbounded_coordination_0"].mean(axis=0)
                summary.coordination_mean = trace["coordination_0"].mean(axis=0)
                summary.latent_brain_mean = trace["latent_brain_0"].mean(axis=0)
                summary.latent_body_mean = trace["latent_body_0"].mean(axis=0)

                summary.unbounded_coordination_std = trace["unbounded_coordination_0"].std(axis=0)
                summary.coordination_std = trace["coordination_0"].std(axis=0)
                summary.latent_brain_std = trace["latent_brain_0"].std(axis=0)
                summary.latent_body_std = trace["latent_body_0"].std(axis=0)

                summaries.append(summary)

        return summaries

    @property
    def sd_uc(self):
        return self._sd_uc_value.get_value()

    @sd_uc.setter
    def sd_uc(self, value: float):
        with self._model:
            pm.set_data({"sd_uc_value": value})

    @property
    def sd_brain(self):
        return self._sd_brain_value.get_value()

    @sd_brain.setter
    def sd_brain(self, value: float):
        with self._model:
            pm.set_data({"sd_brain_value": value})

    @property
    def sd_body(self):
        return self._sd_body_value.get_value()

    @sd_body.setter
    def sd_body(self, value: float):
        with self._model:
            pm.set_data({"sd_body_value": value})

    @property
    def sd_obs_brain(self):
        return self._sd_obs_brain_value.get_value()

    @sd_obs_brain.setter
    def sd_obs_brain(self, value: float):
        with self._model:
            pm.set_data({"sd_obs_brain_value": value})

    @property
    def sd_obs_body(self):
        return self._sd_obs_body_value.get_value()

    @sd_obs_body.setter
    def sd_obs_body(self, value: float):
        with self._model:
            pm.set_data({"sd_obs_body_value": value})
