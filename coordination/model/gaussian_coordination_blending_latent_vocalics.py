from typing import Any, Callable, List

import numpy as np
from scipy.stats import invgamma, norm

from coordination.model.coordination_blending_latent_vocalics import CoordinationBlendingLatentVocalics, \
    LatentVocalicsDataset, LatentVocalicsDataSeries, LatentVocalicsParticles
from coordination.model.coordination_blending_latent_vocalics import default_f, default_g


class GaussianCoordinationBlendingLatentVocalics(CoordinationBlendingLatentVocalics):

    def __init__(self,
                 initial_coordination: float,
                 num_vocalic_features: int,
                 num_speakers: int,
                 a_vcc: float,
                 b_vcc: float,
                 a_va: float,
                 b_va: float,
                 a_vaa: float,
                 b_vaa: float,
                 a_vo: float,
                 b_vo: float,
                 f: Callable = default_f,
                 g: Callable = default_g):
        super().__init__(initial_coordination, num_vocalic_features, num_speakers, a_vcc, b_vcc, a_va, b_va, a_vaa,
                         b_vaa, a_vo, b_vo, f, g)

    def _get_coordination_distribution(self, mean: np.ndarray, std: np.ndarray) -> Any:
        return norm(mean, std)

    # ---------------------------------------------------------
    # PARAMETER ESTIMATION
    # ---------------------------------------------------------
    def _get_initial_coordination_for_gibbs(self, evidence: LatentVocalicsDataset) -> np.ndarray:
        means = np.zeros((evidence.num_trials, evidence.num_time_steps))
        samples = norm(means, 0.1).rvs()
        samples[0] = self.initial_coordination

        return samples

    def _get_coordination_proposal(self, previous_coordination_sample: np.ndarray):

        # Since coordination is constrained to 0 and 1, we don't expect a high variance.
        # We set variance to be smaller than 0.01 such that MCMC don't do big jumps and ends up
        # overestimating coordination.
        std = 0.005
        new_coordination_sample = norm(previous_coordination_sample, std).rvs()

        if previous_coordination_sample.shape[0] == 1:
            # The norm.rvs function does not preserve the dimensions of a unidimensional array.
            # We need to correct that if we are working with a single trial sample.
            new_coordination_sample = np.array([[new_coordination_sample]])

        # Hastings factor
        # nominator = truncnorm(new_coordination_sample, std).logpdf(previous_coordination_sample)
        # denominator = truncnorm(previous_coordination_sample, std).logpdf(new_coordination_sample)
        # factor = np.exp(nominator - denominator).sum(axis=1)
        factor = 1

        return new_coordination_sample, factor

    def _update_latent_parameters_coordination(self, gibbs_step: int, evidence: LatentVocalicsDataset):
        # Variance of the State Transition
        if self.var_cc is None:
            a = self.a_vcc + evidence.num_trials * (evidence.num_time_steps - 1) / 2
            x = self.coordination_samples_[gibbs_step, :, 1:]
            y = self.coordination_samples_[gibbs_step, :, :evidence.num_time_steps - 1]
            b = self.b_vcc + np.square(x - y).sum() / 2
            self.vcc_samples_[gibbs_step] = invgamma(a=a, scale=b).mean()
            if self.vcc_samples_[gibbs_step] == np.nan:
                self.vcc_samples_[gibbs_step] = np.inf
        else:
            # Given
            self.vcc_samples_[gibbs_step] = self.var_cc

    # ---------------------------------------------------------
    # INFERENCE
    # ---------------------------------------------------------

    def _summarize_particles(self, series: LatentVocalicsDataSeries,
                             particles: List[LatentVocalicsParticles]) -> np.ndarray:
        # Mean and variance over time
        summary = np.zeros((2 + 2 * series.num_vocalic_features, len(particles)))

        for t, particles_in_time in enumerate(particles):
            if t == 0:
                summary[0, t] = self.initial_coordination
                summary[1, t] = 0
            else:
                summary[0, t] = particles_in_time.coordination.mean()
                summary[1, t] = particles_in_time.coordination.var()

            if series.observed_vocalics.mask[t] == 1:
                for k in range(series.num_vocalic_features):
                    speaker = series.observed_vocalics.utterances[t].subject_id
                    summary[2 + 2*k, t] = particles_in_time.latent_vocalics[speaker][:, k].mean()
                    summary[3 + 2*k, t] = particles_in_time.latent_vocalics[speaker][:, k].var()

        return summary
