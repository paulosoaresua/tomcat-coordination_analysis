from typing import List, Optional, Tuple

import arviz as az
import numpy as np
import pymc as pm

from coordination.module.coordination import SigmoidGaussianCoordination, \
    CoordinationSamples
from coordination.module.serial_component import SerialComponentSamples
from coordination.module.serial_mass_spring_damper_component import SerialMassSpringDamperComponent
from coordination.module.serial_observation import SerialObservation, SerialObservationSamples


class ConversationSeries:
    """
    Used to encapsulate observations and meta-data.
    """

    def __init__(self,
                 subjects_in_time: np.ndarray,
                 prev_time_same_subject: np.ndarray,
                 prev_time_diff_subject: np.ndarray,
                 observation: np.ndarray):
        self.subjects_in_time = subjects_in_time
        self.observation = observation
        self.prev_time_same_subject = prev_time_same_subject
        self.prev_time_diff_subject = prev_time_diff_subject

    @property
    def prev_same_subject_mask(self) -> np.ndarray:
        return np.where(self.prev_time_same_subject >= 0, 1, 0)

    @property
    def prev_diff_subject_mask(self) -> np.ndarray:
        return np.where(self.prev_time_diff_subject >= 0, 1, 0)


class ConversationSamples:

    def __init__(self,
                 coordination: CoordinationSamples,
                 state: SerialComponentSamples,
                 observation: SerialObservationSamples):
        self.coordination = coordination
        self.state = state
        self.observation = observation


class ConversationModel:
    """
    This class represents the conversation model.
    """

    def __init__(self,
                 num_subjects: int,
                 frequency: np.ndarray,  # one per subject
                 damping_coefficient: np.ndarray,  # one per subject
                 dt: float,
                 self_dependent: bool,
                 sd_mean_uc0: float,
                 sd_sd_uc: float,
                 mean_mean_a0: np.ndarray,
                 sd_mean_a0: np.ndarray,
                 sd_sd_aa: np.ndarray,
                 sd_sd_o: np.ndarray,
                 share_mean_a0_across_subjects: bool = False,
                 share_mean_a0_across_features: bool = False,
                 share_sd_aa_across_subjects: bool = False,
                 share_sd_aa_across_features: bool = False,
                 share_sd_o_across_subjects: bool = False,
                 share_sd_o_across_features: bool = False):
        self.num_subjects = num_subjects

        self.coordination_cpn = SigmoidGaussianCoordination(sd_mean_uc0=sd_mean_uc0,
                                                            sd_sd_uc=sd_sd_uc)
        self.state_space_cpn = SerialMassSpringDamperComponent(uuid="state_space",
                                                               num_springs=num_subjects,
                                                               spring_constant=frequency,
                                                               mass=np.ones(num_subjects),
                                                               damping_coefficient=damping_coefficient,
                                                               dt=dt,
                                                               self_dependent=self_dependent,
                                                               mean_mean_a0=mean_mean_a0,
                                                               sd_mean_a0=sd_mean_a0,
                                                               sd_sd_aa=sd_sd_aa,
                                                               share_mean_a0_across_springs=share_mean_a0_across_subjects,
                                                               share_sd_aa_across_springs=share_sd_aa_across_subjects,
                                                               share_mean_a0_across_features=share_mean_a0_across_features,
                                                               share_sd_aa_across_features=share_sd_aa_across_features)
        self.observation_cpn = SerialObservation(uuid="observation",
                                                 num_subjects=num_subjects,
                                                 dim_value=self.state_space_cpn.dim_value,
                                                 sd_sd_o=sd_sd_o,
                                                 share_sd_o_across_subjects=share_sd_o_across_subjects,
                                                 share_sd_o_across_features=share_sd_o_across_features)

    @property
    def parameter_names(self) -> List[str]:
        names = self.coordination_cpn.parameter_names
        names.extend(self.state_space_cpn.parameter_names)
        names.extend(self.observation_cpn.parameter_names)

        return names

    def clear_parameter_values(self):
        self.coordination_cpn.parameters.clear_values()
        self.state_space_cpn.clear_parameter_values()
        self.observation_cpn.parameters.clear_values()

    def draw_samples(self,
                     num_series: int,
                     num_time_steps: int,
                     coordination_samples: Optional[np.ndarray],
                     seed: Optional[int] = None,
                     can_repeat_subject: bool = False,
                     fixed_subject_sequence: bool = False) -> ConversationSamples:
        if coordination_samples is None:
            coordination_samples = self.coordination_cpn.draw_samples(num_series, num_time_steps,
                                                                      seed).coordination
            seed = None

        state_samples = self.state_space_cpn.draw_samples(num_series,
                                                          time_scale_density=1,
                                                          # Same scale as coordination
                                                          can_repeat_subject=can_repeat_subject,
                                                          coordination=coordination_samples,
                                                          seed=seed,
                                                          fixed_subject_sequence=fixed_subject_sequence)
        observation_samples = self.observation_cpn.draw_samples(
            latent_component=state_samples.values,
            subjects=state_samples.subjects)

        samples = ConversationSamples(coordination=coordination_samples,
                                      state=state_samples,
                                      observation=observation_samples)

        return samples

    def fit(self,
            evidence: ConversationSeries,
            burn_in: int,
            num_samples: int,
            num_chains: int,
            seed: Optional[int] = None,
            num_jobs: int = 1,
            init_method: str = "jitter+adapt_diag",
            target_accept: float = 0.8) -> Tuple[pm.Model, az.InferenceData]:
        assert evidence.observation.shape[0] == self.state_space_cpn.dim_value

        pymc_model = self._define_pymc_model(evidence)
        with pymc_model:
            idata = pm.sample(num_samples,
                              init=init_method,
                              tune=burn_in,
                              chains=num_chains,
                              random_seed=seed,
                              cores=num_jobs,
                              target_accept=target_accept)

        return pymc_model, idata

    def _define_pymc_model(self, evidence: ConversationSeries):
        coords = {"feature": ["position", "velocity"],
                  "coordination_time": np.arange(evidence.observation.shape[-1]),
                  "subject_time": np.arange(evidence.observation.shape[-1])}

        pymc_model = pm.Model(coords=coords)
        with pymc_model:
            coordination_dist = \
                self.coordination_cpn.update_pymc_model(time_dimension="coordination_time")[1]
            state_space_dist = \
                self.state_space_cpn.update_pymc_model(coordination=coordination_dist,
                                                       feature_dimension="feature",
                                                       time_dimension="subject_time",
                                                       subjects=evidence.subjects_in_time,
                                                       prev_time_same_subject=evidence.prev_time_same_subject,
                                                       prev_time_diff_subject=evidence.prev_time_diff_subject,
                                                       prev_same_subject_mask=evidence.prev_same_subject_mask,
                                                       prev_diff_subject_mask=evidence.prev_diff_subject_mask)[
                    0]
            self.observation_cpn.update_pymc_model(latent_component=state_space_dist,
                                                   feature_dimension="feature",
                                                   time_dimension="subject_time",
                                                   subjects=evidence.subjects_in_time,
                                                   observed_values=evidence.observation)

        return pymc_model
