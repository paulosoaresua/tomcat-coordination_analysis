from typing import List, Optional, Tuple

import arviz as az
import numpy as np
import pymc as pm

from coordination.module.coordination_old import (CoordinationSamples,
                                              SigmoidGaussianCoordination)
from coordination.module.non_serial_component import NonSerialComponentSamples
from coordination.module.non_serial_mass_spring_damper_component import \
    NonSerialMassSpringDamperComponent
from coordination.module.non_serial_observation import (
    NonSerialObservation, NonSerialObservationSamples)


class SpringSamples:
    def __init__(
        self,
        coordination: CoordinationSamples,
        state: NonSerialComponentSamples,
        observation: NonSerialObservationSamples,
    ):
        self.coordination = coordination
        self.state = state
        self.observation = observation


class SpringModel:
    def __init__(
        self,
        num_springs: int,
        spring_constant: np.ndarray,
        mass: np.ndarray,
        damping_coefficient: np.ndarray,
        dt: float,
        self_dependent: bool,
        sd_mean_uc0: float,
        sd_sd_uc: float,
        mean_mean_a0: np.ndarray,
        sd_mean_a0: np.ndarray,
        sd_sd_aa: np.ndarray,
        sd_sd_o: np.ndarray,
        share_mean_a0_across_springs: bool = False,
        share_mean_a0_across_features: bool = False,
        share_sd_aa_across_springs: bool = False,
        share_sd_aa_across_features: bool = False,
        share_sd_o_across_springs: bool = False,
        share_sd_o_across_features: bool = False,
    ):
        self.num_springs = num_springs

        self.coordination_cpn = SigmoidGaussianCoordination(
            sd_mean_uc0=sd_mean_uc0, sd_sd_uc=sd_sd_uc
        )
        self.state_space_cpn = NonSerialMassSpringDamperComponent(
            uuid="state_space",
            num_springs=num_springs,
            spring_constant=spring_constant,
            mass=mass,
            damping_coefficient=damping_coefficient,
            dt=dt,
            self_dependent=self_dependent,
            mean_mean_a0=mean_mean_a0,
            sd_mean_a0=sd_mean_a0,
            sd_sd_aa=sd_sd_aa,
            share_mean_a0_across_springs=share_mean_a0_across_springs,
            share_sd_aa_across_springs=share_sd_aa_across_springs,
            share_mean_a0_across_features=share_mean_a0_across_features,
            share_sd_aa_across_features=share_sd_aa_across_features,
        )
        self.observation_cpn = NonSerialObservation(
            uuid="observation",
            num_subjects=num_springs,
            dim_value=self.state_space_cpn.dim_value,
            sd_sd_o=sd_sd_o,
            share_sd_o_across_subjects=share_sd_o_across_springs,
            share_sd_o_across_features=share_sd_o_across_features,
        )

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

    def draw_samples(
        self,
        num_series: int,
        num_time_steps: int,
        coordination_samples: Optional[np.ndarray],
        seed: Optional[int] = None,
    ) -> SpringSamples:
        if coordination_samples is None:
            coordination_samples = self.coordination_cpn.draw_samples(
                num_series, num_time_steps, seed
            ).coordination
            seed = None

        state_samples = self.state_space_cpn.draw_samples(
            relative_frequency=1,  # Same scale as coordination
            coordination=coordination_samples,
            seed=seed,
        )
        observation_samples = self.observation_cpn.draw_samples(
            latent_component=state_samples.values
        )

        samples = SpringSamples(
            coordination=coordination_samples,
            state=state_samples,
            observation=observation_samples,
        )

        return samples

    def fit(
        self,
        evidence: np.ndarray,
        burn_in: int,
        num_samples: int,
        num_chains: int,
        seed: Optional[int] = None,
        num_jobs: int = 1,
        init_method: str = "jitter+adapt_diag",
        target_accept: float = 0.8,
    ) -> Tuple[pm.Model, az.InferenceData]:
        assert evidence.shape[0] == self.state_space_cpn.num_subjects
        assert evidence.shape[1] == self.state_space_cpn.dim_value

        pymc_model = self._define_pymc_model(evidence)
        with pymc_model:
            idata = pm.sample(
                num_samples,
                init=init_method,
                tune=burn_in,
                chains=num_chains,
                random_seed=seed,
                cores=num_jobs,
                target_accept=target_accept,
            )

        return pymc_model, idata

    def _define_pymc_model(self, evidence: np.ndarray):
        coords = {
            "feature": ["position", "velocity"],
            "coordination_time": np.arange(evidence.shape[-1]),
            "spring": np.arange(self.num_springs),
            "spring_time": np.arange(evidence.shape[-1]),
        }

        pymc_model = pm.Model(coords=coords)
        with pymc_model:
            coordination_dist = self.coordination_cpn.update_pymc_model(
                time_dimension="coordination_time"
            )[1]
            state_space_dist = self.state_space_cpn.update_pymc_model(
                coordination=coordination_dist,
                subject_dimension="spring",
                feature_dimension="feature",
                time_dimension="spring_time",
            )[0]
            self.observation_cpn.update_pymc_model(
                latent_component=state_space_dist,
                subject_dimension="spring",
                feature_dimension="feature",
                time_dimension="spring_time",
                observed_values=evidence,
            )

        return pymc_model
