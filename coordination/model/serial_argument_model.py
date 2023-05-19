from typing import List, Optional, Tuple

import arviz as az
import numpy as np
import pymc as pm

from coordination.component.coordination_component import SigmoidGaussianCoordinationComponent, \
    CoordinationComponentSamples
from coordination.component.serial_component import SerialComponentSamples
from coordination.component.serial_mass_spring_damper_component import SerialMassSpringDamperComponent
from coordination.component.serial_observation_component import SerialObservationComponent, \
    SerialObservationComponentSamples
from coordination.model.coordination_model import CoordinationPosteriorSamples


class SerialArgumentSeries:
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


class SerialArgumentSamples:

    def __init__(self, coordination: CoordinationComponentSamples, state: SerialComponentSamples,
                 observation: SerialObservationComponentSamples):
        self.coordination = coordination
        self.state = state
        self.observation = observation


class SerialArgumentModel:
    """
    This class represents the Spring model.
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

        self.coordination_cpn = SigmoidGaussianCoordinationComponent(sd_mean_uc0=sd_mean_uc0,
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
        self.observation_cpn = SerialObservationComponent(uuid="observation",
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
                     can_repeat_subject: bool = False) -> SerialArgumentSamples:
        if coordination_samples is None:
            coordination_samples = self.coordination_cpn.draw_samples(num_series, num_time_steps, seed).coordination
            seed = None

        state_samples = self.state_space_cpn.draw_samples(num_series,
                                                          time_scale_density=1,  # Same scale as coordination
                                                          can_repeat_subject=can_repeat_subject,
                                                          coordination=coordination_samples,
                                                          seed=seed)
        observation_samples = self.observation_cpn.draw_samples(latent_component=state_samples.values,
                                                                subjects=state_samples.subjects)

        samples = SerialArgumentSamples(coordination=coordination_samples,
                                        state=state_samples,
                                        observation=observation_samples)

        return samples

    def fit(self,
            evidence: SerialArgumentSeries,
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

    def _define_pymc_model(self, evidence: SerialArgumentSeries):
        coords = {"feature": ["position", "velocity"],
                  "coordination_time": np.arange(evidence.observation.shape[-1]),
                  "subject_time": np.arange(evidence.observation.shape[-1])}

        pymc_model = pm.Model(coords=coords)
        with pymc_model:
            coordination_dist = self.coordination_cpn.update_pymc_model(time_dimension="coordination_time")[1]
            state_space_dist = self.state_space_cpn.update_pymc_model(coordination=coordination_dist,
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


if __name__ == "__main__":
    model = SerialArgumentModel(num_subjects=2,
                                frequency=np.array([0.8, 0.2]),
                                damping_coefficient=np.array([0, 0]),
                                dt=0.2,
                                self_dependent=True,
                                sd_mean_uc0=1,
                                sd_sd_uc=1,
                                mean_mean_a0=np.zeros((2, 2)),
                                sd_mean_a0=np.ones((2, 2)),
                                sd_sd_aa=np.ones(1),
                                sd_sd_o=np.ones(1),
                                share_sd_aa_across_subjects=True,
                                share_sd_aa_across_features=True,
                                share_sd_o_across_subjects=True,
                                share_sd_o_across_features=True)

    model.state_space_cpn.parameters.mean_a0.value = np.array([[1, 0], [1, 0]])
    model.state_space_cpn.parameters.sd_aa.value = np.ones(1) * 0.1
    model.observation_cpn.parameters.sd_o.value = np.ones(1) * 0.01

    C = 0.5
    T = 100
    PRX = "sd_sd_aa = 1, sd_aa = 0.1, sd_o = 0.5"
    samples = model.draw_samples(num_series=1, num_time_steps=T, coordination_samples=np.ones((1, T)) * C, seed=0)

    import matplotlib.pyplot as plt

    plt.figure()
    for s in range(model.num_subjects):
        tt = [t for t, subject in enumerate(samples.state.subjects[0]) if s == subject]
        plt.scatter(tt, samples.observation.values[0][0, tt],
                    label=f"Subject {s + 1}", s=15)
        plt.title(f"Coordination = {C}")
    plt.xlabel("Time Step")
    plt.ylabel("CLO Value")
    plt.legend()
    plt.show()

    evidence = SerialArgumentSeries(subjects_in_time=samples.state.subjects[0],
                                    prev_time_same_subject=samples.state.prev_time_same_subject[0],
                                    prev_time_diff_subject=samples.state.prev_time_diff_subject[0],
                                    observation=samples.observation.values[0])

    # Inference
    model.clear_parameter_values()
    _, idata = model.fit(
        evidence=evidence,
        burn_in=1000,
        num_samples=1000,
        num_chains=2,
        seed=0,
        num_jobs=2,
        init_method="jitter+adapt_diag",
        target_accept=0.9
    )

    sampled_vars = set(idata.posterior.data_vars)
    var_names = sorted(list(set(model.parameter_names).intersection(sampled_vars)))
    if len(var_names) > 0:
        az.plot_trace(idata, var_names=var_names)
        plt.tight_layout()
        plt.show()

    plt.figure()
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for s in range(model.num_subjects):
        tt = [t for t, subject in enumerate(samples.state.subjects[0]) if s == subject]
        mean = idata.posterior["state_space"].sel(feature="position").mean(dim=["draw", "chain"]).to_numpy()[tt]
        plt.scatter(tt, mean, label=f"Subject {s + 1}", s=15, c=colors[s])
    plt.xlabel("Time Step")
    plt.ylabel("CLO Value")
    plt.title(PRX)
    plt.legend()
    plt.show()

    coordination_posterior = CoordinationPosteriorSamples.from_inference_data(idata)
    coordination_posterior.plot(plt.figure().gca(), show_samples=False)
    plt.title(PRX)
    plt.show()
