from typing import Callable, List, Optional, Tuple

import arviz as az
import numpy as np
import pymc as pm

from coordination.component.coordination_component import SigmoidGaussianCoordinationComponent, \
    CoordinationComponentSamples
from coordination.component.mixture_component import MixtureComponentSamples
from coordination.component.mixture_mass_spring_damper_component import MixtureMassSpringDamperComponent
from coordination.component.observation_component import ObservationComponent, ObservationComponentSamples
from coordination.model.coordination_model import CoordinationPosteriorSamples


class MassSpringDamperSamples:

    def __init__(self, coordination: CoordinationComponentSamples, state: MixtureComponentSamples,
                 observation: ObservationComponentSamples):
        self.coordination = coordination
        self.state = state
        self.observation = observation


class MassSpringDamperModel:

    def __init__(self,
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
                 a_mixture_weights: np.ndarray,
                 sd_sd_o: np.ndarray,
                 share_mean_a0_across_springs: bool = False,
                 share_mean_a0_across_features: bool = False,
                 share_sd_aa_across_springs: bool = False,
                 share_sd_aa_across_features: bool = False,
                 share_sd_o_across_springs: bool = False,
                 share_sd_o_across_features: bool = False):
        self.num_springs = num_springs

        self.coordination_cpn = SigmoidGaussianCoordinationComponent(sd_mean_uc0=sd_mean_uc0,
                                                                     sd_sd_uc=sd_sd_uc)
        self.state_space_cpn = MixtureMassSpringDamperComponent(uuid="state_space",
                                                                num_springs=num_springs,
                                                                spring_constant=spring_constant,
                                                                mass=mass,
                                                                damping_coefficient=damping_coefficient,
                                                                dt=dt,
                                                                self_dependent=self_dependent,
                                                                mean_mean_a0=mean_mean_a0,
                                                                sd_mean_a0=sd_mean_a0,
                                                                sd_sd_aa=sd_sd_aa,
                                                                a_mixture_weights=a_mixture_weights,
                                                                share_mean_a0_across_springs=share_mean_a0_across_springs,
                                                                share_sd_aa_across_springs=share_sd_aa_across_springs,
                                                                share_mean_a0_across_features=share_mean_a0_across_features,
                                                                share_sd_aa_across_features=share_sd_aa_across_features)
        self.observation_cpn = ObservationComponent(uuid="observation",
                                                    num_subjects=num_springs,
                                                    dim_value=self.state_space_cpn.dim_value,
                                                    sd_sd_o=sd_sd_o,
                                                    share_sd_o_across_subjects=share_sd_o_across_springs,
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

    def draw_samples(self, num_series: int, num_time_steps: int, coordination_samples: Optional[np.ndarray],
                     seed: Optional[int] = None) -> MassSpringDamperSamples:
        if coordination_samples is None:
            coordination_samples = self.coordination_cpn.draw_samples(num_series, num_time_steps, seed).coordination
            seed = None

        state_samples = self.state_space_cpn.draw_samples(num_series,
                                                          relative_frequency=1,  # Same scale as coordination
                                                          coordination=coordination_samples,
                                                          seed=seed)
        observation_samples = self.observation_cpn.draw_samples(latent_component=state_samples.values)

        samples = MassSpringDamperSamples(coordination=coordination_samples,
                                          state=state_samples,
                                          observation=observation_samples)

        return samples

    def fit(self, evidence: np.ndarray, burn_in: int, num_samples: int, num_chains: int,
            seed: Optional[int] = None, num_jobs: int = 1, init_method: str = "jitter+adapt_diag") -> Tuple[
        pm.Model, az.InferenceData]:
        assert evidence.shape[0] == self.state_space_cpn.num_subjects
        assert evidence.shape[1] == self.state_space_cpn.dim_value

        pymc_model = self._define_pymc_model(evidence)
        with pymc_model:
            idata = pm.sample(num_samples, init=init_method, tune=burn_in, chains=num_chains, random_seed=seed,
                              cores=num_jobs)

        return pymc_model, idata

    def _define_pymc_model(self, evidence: np.ndarray):
        coords = {"feature": ["position", "velocity"],
                  "coordination_time": np.arange(evidence.shape[-1]),
                  "spring": np.arange(self.num_springs),
                  "spring_time": np.arange(evidence.shape[-1])}

        pymc_model = pm.Model(coords=coords)
        with pymc_model:
            coordination_dist = self.coordination_cpn.update_pymc_model(time_dimension="coordination_time")[1]
            state_space_dist = self.state_space_cpn.update_pymc_model(coordination=coordination_dist,
                                                                      subject_dimension="spring",
                                                                      feature_dimension="feature",
                                                                      time_dimension="spring_time",
                                                                      num_time_steps=evidence.shape[-1])[0]
            self.observation_cpn.update_pymc_model(latent_component=state_space_dist,
                                                   subject_dimension="spring",
                                                   feature_dimension="feature",
                                                   time_dimension="spring_time",
                                                   observed_values=evidence)

        return pymc_model


if __name__ == "__main__":
    model = MassSpringDamperModel(num_springs=3,
                                  spring_constant=np.array([16, 8, 4]),
                                  mass=np.ones(3) * 10,
                                  damping_coefficient=np.ones(3) * 0,
                                  dt=0.5,
                                  self_dependent=True,
                                  sd_mean_uc0=1,
                                  sd_sd_uc=1,
                                  mean_mean_a0=np.zeros((3, 2)),
                                  sd_mean_a0=np.ones((3, 2)) * 20,
                                  sd_sd_aa=np.ones(1),
                                  a_mixture_weights=np.ones((3, 2)),
                                  sd_sd_o=np.ones(1),
                                  share_sd_aa_across_springs=True,
                                  share_sd_aa_across_features=True,
                                  share_sd_o_across_springs=True,
                                  share_sd_o_across_features=True)

    model.state_space_cpn.parameters.mean_a0.value = np.array([[1, 0], [3, 0], [5, 0]])
    model.state_space_cpn.parameters.sd_aa.value = np.ones(1) * 0.1
    model.state_space_cpn.parameters.mixture_weights.value = np.array([[1, 0], [1, 0], [0, 1]])
    model.observation_cpn.parameters.sd_o.value = np.ones(1) * 0.5

    C = 0
    T = 100
    samples = model.draw_samples(num_series=1, num_time_steps=T, coordination_samples=np.ones((1, T)) * C, seed=0)

    import matplotlib.pyplot as plt

    plt.figure()
    for s in range(model.num_springs):
        plt.scatter(samples.state.time_steps_in_coordination_scale, samples.observation.values[0, s, 0],
                    label=f"Spring {s + 1}", s=15)
        plt.title(f"Coordination = {C}")
    plt.xlabel("Time Step")
    plt.ylabel("Position")
    plt.legend()
    plt.show()

    # Inference
    model.clear_parameter_values()
    _, idata = model.fit(
        evidence=samples.observation.values[0],
        burn_in=200,
        num_samples=200,
        num_chains=2,
        seed=0,
        num_jobs=2,
        init_method="jitter+adapt_diag"
    )

    sampled_vars = set(idata.posterior.data_vars)
    var_names = sorted(list(set(model.parameter_names).intersection(sampled_vars)))
    if len(var_names) > 0:
        az.plot_trace(idata, var_names=var_names)
        plt.tight_layout()
        plt.show()

    plt.figure()
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for s in range(model.num_springs):
        mean = idata.posterior["state_space"].sel(spring=s, feature="position").mean(dim=["draw", "chain"]).to_numpy()
        plt.scatter(samples.state.time_steps_in_coordination_scale, mean, label=f"Spring {s + 1}", s=15, c=colors[s])
    plt.xlabel("Time Step")
    plt.ylabel("Position")
    plt.legend()
    plt.show()

    coordination_posterior = CoordinationPosteriorSamples.from_inference_data(idata)
    coordination_posterior.plot(plt.figure().gca(), show_samples=False)
    plt.show()
