import pymc as pm
import pytensor.tensor as ptt
import numpy as np

from coordination.model.components.mixture_component import MixtureComponent
from coordination.model.components.observation_component import ObservationComponent

if __name__ == "__main__":
    mix = MixtureComponent("mix", 3, 2, True)
    obs = ObservationComponent("obs")

    mix.parameters.mean_a0 = np.array([0])
    mix.parameters.sd_aa = np.array([1])
    mix.parameters.mixture_weights = np.array([[0.7, 0.3]])
    obs.parameters.sd_o = np.array([1])

    coordination_values = np.ones(shape=(1, 100))
    mix_samples = mix.draw_samples(num_series=1, num_time_steps=100, seed=0, relative_frequency=1,
                                   coordination=coordination_values)
    obs_samples = obs.draw_samples(seed=0,
                                   latent_component=mix_samples.values,
                                   latent_mask=mix_samples.mask)

    mix.parameters.sd_aa = None
    # obs.parameters.sd_o = None
    prev_time_mask = np.where(mix_samples.prev_time >= 0, 1, 0)
    with pm.Model(coords={"sub": np.arange(3), "time": np.arange(100), "fea": np.arange(2)}) as model:
        latent_component = mix.update_pymc_model(ptt.constant(coordination_values[0]),
                                                 ptt.constant(mix_samples.prev_time[0]),
                                                 ptt.constant(np.where(mix_samples.prev_time[0] >= 0, 1, 0)),
                                                 ptt.constant(prev_time_mask[0]),
                                                 subject_dimension="sub",
                                                 time_dimension="time",
                                                 feature_dimension="fea",
                                                 observation=mix_samples.values[0])
        # obs.update_pymc_model(ptt.constant(mix_samples.values[0]), [3, 2], obs_samples.values[0])
        # obs.update_pymc_model(latent_component, [3, 2], obs_samples.values[0])
        # pm.sample()
        pm.sample_prior_predictive()
