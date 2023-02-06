import pymc as pm
import pytensor.tensor as ptt
import numpy as np
import matplotlib.pyplot as plt
import arviz as az

from coordination.model.components.mixture_component import MixtureComponent, mixture_random_with_self_dependency
from coordination.model.components.observation_component import ObservationComponent

if __name__ == "__main__":
    T = 200
    F = 2

    mix = MixtureComponent("mix", 3, F, True)
    obs = ObservationComponent("obs")

    mix.parameters.mean_a0 = np.zeros((3, F))
    mix.parameters.sd_aa = np.ones((3, F))
    mix.parameters.mixture_weights = np.array([[0.3, 0.7], [0.8, 0.2], [0.1, 0.9]])
    obs.parameters.sd_o = np.ones((3, F))

    coordination_values = np.ones(shape=(1, T))
    mix_samples = mix.draw_samples(num_series=1, num_time_steps=T, seed=0, relative_frequency=1,
                                   coordination=coordination_values)
    obs_samples = obs.draw_samples(seed=None,
                                   latent_component=mix_samples.values,
                                   latent_mask=mix_samples.mask)

    # sample = mixture_random_with_self_dependency(
    #     initial_mean=mix.parameters.mean_a0,
    #     sigma=mix.parameters.sd_aa,
    #     mixture_weights=mix.parameters.mixture_weights,
    #     coordination=coordination_values[0],
    #     prev_time=mix_samples.prev_time[0],
    #     prev_time_mask=np.where(mix_samples.prev_time[0] >= 0, 1, 0),
    #     subject_mask=mix_samples.mask[0],
    #     expander_aux_mask_matrix=np.array([]),
    #     aggregation_aux_mask_matrix=np.array([]),
    #     num_subjects=3,
    #     dim_value=F,
    #     rng=np.random.default_rng(),
    #     size=(3, F, T)
    # )
    # print(sample)

    plt.figure()
    plt.plot(np.arange(T), mix_samples.values[0, 0, 0], marker="o", color="tab:blue", markersize=10)
    plt.plot(np.arange(T), obs_samples.values[0, 0, 0], marker="o", color="tab:pink", markersize=5)
    plt.title(f"Samples")
    plt.show()

    mix.parameters.mixture_weights = None
    mix.parameters.mean_a0 = None
    mix.parameters.sd_aa = None
    obs.parameters.sd_o = None
    prev_time_mask = np.where(mix_samples.prev_time >= 0, 1, 0)
    with pm.Model(coords={"sub": np.arange(3), "time": np.arange(T), "fea": np.arange(F)}) as model:
        latent_component = mix.update_pymc_model(ptt.constant(coordination_values[0]),
                                                 ptt.constant(mix_samples.prev_time[0]),
                                                 ptt.constant(np.where(mix_samples.prev_time[0] >= 0, 1, 0)),
                                                 ptt.constant(mix_samples.mask[0]),
                                                 subject_dimension="sub",
                                                 time_dimension="time",
                                                 feature_dimension="fea", observation=None)  # mix_samples.values[0])
        # obs.update_pymc_model(ptt.constant(mix_samples.values[0]), [3, 2], obs_samples.values[0])
        obs.update_pymc_model(latent_component, [3, F], obs_samples.values[0])

        idata = pm.sample_prior_predictive(random_seed=0)

        prior_obs = idata.prior_predictive["obs"].sel(chain=0).to_numpy()

        plt.figure()
        plt.plot(np.arange(T)[:, None].repeat(500, axis=1), prior_obs[:, 0, 0].T, color="tab:blue", alpha=0.3)
        plt.plot(np.arange(T), obs_samples.values[0, 0, 0], marker="o", color="black")
        plt.title(f"Prior Predictive Obs")
        plt.show()

        idata = pm.sample(draws=1000, tune=1000, chains=2, init="jitter+adapt_diag", random_seed=0)

        az.plot_trace(idata, var_names=["mixture_weights_mix", "mean_a0_mix", "sd_aa_mix", "sd_o_obs"])
        plt.show()

        posterior = idata.posterior["mix"].sel(chain=0).to_numpy()

        plt.figure()
        plt.plot(np.arange(T)[:, None].repeat(1000, axis=1), posterior[:, 0, 0].T, color="tab:blue", alpha=0.3)
        plt.plot(np.arange(T), mix_samples.values[0, 0, 0], marker="o", color="black", alpha=0.7)
        plt.title(f"Posterior Predictive")
        plt.show()
