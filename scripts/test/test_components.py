import pymc as pm
import pytensor.tensor as ptt
import numpy as np
import matplotlib.pyplot as plt
import arviz as az

from coordination.model.components.coordination_component import SigmoidGaussianCoordinationComponent
from coordination.model.components.mixture_component import MixtureComponent, mixture_random_with_self_dependency
from coordination.model.components.observation_component import ObservationComponent

from coordination.common.functions import sigmoid

if __name__ == "__main__":
    T = 1024
    F = 2
    FREQ = 4

    coord = SigmoidGaussianCoordinationComponent(0.5, 1, 0.01)
    mix = MixtureComponent("mix", 3, F, False, sd_mean_a0=np.ones((3, F)), sd_sd_aa=np.full((3, F), fill_value=2),
                           a_mixture_weights=np.ones((3, 2)))
    obs = ObservationComponent("obs", 3, F, sd_sd_o=np.ones((3, F)))

    coord.parameters.sd_uc.value = np.full(1, fill_value=0.1)
    mix.parameters.mean_a0.value = np.zeros((3, F))
    mix.parameters.sd_aa.value = np.ones((3, F))
    mix.parameters.mixture_weights.value = np.array([[0.3, 0.7], [0.8, 0.2], [0.1, 0.9]])
    obs.parameters.sd_o.value = np.ones((3, F))

    np.random.seed(0)
    # coordination_values = np.random.rand(1, T)
    # coordination_values = np.zeros(shape=(1, T))
    coordination_values = coord.draw_samples(1, T, 0).coordination
    mix_samples = mix.draw_samples(num_series=1, seed=None, relative_frequency=FREQ, coordination=coordination_values)
    obs_samples = obs.draw_samples(seed=None,
                                   latent_component=mix_samples.values)

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
    plt.plot(np.arange(mix_samples.num_time_steps), mix_samples.values[0, 0, 0], marker="o", color="tab:blue",
             markersize=10)
    plt.plot(np.arange(mix_samples.num_time_steps), obs_samples.values[0, 0, 0], marker="o", color="tab:pink",
             markersize=5)
    plt.title(f"Samples Component")
    plt.show()

    plt.figure()
    plt.plot(np.arange(T), coordination_values[0], marker="o", color="tab:blue",
             markersize=10)
    plt.title(f"Samples Coordination")
    plt.show()

    mix.parameters.clear_values()
    obs.parameters.clear_values()
    coord.parameters.clear_values()
    with pm.Model(
            coords={"sub": np.arange(3), "coord_time": np.arange(T),
                    "cpn_time": np.arange(mix_samples.num_time_steps), "fea": np.arange(F)}) as model:
        _, coordination = coord.update_pymc_model("coord_time")
        latent_component = mix.update_pymc_model(
            # ptt.constant(coordination[mix_samples.time_steps_in_coordination_scale[0]]),
            coordination[mix_samples.time_steps_in_coordination_scale[0]],
            subject_dimension="sub",
            time_dimension="cpn_time",
            feature_dimension="fea")
        obs.update_pymc_model(latent_component, obs_samples.values[0])

        idata = pm.sample_prior_predictive(random_seed=0)

        prior_obs = idata.prior_predictive["obs"].sel(chain=0).to_numpy()

        plt.figure()
        plt.plot(np.arange(mix_samples.num_time_steps)[:, None].repeat(500, axis=1), prior_obs[:, 0, 0].T,
                 color="tab:blue", alpha=0.3)
        plt.plot(np.arange(mix_samples.num_time_steps), obs_samples.values[0, 0, 0], marker="o", color="black")
        plt.title(f"Prior Predictive Obs")
        plt.show()

        idata = pm.sample(draws=1000, tune=1000, chains=2, init="jitter+adapt_diag", random_seed=0)

        az.plot_trace(idata, var_names=["mixture_weights_mix", "mean_a0_mix", "sd_aa_mix", "sd_o_obs", "sd_uc"])
        plt.show()

        posterior = idata.posterior["mix"].sel(chain=0).to_numpy()

        plt.figure()
        plt.plot(np.arange(mix_samples.num_time_steps)[:, None].repeat(1000, axis=1), posterior[:, 0, 0].T,
                 color="tab:blue", alpha=0.3)
        plt.plot(np.arange(mix_samples.num_time_steps), mix_samples.values[0, 0, 0], marker="o", color="black",
                 alpha=0.7)
        plt.plot(np.arange(mix_samples.num_time_steps), posterior[:, 0, 0].mean(axis=0),
                 color="tab:pink", alpha=1)
        plt.title(f"Posterior Predictive")
        plt.show()

        posterior = sigmoid(idata.posterior["unbounded_coordination"].sel(chain=0).to_numpy())

        plt.figure()
        plt.plot(np.arange(T)[:, None].repeat(1000, axis=1), posterior.T,
                 color="tab:blue", alpha=0.3)
        plt.plot(np.arange(T), coordination_values[0], marker="o", color="black",
                 alpha=0.7)
        plt.plot(np.arange(T), posterior.mean(axis=0),
                 color="tab:pink", alpha=1)
        plt.title(f"Posterior Coordination")
        plt.show()
