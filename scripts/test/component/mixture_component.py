import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import arviz as az

from coordination.component.coordination_component import SigmoidGaussianCoordinationComponent
from coordination.component.mixture_component import MixtureComponent
from coordination.component.observation_component import ObservationComponent

from coordination.common.functions import sigmoid

if __name__ == "__main__":
    T = 100
    F = 2
    FREQ = 1

    # Components
    coordination_cpn = SigmoidGaussianCoordinationComponent(initial_coordination=0.5, sd_uc=1)
    mixture_cpn = MixtureComponent("mix", 3, F, True, sd_mean_a0=np.ones((3, F)),
                                   sd_sd_aa=np.full((3, F), fill_value=2),
                                   a_mixture_weights=np.ones((3, 2)))
    observation_cpn = ObservationComponent("obs", 3, F, sd_sd_o=np.ones((3, F)))

    # Parameters
    coordination_cpn.parameters.sd_uc.value = np.full(1, fill_value=0.1)
    mixture_cpn.parameters.mean_a0.value = np.zeros((3, F))
    mixture_cpn.parameters.sd_aa.value = np.ones((3, F))
    mixture_cpn.parameters.mixture_weights.value = np.array([[0.3, 0.7], [0.8, 0.2], [0.1, 0.9]])
    observation_cpn.parameters.sd_o.value = np.ones((3, F))

    # Samples
    coordination_values = np.ones((1, T))  # coordination_cpn.draw_samples(num_series=1, num_time_steps=T, seed=0).coordination
    mix_samples = mixture_cpn.draw_samples(num_series=1, seed=None, relative_frequency=FREQ,
                                           coordination=coordination_values)
    obs_samples = observation_cpn.draw_samples(seed=None, latent_component=mix_samples.values)

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

    # Plot feature
    plt.figure()
    plt.plot(np.arange(mix_samples.num_time_steps), mix_samples.values[0, 0, 0], marker="o", color="tab:blue",
             markersize=10)
    plt.plot(np.arange(mix_samples.num_time_steps), obs_samples.values[0, 0, 0], marker="o", color="tab:pink",
             markersize=5)
    plt.title(f"Samples - 1st Feature from 1st Individual")
    plt.show()

    # Plot coordination
    plt.figure()
    plt.plot(np.arange(T), coordination_values[0], marker="o", color="tab:blue", markersize=5)
    plt.title(f"Samples Coordination")
    plt.show()

    # Joint inference
    mixture_cpn.parameters.clear_values()
    observation_cpn.parameters.clear_values()
    coordination_cpn.parameters.clear_values()
    with pm.Model(
            coords={"sub": np.arange(3), "coord_time": np.arange(T),
                    "cpn_time": np.arange(mix_samples.num_time_steps), "fea": np.arange(F)}) as model:
        _, coordination = coordination_cpn.update_pymc_model("coord_time")
        latent_component = mixture_cpn.update_pymc_model(
            coordination[mix_samples.time_steps_in_coordination_scale[0]],
            subject_dimension="sub",
            time_dimension="cpn_time",
            feature_dimension="fea")
        observation_cpn.update_pymc_model(latent_component, obs_samples.values[0])

        # Predictive prior
        idata = pm.sample_prior_predictive(random_seed=0)

        prior_obs = idata.prior_predictive["obs"].sel(chain=0).to_numpy()

        plt.figure()
        plt.plot(np.arange(mix_samples.num_time_steps)[:, None].repeat(500, axis=1), prior_obs[:, 0, 0].T,
                 color="tab:blue", alpha=0.3)
        plt.plot(np.arange(mix_samples.num_time_steps), obs_samples.values[0, 0, 0], marker="o", color="black",
                 markersize=5)
        plt.title(f"Prior Predictive Obs")
        plt.show()

        # Posterior
        idata = pm.sample(draws=1000, tune=1000, chains=2, init="jitter+adapt_diag", random_seed=0)

        #   Parameters
        az.plot_trace(idata, var_names=["mixture_weights_mix", "mean_a0_mix", "sd_aa_mix", "sd_o_obs", "sd_uc"])
        plt.show()

        #   Latent feature
        posterior = idata.posterior["mix"].sel(chain=0).to_numpy()

        plt.figure()
        plt.plot(np.arange(mix_samples.num_time_steps)[:, None].repeat(1000, axis=1), posterior[:, 0, 0].T,
                 color="tab:blue", alpha=0.3)
        plt.plot(np.arange(mix_samples.num_time_steps), mix_samples.values[0, 0, 0], marker="o", color="black",
                 alpha=0.7, markersize=5)
        plt.plot(np.arange(mix_samples.num_time_steps), posterior[:, 0, 0].mean(axis=0), color="tab:pink", alpha=1)
        plt.title(f"Posterior - 1st Feature from 1st Individual")
        plt.show()

        #   Coordination
        posterior = sigmoid(idata.posterior["unbounded_coordination"].sel(chain=0).to_numpy())

        plt.figure()
        plt.plot(np.arange(T)[:, None].repeat(1000, axis=1), posterior.T, color="tab:blue", alpha=0.3)
        plt.plot(np.arange(T), coordination_values[0], marker="o", color="black", alpha=0.7, markersize=5)
        plt.plot(np.arange(T), posterior.mean(axis=0), color="tab:pink", alpha=1)
        plt.title(f"Posterior Coordination")
        plt.show()
