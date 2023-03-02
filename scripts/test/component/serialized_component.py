import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import pytensor.tensor as ptt

from coordination.model.components.coordination_component import SigmoidGaussianCoordinationComponent
from coordination.model.components.serialized_component import SerializedComponent
from coordination.model.components.observation_component import SerializedObservationComponent

from coordination.common.functions import sigmoid

if __name__ == "__main__":
    T = 1000
    F = 2
    S = 2
    OBS_DENSITY = 0.2

    # Components
    coordination_cpn = SigmoidGaussianCoordinationComponent(initial_coordination=0.5, sd_uc=1)
    serial_cpn = SerializedComponent("serial", S, F, False, sd_mean_a0=np.ones((S, F)),
                                     sd_sd_aa=np.full((S, F), fill_value=2))
    observation_cpn = SerializedObservationComponent("obs", S, F, sd_sd_o=np.ones((S, F)))

    # Parameters
    coordination_cpn.parameters.sd_uc.value = np.full(1, fill_value=0.1)
    serial_cpn.parameters.mean_a0.value = np.zeros((S, F))
    serial_cpn.parameters.sd_aa.value = np.ones((S, F))
    observation_cpn.parameters.sd_o.value = np.ones((S, F))

    # Samples
    coordination_values = coordination_cpn.draw_samples(num_series=1, num_time_steps=T, seed=0).coordination
    serial_samples = serial_cpn.draw_samples(num_series=1, seed=None, time_scale_density=OBS_DENSITY,
                                             coordination=coordination_values, can_repeat_subject=False)
    obs_samples = observation_cpn.draw_samples(seed=None, latent_component=serial_samples.values,
                                               subjects=serial_samples.subjects)

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
    plt.plot(np.arange(serial_samples.num_time_steps), serial_samples.values[0][0], marker="o", color="tab:blue",
             markersize=10)
    plt.plot(np.arange(serial_samples.num_time_steps), obs_samples.values[0][0], marker="o", color="tab:pink",
             markersize=5)
    plt.title(f"Samples - 1st Feature from 1st Individual")
    plt.show()

    # Plot coordination
    plt.figure()
    plt.plot(np.arange(T), coordination_values[0], marker="o", color="tab:blue", markersize=5)
    plt.title(f"Samples Coordination")
    plt.show()

    # Joint inference
    serial_cpn.parameters.clear_values()
    observation_cpn.parameters.clear_values()
    coordination_cpn.parameters.clear_values()
    with pm.Model(
            coords={"sub": np.arange(S), "coord_time": np.arange(T),
                    "cpn_time": np.arange(serial_samples.num_time_steps), "fea": np.arange(F)}) as model:
        _, coordination = coordination_cpn.update_pymc_model("coord_time")
        latent_component = serial_cpn.update_pymc_model(
            coordination=coordination[serial_samples.time_steps_in_coordination_scale[0]],
            prev_time_same_subject=ptt.constant(serial_samples.prev_time_same_subject[0]),
            prev_time_diff_subject=ptt.constant(serial_samples.prev_time_diff_subject[0]),
            prev_same_subject_mask=ptt.constant(serial_samples.prev_time_same_subject_mask[0]),
            prev_diff_subject_mask=ptt.constant(serial_samples.prev_time_diff_subject_mask[0]),
            subjects=ptt.constant(serial_samples.subjects[0]),
            time_dimension="cpn_time",
            feature_dimension="fea")
        observation_cpn.update_pymc_model(latent_component, ptt.constant(serial_samples.subjects[0]),
                                          obs_samples.values[0])

        # Predictive prior
        idata = pm.sample_prior_predictive(random_seed=0)

        prior_obs = idata.prior_predictive["obs"].sel(chain=0).to_numpy()

        plt.figure()
        plt.plot(np.arange(serial_samples.num_time_steps)[:, None].repeat(500, axis=1), prior_obs[:, 0].T,
                 color="tab:blue", alpha=0.3)
        plt.plot(np.arange(serial_samples.num_time_steps), obs_samples.values[0][0], marker="o", color="black",
                 markersize=5)
        plt.title(f"Prior Predictive Obs")
        plt.show()

        # Posterior
        idata = pm.sample(draws=1000, tune=1000, chains=2, init="jitter+adapt_diag", random_seed=0)

        #   Parameters
        # az.plot_trace(idata, var_names=["mean_a0_serial", "sd_aa_serial", "sd_o_obs", "sd_uc"])
        # plt.show()

        #   Latent feature
        # posterior = idata.posterior["serial"].sel(chain=0).to_numpy()
        #
        # plt.figure()
        # plt.plot(np.arange(serial_samples.num_time_steps)[:, None].repeat(1000, axis=1), posterior[:, 0].T,
        #          color="tab:blue", alpha=0.3)
        # plt.plot(np.arange(serial_samples.num_time_steps), serial_samples.values[0][0], marker="o", color="black",
        #          alpha=0.7, markersize=5)
        # plt.plot(np.arange(serial_samples.num_time_steps), posterior[:, 0].mean(axis=0), color="tab:pink", alpha=1)
        # plt.title(f"Posterior - 1st Feature from 1st Individual")
        # plt.show()

        #   Coordination
        posterior = sigmoid(idata.posterior["unbounded_coordination"].sel(chain=0).to_numpy())

        plt.figure()
        plt.plot(np.arange(T)[:, None].repeat(1000, axis=1), posterior.T, color="tab:blue", alpha=0.3)
        plt.plot(np.arange(T), coordination_values[0], marker="o", color="black", alpha=0.7, markersize=5)
        plt.plot(np.arange(T), posterior.mean(axis=0), color="tab:pink", alpha=1)
        plt.title(f"Posterior Coordination")
        plt.show()
