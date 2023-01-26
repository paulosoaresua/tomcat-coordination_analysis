import arviz as az
import matplotlib.pyplot as plt

from copy import deepcopy

from coordination.model.brain_body_model2 import BrainBodyModel, BrainBodySeries, BrainBodyInferenceSummary
from coordination.model.components.mixture_component import mixture_logp_with_self_dependency

import numpy as np
import pymc as pm

# Parameters
TIME_STEPS = 50
NUM_BRAIN_CHANNELS = 10
SEED = 0

if __name__ == "__main__":
    model = BrainBodyModel(initial_coordination=0.5, num_subjects=3, num_brain_channels=NUM_BRAIN_CHANNELS,
                           self_dependent=True)

    model.coordination_cpn.parameters.sd_uc = np.array([0.1])
    model.coordination_cpn.parameters.sd_c = np.array([0.01])

    # model.latent_brain_cpn.parameters.mixture_weights = np.full(shape=(3, 2), fill_value=0.5)
    # model.latent_brain_cpn.parameters.mean_a0 = np.zeros(shape=(3, NUM_BRAIN_CHANNELS))
    # model.latent_brain_cpn.parameters.sd_aa = np.full(shape=(3, NUM_BRAIN_CHANNELS), fill_value=0.0001)
    # model.latent_body_cpn.parameters.mixture_weights = np.full(shape=(3, 2), fill_value=0.5)
    # model.latent_body_cpn.parameters.mean_a0 = np.zeros(shape=(3, 1))
    # model.latent_body_cpn.parameters.sd_aa = np.full(shape=(3, 1), fill_value=0.0001)
    #
    # model.obs_brain_cpn.parameters.sd_o = np.full(shape=(3, NUM_BRAIN_CHANNELS), fill_value=0.0001)
    # model.obs_body_cpn.parameters.sd_o = np.full(shape=(3, 1), fill_value=0.0001)

    model.latent_brain_cpn.parameters.mixture_weights = np.array([[0.7, 0.3]])
    model.latent_brain_cpn.parameters.mean_a0 = np.array([0])
    model.latent_brain_cpn.parameters.sd_aa = np.array([0.01])
    model.latent_body_cpn.parameters.mixture_weights = np.array([[0.2, 0.8]])
    model.latent_body_cpn.parameters.mean_a0 = np.array([0])
    model.latent_body_cpn.parameters.sd_aa = np.array([0.01])

    model.obs_brain_cpn.parameters.sd_o = np.array([0.001])
    model.obs_body_cpn.parameters.sd_o = np.array([0.001])

    full_samples = model.draw_samples(num_series=1,
                                      num_time_steps=TIME_STEPS,
                                      seed=SEED,
                                      brain_relative_frequency=1,
                                      body_relative_frequency=1)

    evidence = BrainBodySeries(full_samples.obs_brain.values[0],
                               full_samples.obs_brain.mask[0],
                               full_samples.latent_brain.prev_time[0],
                               full_samples.obs_body.values[0],
                               full_samples.obs_body.mask[0],
                               full_samples.latent_body.prev_time[0])

    # 202300007735425

    import pytensor.tensor as pt

    logp = mixture_logp_with_self_dependency(
        mixture_component=pt.constant(full_samples.latent_brain.values[0]),
        initial_mean=pt.constant(model.latent_brain_cpn.parameters.mean_a0),
        sigma=pt.constant(model.latent_brain_cpn.parameters.sd_aa),
        mixture_weights=pt.constant(model.latent_brain_cpn.parameters.mixture_weights),
        coordination=pt.constant(full_samples.coordination.coordination[0]),
        prev_time=pt.constant(full_samples.latent_brain.prev_time[0], dtype=int),
        prev_time_mask=pt.constant(evidence.brain_prev_time_mask),
        subject_mask=pt.constant(full_samples.latent_brain.mask[0])
    )

    print(logp.eval())

    # model.coordination_cpn.parameters.reset()
    # model.latent_brain_cpn.parameters.sd_aa = None
    # model.latent_body_cpn.parameters.sd_aa = None
    model.obs_brain_cpn.parameters.sd_o = None
    model.obs_body_cpn.parameters.sd_o = None
    # model.obs_body_cpn.parameters.sd_o = np.array([0.01])
    pymc_model, idata = model.fit(evidence=evidence,
                         burn_in=1,
                         num_samples=1,
                         num_chains=2,
                         seed=SEED,
                         num_jobs=4)
    # az.plot_trace(idata, var_names=["sd_uc", "sd_aa_latent_brain", "sd_aa_latent_body", "sd_o_obs_brain", "sd_o_obs_body"])
    az.plot_trace(idata, var_names=["sd_o_obs_brain", "sd_o_obs_body"])
    plt.show()

    with pymc_model:
        samples = pm.sample_posterior_predictive(idata, var_names=["obs_brain"])



    inference_summary = BrainBodyInferenceSummary.from_inference_data(idata)

    m = inference_summary.coordination_means
    std = inference_summary.coordination_sds

    plt.figure(figsize=(15, 8))
    plt.plot(range(TIME_STEPS), full_samples.coordination.coordination[0], label="Real", color="tab:blue", marker="o")
    plt.plot(range(TIME_STEPS), m, label="Inferred", color="tab:orange", marker="o")
    plt.fill_between(range(TIME_STEPS), m - std, m + std, color="tab:orange", alpha=0.4)
    plt.title("Coordination")
    plt.legend()
    plt.show()

    print(f"Real Coordination Average: {full_samples.coordination.coordination[0].mean()}")
    print(f"Estimated Coordination Average: {m.mean()}")
    #
    # m = inference_summary.unbounded_coordination_mean
    # std = inference_summary.unbounded_coordination_std
    #
    # plt.figure(figsize=(15, 8))
    # plt.plot(range(TIME_STEPS), full_samples.unbounded_coordination[0], label="Real", color="tab:blue", marker="o")
    # plt.plot(range(TIME_STEPS), m, label="Inferred", color="tab:orange", marker="o")
    # plt.fill_between(range(TIME_STEPS), m - std, m + std, color="tab:orange", alpha=0.4)
    # plt.title("Unbounded Coordination")
    # plt.legend()
    # plt.show()
    #
    # m = inference_summary.coordination_mean
    # std = inference_summary.coordination_std
    #
    # plt.figure(figsize=(15, 8))
    # plt.plot(range(TIME_STEPS), full_samples.coordination[0], label="Real", color="tab:blue", marker="o")
    # plt.plot(range(TIME_STEPS), m, label="Inferred", color="tab:orange", marker="o")
    # plt.fill_between(range(TIME_STEPS), m - std, m + std, color="tab:orange", alpha=0.4)
    # plt.title("Coordination")
    # plt.legend()
    # plt.show()
    #

    time_steps = [t for t, m in enumerate(evidence.brain_mask) if m == 1]

    m = inference_summary.latent_brain_means[0, 0][time_steps]
    std = inference_summary.latent_brain_sds[0, 0][time_steps]

    plt.figure(figsize=(15, 8))
    plt.plot(time_steps, full_samples.latent_brain.values[0, 0, 0][time_steps], label="Real", color="tab:blue",
             marker="o")
    plt.plot(time_steps, m, label="Inferred", color="tab:orange", marker="o")
    plt.fill_between(time_steps, m - std, m + std, color="tab:orange", alpha=0.4)
    plt.title("Brain")
    plt.legend()
    plt.show()

    time_steps = [t for t, m in enumerate(evidence.body_mask) if m == 1]

    m = inference_summary.latent_body_means[0, 0][time_steps]
    std = inference_summary.latent_body_sds[0, 0][time_steps]

    plt.figure(figsize=(15, 8))
    plt.plot(time_steps, full_samples.latent_body.values[0, 0, 0][time_steps], label="Real", color="tab:blue",
             marker="o")
    plt.plot(time_steps, m, label="Inferred", color="tab:orange", marker="o")
    plt.fill_between(time_steps, m - std, m + std, color="tab:orange", alpha=0.4)
    plt.title("Body")
    plt.legend()
    plt.show()
