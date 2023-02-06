import arviz as az
import matplotlib.pyplot as plt

from copy import deepcopy

from coordination.model.brain_body_model import BrainBodyModel, multi_influencers_mixture_logp
from coordination.model.utils.brain_body_model import BrainBodyDataset, BrainBodyParticlesSummary

import numpy as np
import pymc as pm

# Parameters
TIME_STEPS = 1200
NUM_SERIES = 1
NUM_BRAIN_CHANNELS = 20
SEED = 0

if __name__ == "__main__":
    model = BrainBodyModel(0.5, NUM_BRAIN_CHANNELS, 3)

    model.parameters.sd_uc = np.array([0.1])
    model.parameters.sd_c = np.array([0.1])
    model.parameters.sd_brain = np.array([0.1])
    model.parameters.sd_body = np.array([0.1])
    model.parameters.sd_obs_brain = np.array([0.1])
    model.parameters.sd_obs_body = np.array([0.1])

    full_samples = model.sample(NUM_SERIES, TIME_STEPS, SEED)

    partial_samples = deepcopy(full_samples)
    partial_samples.unbounded_coordination = None
    partial_samples.coordination = None
    partial_samples.latent_brain = None
    partial_samples.latent_body = None
    evidence = BrainBodyDataset.from_samples(partial_samples)

    import pytensor.tensor as at

    multi_influencers_mixture_logp(at.constant(evidence.series[0].observed_brain_signals),
                                   at.constant(full_samples.coordination[0]),
                                   at.constant(np.array([1])))

    # model.parameters.reset()
    idata = model.fit(evidence.series[0], 1000, 1000, 2, 0, 6)
    # az.plot_trace(idata, var_names=["sd_uc", "sd_c", "sd_brain", "sd_body", "sd_obs_brain", "sd_obs_body"])
    # plt.show()
    # print(model.parameters.__dict__)

    inference_summary = BrainBodyParticlesSummary.from_inference_data(idata)

    m = inference_summary.coordination_mean
    std = inference_summary.coordination_std

    plt.figure(figsize=(15, 8))
    plt.plot(range(TIME_STEPS), full_samples.coordination[0], label="Real", color="tab:blue", marker="o")
    plt.plot(range(TIME_STEPS), m, label="Inferred", color="tab:orange", marker="o")
    plt.fill_between(range(TIME_STEPS), m - std, m + std, color="tab:orange", alpha=0.4)
    plt.title("Coordination")
    plt.legend()
    plt.show()

    m = inference_summary.unbounded_coordination_mean
    std = inference_summary.unbounded_coordination_std

    plt.figure(figsize=(15, 8))
    plt.plot(range(TIME_STEPS), full_samples.unbounded_coordination[0], label="Real", color="tab:blue", marker="o")
    plt.plot(range(TIME_STEPS), m, label="Inferred", color="tab:orange", marker="o")
    plt.fill_between(range(TIME_STEPS), m - std, m + std, color="tab:orange", alpha=0.4)
    plt.title("Unbounded Coordination")
    plt.legend()
    plt.show()

    m = inference_summary.coordination_mean
    std = inference_summary.coordination_std

    plt.figure(figsize=(15, 8))
    plt.plot(range(TIME_STEPS), full_samples.coordination[0], label="Real", color="tab:blue", marker="o")
    plt.plot(range(TIME_STEPS), m, label="Inferred", color="tab:orange", marker="o")
    plt.fill_between(range(TIME_STEPS), m - std, m + std, color="tab:orange", alpha=0.4)
    plt.title("Coordination")
    plt.legend()
    plt.show()

    m = inference_summary.latent_brain_mean[0, 0]
    std = inference_summary.latent_brain_std[0, 0]

    plt.figure(figsize=(15, 8))
    plt.plot(range(TIME_STEPS), full_samples.latent_brain[0, 0, 0], label="Real", color="tab:blue",
             marker="o")
    plt.plot(range(TIME_STEPS), m, label="Inferred", color="tab:orange", marker="o")
    plt.fill_between(range(TIME_STEPS), m - std, m + std, color="tab:orange", alpha=0.4)
    plt.title("Brain")
    plt.legend()
    plt.show()
