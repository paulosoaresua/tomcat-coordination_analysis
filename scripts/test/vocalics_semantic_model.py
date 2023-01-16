import arviz as az
import matplotlib.pyplot as plt

from copy import deepcopy

from coordination.model.vocalics_semantic_model import VocalicsSemanticModel, serialized_logp
from coordination.model.utils.beta_coordination_blending_latent_vocalics import BetaCoordinationLatentVocalicsDataset, \
    BetaCoordinationLatentVocalicsParticlesSummary

import numpy as np
import pymc as pm

from coordination.common.utils import sigmoid, logit

# Parameters
TIME_STEPS = 100
NUM_SERIES = 1
NUM_FEATURES = 10
SEED = 0

if __name__ == "__main__":
    model = VocalicsSemanticModel(0.5, NUM_FEATURES, 3)

    model.parameters.sd_uc = np.array([0.1])
    model.parameters.sd_vocalics = np.array([0.1])
    model.parameters.sd_obs_vocalics = np.array([0.1])

    full_samples = model.sample(NUM_SERIES, TIME_STEPS, SEED, time_scale_density=1, p_semantic_links=0.5)

    partial_samples = deepcopy(full_samples)
    partial_samples.unbounded_coordination = None
    partial_samples.coordination = None
    partial_samples.latent_vocalics = None
    evidence = BetaCoordinationLatentVocalicsDataset.from_samples(partial_samples)

    # import pytensor.tensor as at
    #
    # serialized_logp(at.constant(evidence.observed_vocalics),
    #                 at.constant(full_samples.coordination[0]),
    #                 at.constant(np.array([1])),
    #                 at.constant(evidence.previous_vocalics_from_self),
    #                 at.constant(evidence.previous_vocalics_from_other),
    #                 at.constant(evidence.previous_vocalics_from_self_mask),
    #                 at.constant(evidence.previous_vocalics_from_other_mask),
    #                 at.constant(evidence.vocalics_mask))

    model.parameters.reset()
    idata = model.fit(evidence.series[0], 1000, 1000, 2, 0, 6)
    az.plot_trace(idata, var_names=["sd_uc", "sd_c", "sd_vocalics", "sd_obs_vocalics"])
    plt.show()
    print(model.parameters.__dict__)

    inference_summary = BetaCoordinationLatentVocalicsParticlesSummary.from_inference_data(idata)

    m = inference_summary.unbounded_coordination_mean
    std = np.sqrt(inference_summary.unbounded_coordination_var)

    plt.figure(figsize=(15, 8))
    plt.scatter(evidence.series[0].speech_semantic_links_times, evidence.series[0].speech_semantic_links_vector_of_ones,
                color="black", marker="s")
    plt.plot(range(TIME_STEPS), full_samples.unbounded_coordination[0], label="Real", color="tab:blue", marker="o")
    plt.plot(range(TIME_STEPS), m, label="Inferred", color="tab:orange", marker="o")
    plt.fill_between(range(TIME_STEPS), m - std, m + std, color="tab:orange", alpha=0.4)
    plt.title("Unbounded Coordination")
    plt.legend()
    plt.show()

    m = inference_summary.coordination_mean
    std = np.sqrt(inference_summary.coordination_var)

    plt.figure(figsize=(15, 8))
    plt.scatter(evidence.series[0].speech_semantic_links_times, evidence.series[0].speech_semantic_links_vector_of_ones,
                color="black", marker="s")
    plt.plot(range(TIME_STEPS), full_samples.coordination[0], label="Real", color="tab:blue", marker="o")
    plt.plot(range(TIME_STEPS), m, label="Inferred", color="tab:orange", marker="o")
    plt.fill_between(range(TIME_STEPS), m - std, m + std, color="tab:orange", alpha=0.4)
    plt.title("Coordination")
    plt.legend()
    plt.show()

    time_steps = [t for t, m in enumerate(evidence.series[0].vocalics_mask) if m == 1]

    m = inference_summary.latent_vocalics_mean[0, time_steps]
    std = np.sqrt(inference_summary.latent_vocalics_var)[0, time_steps]

    plt.figure(figsize=(15, 8))
    plt.plot(time_steps, full_samples.latent_vocalics[0].values[0, time_steps], label="Real", color="tab:blue",
             marker="o")
    plt.plot(time_steps, m, label="Inferred", color="tab:orange", marker="o")
    plt.fill_between(time_steps, m - std, m + std, color="tab:orange", alpha=0.4)
    plt.title("Vocalics")
    plt.legend()
    plt.show()
