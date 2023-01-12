import arviz as az
import matplotlib.pyplot as plt

from copy import deepcopy

from coordination.model.brain_body_model import BrainBodyModel
from coordination.model.utils.brain_body_model import BrainBodyDataset

import numpy as np
import pymc as pm

# Parameters
TIME_STEPS = 600
NUM_SERIES = 1
NUM_BRAIN_CHANNELS = 15
SEED = 0

if __name__ == "__main__":
    model = BrainBodyModel(0.5, NUM_BRAIN_CHANNELS, 3)

    model.parameters.sd_uc = np.array([0.1])
    model.parameters.sd_brain = np.array([1])
    model.parameters.sd_body = np.array([1])
    model.parameters.sd_obs_brain = np.array([1])
    model.parameters.sd_obs_body = np.array([1])

    full_samples = model.sample(NUM_SERIES, TIME_STEPS, SEED)

    partial_samples = deepcopy(full_samples)
    partial_samples.unbounded_coordination = None
    partial_samples.coordination = None
    partial_samples.latent_brain = None
    partial_samples.latent_body = None
    evidence = BrainBodyDataset.from_samples(partial_samples)

    coords = {"trial": np.arange(evidence.num_trials), "subject": np.arange(3),
              "brain_channel": np.arange(NUM_BRAIN_CHANNELS), "body_feature": np.arange(1),
              "time": np.arange(evidence.num_time_steps)}

    pymc_model = pm.Model(coords=coords)

    model.parameters.reset()
    idata = model.fit(evidence, 1000, 1000, 2, 0, 1, 6)
    az.plot_trace(idata, var_names=["sd_uc", "sd_brain", "sd_body", "sd_obs_brain", "sd_obs_body"])
    plt.show()
    print(model.parameters.__dict__)

    inference_summaries = model.predict(evidence, 1000, 1000, 2, 0, 1, 6)

    for i in range(NUM_SERIES):
        m = inference_summaries[i].coordination_mean
        std = inference_summaries[i].coordination_std

        plt.figure(figsize=(15, 8))
        plt.plot(range(TIME_STEPS), full_samples.coordination[i], label="Real", color="tab:blue", marker="o")
        plt.plot(range(TIME_STEPS), m, label="Inferred", color="tab:orange", marker="o")
        plt.fill_between(range(TIME_STEPS), m - std, m + std, color="tab:orange", alpha=0.4)
        plt.title("Coordination")
        plt.legend()
        plt.show()


