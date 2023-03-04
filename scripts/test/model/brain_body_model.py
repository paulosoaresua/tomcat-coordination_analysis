import sys

import arviz as az
import matplotlib.pyplot as plt
import numpy as np

from coordination.model.brain_body_model import BrainBodyModel, BrainBodySeries
from coordination.model.brain_model import BrainSeries
from coordination.model.body_model import BodySeries

# Parameters
TIME_STEPS = 200
NUM_SUBJECTS = 3
NUM_BRAIN_CHANNELS = 10
SEED = 0
SELF_DEPENDENT = True
N = 1000
C = 2

if __name__ == "__main__":
    if not sys.warnoptions:
        # Prevent unimportant warnings from PyMC
        import warnings

        warnings.simplefilter("ignore")

    model = BrainBodyModel(initial_coordination=0.5,
                           subjects=list(map(str, np.arange(NUM_SUBJECTS))),
                           brain_channels=list(map(str, np.arange(NUM_BRAIN_CHANNELS))),
                           self_dependent=True,
                           sd_mean_uc0=1,
                           sd_sd_uc=1,
                           sd_mean_a0_brain=np.ones((NUM_SUBJECTS, NUM_BRAIN_CHANNELS)),
                           sd_sd_aa_brain=np.ones((NUM_SUBJECTS, NUM_BRAIN_CHANNELS)),
                           sd_sd_o_brain=np.ones((NUM_SUBJECTS, NUM_BRAIN_CHANNELS)),
                           sd_mean_a0_body=np.ones((NUM_SUBJECTS, 1)),
                           sd_sd_aa_body=np.ones((NUM_SUBJECTS, 1)),
                           sd_sd_o_body=np.ones((NUM_SUBJECTS, 1)),
                           a_mixture_weights=np.ones((NUM_SUBJECTS, NUM_SUBJECTS - 1)))

    model.coordination_cpn.parameters.sd_uc.value = np.array([0.1])
    model.latent_brain_cpn.parameters.mean_a0.value = np.zeros((NUM_SUBJECTS, NUM_BRAIN_CHANNELS))
    model.latent_brain_cpn.parameters.sd_aa.value = np.ones((NUM_SUBJECTS, NUM_BRAIN_CHANNELS))
    model.latent_brain_cpn.parameters.mixture_weights.value = np.array([[0.3, 0.7], [0.8, 0.2], [0.1, 0.9]])
    model.latent_body_cpn.parameters.mean_a0.value = np.zeros((NUM_SUBJECTS, 1))
    model.latent_body_cpn.parameters.sd_aa.value = np.ones((NUM_SUBJECTS, 1))
    model.latent_body_cpn.parameters.mixture_weights.value = model.latent_brain_cpn.parameters.mixture_weights.value
    model.obs_brain_cpn.parameters.sd_o.value = np.ones((NUM_SUBJECTS, NUM_BRAIN_CHANNELS))
    model.obs_body_cpn.parameters.sd_o.value = np.ones((NUM_SUBJECTS, 1))

    full_samples = model.draw_samples(num_series=1,
                                      num_time_steps=TIME_STEPS,
                                      brain_relative_frequency=4,
                                      body_relative_frequency=4,
                                      seed=SEED)

    brain_series = BrainSeries(uuid="",
                               subjects=model.subjects,
                               channels=model.brain_channels,
                               num_time_steps_in_coordination_scale=TIME_STEPS,
                               observation=full_samples.obs_brain.values[0],
                               time_steps_in_coordination_scale=
                               full_samples.latent_brain.time_steps_in_coordination_scale[0])

    body_series = BodySeries(uuid="",
                             subjects=model.subjects,
                             num_time_steps_in_coordination_scale=TIME_STEPS,
                             observation=full_samples.obs_body.values[0],
                             time_steps_in_coordination_scale=
                             full_samples.latent_body.time_steps_in_coordination_scale[0])

    evidence = BrainBodySeries(uuid="",
                               brain_series=brain_series,
                               body_series=body_series)

    model.clear_parameter_values()
    pymc_model, idata = model.fit(evidence=evidence,
                                  burn_in=N,
                                  num_samples=N,
                                  num_chains=C,
                                  seed=SEED,
                                  num_jobs=C)

    az.plot_trace(idata,
                  var_names=["sd_uc", "mean_a0_latent_brain", "sd_aa_latent_brain", "mean_a0_latent_body",
                             "sd_aa_latent_body", "sd_o_obs_brain", "sd_o_obs_body", "mixture_weights_latent_brain"])
    plt.show()

    posterior_samples = model.inference_data_to_posterior_samples(idata)

    stacked_coordination_samples = posterior_samples.coordination.stack(chain_plus_draw=("chain", "draw"))
    avg_coordination = posterior_samples.coordination.mean(dim=["chain", "draw"]).to_numpy()

    plt.figure(figsize=(15, 8))
    plt.plot(np.arange(TIME_STEPS)[:, None].repeat(N * C, axis=1), stacked_coordination_samples, color="tab:blue",
             alpha=0.3,
             zorder=1)
    plt.plot(range(TIME_STEPS), full_samples.coordination.coordination[0], label="Real", color="black", marker="o",
             markersize=5, zorder=2)
    plt.plot(range(TIME_STEPS), avg_coordination, label="Inferred", color="tab:pink", markersize=5, zorder=3)

    plt.title("Coordination")
    plt.legend()
    plt.show()

    print(f"Real Coordination Average: {full_samples.coordination.coordination[0].mean()}")
    print(f"Estimated Coordination Average: {avg_coordination.mean()}")
