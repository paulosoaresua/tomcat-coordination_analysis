import arviz as az
import matplotlib.pyplot as plt

from coordination.model.brain_body_model import BrainBodyModel, BrainBodySeries, BrainBodyInferenceSummary

from coordination.common.functions import sigmoid

import numpy as np

# Parameters
TIME_STEPS = 1200
NUM_SUBJECTS = 3
NUM_BRAIN_CHANNELS = 20
SEED = 0

if __name__ == "__main__":
    model = BrainBodyModel(initial_coordination=0.5,
                           num_subjects=NUM_SUBJECTS,
                           num_brain_channels=NUM_BRAIN_CHANNELS,
                           self_dependent=True,
                           sd_uc=1,
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

    evidence = BrainBodySeries(num_time_steps_in_coordination_scale=TIME_STEPS,
                               obs_brain=full_samples.obs_brain.values[0],
                               brain_time_steps_in_coordination_scale=
                               full_samples.latent_brain.time_steps_in_coordination_scale[0],
                               obs_body=full_samples.obs_body.values[0],
                               body_time_steps_in_coordination_scale=
                               full_samples.latent_body.time_steps_in_coordination_scale[0])

    model.clear_parameter_values()
    pymc_model, idata = model.fit(evidence=evidence,
                                  burn_in=1000,
                                  num_samples=1000,
                                  num_chains=2,
                                  seed=SEED,
                                  num_jobs=4)

    az.plot_trace(idata,
                  var_names=["sd_uc", "mean_a0_latent_brain", "sd_aa_latent_brain", "mean_a0_latent_body",
                             "sd_aa_latent_body", "sd_o_obs_brain", "sd_o_obs_body", "mixture_weights_latent_brain"])
    plt.show()

    # with pymc_model:
    #     samples = pm.sample_posterior_predictive(idata, var_names=["obs_brain"])

    inference_summary = BrainBodyInferenceSummary.from_inference_data(idata)

    m = inference_summary.coordination_means
    std = inference_summary.coordination_sds

    coordination_posterior = sigmoid(idata.posterior["unbounded_coordination"].sel(chain=0).to_numpy())

    plt.figure(figsize=(15, 8))
    # plt.fill_between(range(TIME_STEPS), m - std, m + std, color="tab:pink", alpha=0.4)
    plt.plot(np.arange(TIME_STEPS)[:, None].repeat(1000, axis=1), coordination_posterior.T, color="tab:blue", alpha=0.3)
    plt.plot(range(TIME_STEPS), full_samples.coordination.coordination[0], label="Real", color="black", marker="o",
             markersize=5)
    plt.plot(range(TIME_STEPS), m, label="Inferred", color="tab:pink", markersize=5)
    plt.title("Coordination")
    plt.legend()
    plt.show()

    print(f"Real Coordination Average: {full_samples.coordination.coordination[0].mean()}")
    print(f"Estimated Coordination Average: {m.mean()}")
