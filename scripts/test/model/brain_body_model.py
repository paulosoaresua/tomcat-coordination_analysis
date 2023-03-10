import sys

import arviz as az
import matplotlib.pyplot as plt
import numpy as np

from coordination.common.functions import logit
from coordination.model.brain_body_model import BrainBodyModel, BrainBodySeries
from coordination.model.brain_model import BrainSeries
from coordination.model.body_model import BodySeries

# Parameters
INITIAL_COORDINATION = 0.5
ESTIMATE_INITIAL_COORDINATION = True
TIME_STEPS = 200
NUM_SUBJECTS = 3
NUM_BRAIN_CHANNELS = 2
SEED = 0
SELF_DEPENDENT = False
N = 1000
C = 2
SHARE_PARAMS = True

PARAM_ONES_BRAIN = np.ones(NUM_BRAIN_CHANNELS) if SHARE_PARAMS else np.ones((NUM_SUBJECTS, NUM_BRAIN_CHANNELS))
PARAM_ZEROS_BRAIN = np.zeros(NUM_BRAIN_CHANNELS) if SHARE_PARAMS else np.zeros((NUM_SUBJECTS, NUM_BRAIN_CHANNELS))
PARAM_ONES_BODY = np.ones(1) if SHARE_PARAMS else np.ones((NUM_SUBJECTS, 1))
PARAM_ZEROS_BODY = np.zeros(1) if SHARE_PARAMS else np.zeros((NUM_SUBJECTS, 1))

if __name__ == "__main__":
    if not sys.warnoptions:
        # Prevent unimportant warnings from PyMC
        import warnings

        warnings.simplefilter("ignore")

    model = BrainBodyModel(initial_coordination=INITIAL_COORDINATION,
                           subjects=list(map(str, np.arange(NUM_SUBJECTS))),
                           brain_channels=list(map(str, np.arange(NUM_BRAIN_CHANNELS))),
                           self_dependent=True,
                           sd_mean_uc0=1,
                           sd_sd_uc=1,
                           sd_mean_a0_brain=PARAM_ONES_BRAIN,
                           sd_sd_aa_brain=PARAM_ONES_BRAIN,
                           sd_sd_o_brain=PARAM_ONES_BRAIN,
                           sd_mean_a0_body=PARAM_ONES_BODY,
                           sd_sd_aa_body=PARAM_ONES_BODY,
                           sd_sd_o_body=PARAM_ONES_BODY,
                           a_mixture_weights=np.ones((NUM_SUBJECTS, NUM_SUBJECTS - 1)),
                           share_params_across_subjects=SHARE_PARAMS)

    # Generate samples with different feature values per subject and different scales per feature
    model.coordination_cpn.parameters.sd_uc.value = np.array([1])
    model.latent_brain_cpn.parameters.mean_a0.value = np.array([[0.1, 2000], [0.5, 5000], [0.8, 9000]])
    model.latent_brain_cpn.parameters.sd_aa.value = np.array([[0.5, 1000], [0.5, 1000], [0.5, 1000]])
    model.latent_brain_cpn.parameters.mixture_weights.value = np.array([[0.3, 0.7], [0.8, 0.2], [0.1, 0.9]])
    model.latent_body_cpn.parameters.mean_a0.value = np.array([[20000], [50000], [90000]])
    model.latent_body_cpn.parameters.sd_aa.value = np.array([[10000], [10000], [10000]])
    model.latent_body_cpn.parameters.mixture_weights.value = model.latent_brain_cpn.parameters.mixture_weights.value
    model.obs_brain_cpn.parameters.sd_o.value = np.ones((NUM_SUBJECTS, NUM_BRAIN_CHANNELS))
    model.obs_body_cpn.parameters.sd_o.value = np.ones((NUM_SUBJECTS, 1))

    # Disable parameter sharing temporarily so we can generate samples
    model.share_params_across_subjects = False
    model.latent_brain_cpn.share_params_across_subjects = False
    model.latent_body_cpn.share_params_across_subjects = False
    model.obs_brain_cpn.share_params_across_subjects = False
    model.obs_body_cpn.share_params_across_subjects = False

    full_samples = model.draw_samples(num_series=1,
                                      num_time_steps=TIME_STEPS,
                                      brain_relative_frequency=4,
                                      body_relative_frequency=4,
                                      seed=SEED)

    model.share_params_across_subjects = SHARE_PARAMS
    model.latent_brain_cpn.share_params_across_subjects = SHARE_PARAMS
    model.latent_body_cpn.share_params_across_subjects = SHARE_PARAMS
    model.obs_brain_cpn.share_params_across_subjects = SHARE_PARAMS
    model.obs_body_cpn.share_params_across_subjects = SHARE_PARAMS

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

    evidence.normalize_across_subject()
    evidence.standardize()

    model.clear_parameter_values()
    if not ESTIMATE_INITIAL_COORDINATION:
        model.coordination_cpn.parameters.mean_uc0.value = np.array([logit(INITIAL_COORDINATION)])
    model.prior_predictive(evidence, 2)

    pymc_model, idata = model.fit(evidence=evidence,
                                  burn_in=N,
                                  num_samples=N,
                                  num_chains=C,
                                  seed=SEED,
                                  num_jobs=4)

    sampled_vars = set(idata.posterior.data_vars)
    var_names = list(set(model.parameter_names).intersection(sampled_vars))
    az.plot_trace(idata, var_names=var_names)
    plt.tight_layout()
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
