import arviz as az
import matplotlib.pyplot as plt

import numpy as np

from coordination.model.vocalic_model import VocalicModel, VocalicSeries, VocalicPosteriorSamples
from coordination.common.functions import logit

# Parameters
TIME_STEPS = 200
NUM_SUBJECTS = 3
NUM_VOCALIC_FEATURES = 2
SEED = 0

if __name__ == "__main__":
    model = VocalicModel(num_subjects=NUM_SUBJECTS,
                         vocalic_features=list(map(str, np.arange(NUM_VOCALIC_FEATURES))),
                         self_dependent=False,
                         sd_mean_uc0=1,
                         sd_sd_uc=1,
                         sd_mean_a0_vocalic=np.ones((NUM_SUBJECTS, NUM_VOCALIC_FEATURES)),
                         sd_sd_aa_vocalic=np.ones((NUM_SUBJECTS, NUM_VOCALIC_FEATURES)),
                         sd_sd_o_vocalic=np.ones((NUM_SUBJECTS, NUM_VOCALIC_FEATURES)))

    model.coordination_cpn.parameters.mean_uc0.value = logit(0.5)
    model.coordination_cpn.parameters.sd_uc.value = np.array([2.5])
    model.latent_vocalic_cpn.parameters.mean_a0.value = np.zeros((NUM_SUBJECTS, NUM_VOCALIC_FEATURES))
    model.latent_vocalic_cpn.parameters.sd_aa.value = np.ones((NUM_SUBJECTS, NUM_VOCALIC_FEATURES))
    model.obs_vocalic_cpn.parameters.sd_o.value = np.ones((NUM_SUBJECTS, NUM_VOCALIC_FEATURES))

    full_samples = model.draw_samples(num_series=1,
                                      num_time_steps=TIME_STEPS,
                                      vocalic_time_scale_density=1,
                                      can_repeat_subject=False,
                                      seed=SEED)

    evidence = VocalicSeries(uuid="",
                             vocalic_features=list(map(str, np.arange(NUM_VOCALIC_FEATURES))),
                             num_time_steps_in_coordination_scale=TIME_STEPS,
                             vocalic_subjects=full_samples.latent_vocalic.subjects[0],
                             obs_vocalic=full_samples.obs_vocalic.values[0],
                             vocalic_prev_time_same_subject=full_samples.latent_vocalic.prev_time_same_subject[0],
                             vocalic_prev_time_diff_subject=full_samples.latent_vocalic.prev_time_diff_subject[0],
                             vocalic_time_steps_in_coordination_scale=
                             full_samples.latent_vocalic.time_steps_in_coordination_scale[0])

    # model.prior_predictive(evidence, 500, 0)

    model.clear_parameter_values()
    pymc_model, idata = model.fit(evidence=evidence,
                                  burn_in=1000,
                                  num_samples=1000,
                                  num_chains=2,
                                  seed=SEED,
                                  num_jobs=4)

    az.plot_trace(idata,
                  var_names=["sd_uc", "mean_a0_latent_vocalic", "sd_aa_latent_vocalic", "sd_o_obs_vocalic"])
    plt.show()

    posterior_samples = VocalicPosteriorSamples.from_inference_data(idata)

    coordination_posterior = posterior_samples.coordination.sel(chain=0)
    avg_coordination = posterior_samples.coordination.mean(dim=["chain", "draw"]).to_numpy()

    plt.figure(figsize=(15, 8))
    plt.plot(np.arange(TIME_STEPS)[:, None].repeat(coordination_posterior.shape[0], axis=1), coordination_posterior.T,
             color="tab:blue", alpha=0.3)
    plt.plot(range(TIME_STEPS), full_samples.coordination.coordination[0], label="Real", color="black", marker="o",
             markersize=5)
    plt.plot(range(TIME_STEPS), avg_coordination, label="Inferred", color="tab:pink", markersize=5)
    plt.title("Coordination")
    plt.legend()
    plt.show()

    print(f"Real Coordination Average: {full_samples.coordination.coordination[0].mean()}")
    print(f"Estimated Coordination Average: {avg_coordination.mean()}")
