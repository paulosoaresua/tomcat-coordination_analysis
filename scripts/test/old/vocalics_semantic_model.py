import arviz as az
import matplotlib.pyplot as plt

import numpy as np

from coordination.model.vocalic_semantic_model import VocalicSemanticModel, VocalicSemanticSeries, \
    VocalicSemanticInferenceSummary
from coordination.common.functions import sigmoid

# Parameters
TIME_STEPS = 120
NUM_SUBJECTS = 3
NUM_VOCALIC_FEATURES = 2
SEED = 0

if __name__ == "__main__":
    model = VocalicSemanticModel(initial_coordination=0.5,
                                 num_subjects=NUM_SUBJECTS,
                                 num_vocalic_features=NUM_VOCALIC_FEATURES,
                                 self_dependent=True,
                                 sd_uc=1,
                                 sd_mean_a0_vocalic=np.ones((NUM_SUBJECTS, NUM_VOCALIC_FEATURES)),
                                 sd_sd_aa_vocalic=np.ones((NUM_SUBJECTS, NUM_VOCALIC_FEATURES)),
                                 sd_sd_o_vocalic=np.ones((NUM_SUBJECTS, NUM_VOCALIC_FEATURES)),
                                 a_p_semantic_link=1,
                                 b_p_semantic_link=1)

    model.coordination_cpn.parameters.sd_uc.value = np.array([0.1])
    model.latent_vocalic_cpn.parameters.mean_a0.value = np.zeros((NUM_SUBJECTS, NUM_VOCALIC_FEATURES))
    model.latent_vocalic_cpn.parameters.sd_aa.value = np.ones((NUM_SUBJECTS, NUM_VOCALIC_FEATURES))
    model.obs_vocalic_cpn.parameters.sd_o.value = np.ones((NUM_SUBJECTS, NUM_VOCALIC_FEATURES))
    model.semantic_link_cpn.parameters.p.value = 0.7

    full_samples = model.draw_samples(num_series=1,
                                      num_time_steps=TIME_STEPS,
                                      vocalic_time_scale_density=1,
                                      semantic_link_time_Scale_density=0.5,
                                      can_repeat_subject=False,
                                      seed=SEED)

    evidence = VocalicSemanticSeries(num_time_steps_in_coordination_scale=TIME_STEPS,
                                     vocalic_subjects=full_samples.latent_vocalic.subjects[0],
                                     obs_vocalic=full_samples.obs_vocalic.values[0],
                                     vocalic_prev_time_same_subject=full_samples.latent_vocalic.prev_time_same_subject[
                                         0],
                                     vocalic_prev_time_diff_subject=full_samples.latent_vocalic.prev_time_diff_subject[
                                         0],
                                     vocalic_time_steps_in_coordination_scale=
                                     full_samples.latent_vocalic.time_steps_in_coordination_scale[0],
                                     semantic_link_time_steps_in_coordination_scale=
                                     full_samples.semantic_link.time_steps_in_coordination_scale[0])

    model.clear_parameter_values()
    pymc_model, idata = model.fit(evidence=evidence,
                                  burn_in=1000,
                                  num_samples=1000,
                                  num_chains=2,
                                  seed=SEED,
                                  num_jobs=4)

    az.plot_trace(idata,
                  var_names=["sd_uc", "mean_a0_latent_vocalic", "sd_aa_latent_vocalic", "sd_o_obs_vocalic",
                             "p_semantic_link"])
    plt.show()

    # with pymc_model:
    #     samples = pm.sample_posterior_predictive(idata, var_names=["obs_brain"])

    inference_summary = VocalicSemanticInferenceSummary.from_inference_data(idata)

    m = inference_summary.coordination_means
    std = inference_summary.coordination_sds

    coordination_posterior = sigmoid(idata.posterior["unbounded_coordination"].sel(chain=0).to_numpy())

    plt.figure(figsize=(15, 8))
    # plt.fill_between(range(TIME_STEPS), m - std, m + std, color="tab:pink", alpha=0.4)
    plt.plot(np.arange(TIME_STEPS)[:, None].repeat(1000, axis=1), coordination_posterior.T, color="tab:blue", alpha=0.3)
    plt.plot(range(TIME_STEPS), full_samples.coordination.coordination[0], label="Real", color="black", marker="o",
             markersize=5)
    plt.scatter(full_samples.semantic_link.time_steps_in_coordination_scale[0],
                full_samples.coordination.coordination[
                    0, full_samples.semantic_link.time_steps_in_coordination_scale[0]],
                c="white", marker="*", s=3, zorder=3)
    plt.plot(range(TIME_STEPS), m, label="Inferred", color="tab:pink", markersize=5)
    plt.title("Coordination")
    plt.legend()
    plt.show()

    print(f"Real Coordination Average: {full_samples.coordination.coordination[0].mean()}")
    print(f"Estimated Coordination Average: {m.mean()}")
