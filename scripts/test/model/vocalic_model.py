import sys

import arviz as az
import matplotlib.pyplot as plt
import numpy as np

from coordination.model.vocalic_model import VocalicModel, VocalicSeries
from coordination.model.vocalic_semantic_model import VocalicSemanticModel, VocalicSemanticSeries
from coordination.common.functions import logit

# Parameters
INITIAL_COORDINATION = 0.5
TIME_STEPS = 200
NUM_SUBJECTS = 3
NUM_VOCALIC_FEATURES = 4
TIME_SCALE_DENSITY = 0.2
SEED = 0
ADD_SEMANTIC_LINK = False
SELF_DEPENDENT = True
N = 1000
C = 2

if __name__ == "__main__":
    if not sys.warnoptions:
        # Prevent unimportant warnings from PyMC
        import warnings

        warnings.simplefilter("ignore")

    if ADD_SEMANTIC_LINK:
        model = VocalicSemanticModel(num_subjects=NUM_SUBJECTS,
                                     vocalic_features=list(map(str, np.arange(NUM_VOCALIC_FEATURES))),
                                     self_dependent=SELF_DEPENDENT,
                                     sd_mean_uc0=1,
                                     sd_sd_uc=1,
                                     sd_mean_a0_vocalic=np.ones((NUM_SUBJECTS, NUM_VOCALIC_FEATURES)),
                                     sd_sd_aa_vocalic=np.ones((NUM_SUBJECTS, NUM_VOCALIC_FEATURES)),
                                     sd_sd_o_vocalic=np.ones((NUM_SUBJECTS, NUM_VOCALIC_FEATURES)),
                                     a_p_semantic_link=1,
                                     b_p_semantic_link=1,
                                     initial_coordination=INITIAL_COORDINATION)

        model.semantic_link_cpn.parameters.p.value = 0.7
    else:
        model = VocalicModel(num_subjects=NUM_SUBJECTS,
                             vocalic_features=list(map(str, np.arange(NUM_VOCALIC_FEATURES))),
                             self_dependent=SELF_DEPENDENT,
                             sd_mean_uc0=1,
                             sd_sd_uc=1,
                             sd_mean_a0_vocalic=np.ones((NUM_SUBJECTS, NUM_VOCALIC_FEATURES)),
                             sd_sd_aa_vocalic=np.ones((NUM_SUBJECTS, NUM_VOCALIC_FEATURES)),
                             sd_sd_o_vocalic=np.ones((NUM_SUBJECTS, NUM_VOCALIC_FEATURES)),
                             initial_coordination=INITIAL_COORDINATION)

    model.coordination_cpn.parameters.sd_uc.value = np.array([1])
    model.latent_vocalic_cpn.parameters.mean_a0.value = np.zeros((NUM_SUBJECTS, NUM_VOCALIC_FEATURES))
    model.latent_vocalic_cpn.parameters.sd_aa.value = np.ones((NUM_SUBJECTS, NUM_VOCALIC_FEATURES))
    model.obs_vocalic_cpn.parameters.sd_o.value = np.ones((NUM_SUBJECTS, NUM_VOCALIC_FEATURES))

    if ADD_SEMANTIC_LINK:
        full_samples = model.draw_samples(num_series=1,
                                          num_time_steps=TIME_STEPS,
                                          vocalic_time_scale_density=TIME_SCALE_DENSITY,
                                          semantic_link_time_scale_density=TIME_SCALE_DENSITY,
                                          can_repeat_subject=False,
                                          seed=SEED)
    else:
        full_samples = model.draw_samples(num_series=1,
                                          num_time_steps=TIME_STEPS,
                                          vocalic_time_scale_density=TIME_SCALE_DENSITY,
                                          can_repeat_subject=False,
                                          seed=SEED)

    evidence = VocalicSeries(uuid="",
                             features=list(map(str, np.arange(NUM_VOCALIC_FEATURES))),
                             num_time_steps_in_coordination_scale=TIME_STEPS,
                             subjects_in_time=full_samples.latent_vocalic.subjects[0],
                             observation=full_samples.obs_vocalic.values[0],
                             previous_time_same_subject=full_samples.latent_vocalic.prev_time_same_subject[0],
                             previous_time_diff_subject=full_samples.latent_vocalic.prev_time_diff_subject[0],
                             time_steps_in_coordination_scale=
                             full_samples.latent_vocalic.time_steps_in_coordination_scale[0])

    if ADD_SEMANTIC_LINK:
        evidence = VocalicSemanticSeries(uuid="",
                                         vocalic_series=evidence,
                                         semantic_link_time_steps_in_coordination_scale=
                                         full_samples.semantic_link.time_steps_in_coordination_scale[0])

    model.clear_parameter_values()
    pymc_model, idata = model.fit(evidence=evidence,
                                  burn_in=N,
                                  num_samples=N,
                                  num_chains=C,
                                  seed=SEED,
                                  num_jobs=C)

    az.plot_trace(idata, var_names=model.parameter_names)
    plt.show()

    posterior_samples = model.inference_data_to_posterior_samples(idata)

    stacked_coordination_samples = posterior_samples.coordination.stack(chain_plus_draw=("chain", "draw"))
    avg_coordination = posterior_samples.coordination.mean(dim=["chain", "draw"]).to_numpy()

    plt.figure(figsize=(15, 8))
    plt.plot(np.arange(TIME_STEPS)[:, None].repeat(N * C, axis=1), stacked_coordination_samples, color="tab:blue",
             alpha=0.3, zorder=1)
    plt.plot(range(TIME_STEPS), full_samples.coordination.coordination[0], label="Real", color="black", marker="o",
             markersize=5, zorder=2)
    plt.plot(range(TIME_STEPS), avg_coordination, label="Inferred", color="tab:pink", markersize=5, zorder=3)

    if ADD_SEMANTIC_LINK:
        plt.scatter(full_samples.semantic_link.time_steps_in_coordination_scale[0],
                    full_samples.coordination.coordination[
                        0, full_samples.semantic_link.time_steps_in_coordination_scale[0]], c="white", marker="*", s=3,
                    zorder=4)
    plt.title("Coordination")
    plt.legend()
    plt.show()

    print(f"Real Coordination Average: {full_samples.coordination.coordination[0].mean()}")
    print(f"Estimated Coordination Average: {avg_coordination.mean()}")
