import sys

import arviz as az
import matplotlib.pyplot as plt
import numpy as np

from coordination.common.utils import set_random_seed
from coordination.common.functions import logit
from coordination.model.vocalic_model import VocalicModel, VocalicSeries
from coordination.model.vocalic_semantic_model import VocalicSemanticModel, VocalicSemanticSeries

# Parameters
INITIAL_COORDINATION = 0.5
ESTIMATE_INITIAL_COORDINATION = True
TIME_STEPS = 200
NUM_SUBJECTS = 3
NUM_VOCALIC_FEATURES = 2
TIME_SCALE_DENSITY = 1
SEED = 0
ADD_SEMANTIC_LINK = False
SELF_DEPENDENT = True
N = 1000
C = 2
SHARE_PARAMS_ACROSS_SUBJECTS_GEN = False
SHARE_PARAMS_ACROSS_GENDERS_GEN = True
SHARE_PARAMS_ACROSS_SUBJECTS_INF = False
SHARE_PARAMS_ACROSS_GENDERS_INF = True

# Different scales per features to test the model robustness
set_random_seed(SEED)
PARAM_SCALES = [10 ** (3 * i) for i in range(NUM_VOCALIC_FEATURES)]
if SHARE_PARAMS_ACROSS_SUBJECTS_GEN:
    TRUE_MEAN_AA = np.random.random(NUM_VOCALIC_FEATURES) * PARAM_SCALES
    TRUE_SD_AA = np.random.random(NUM_VOCALIC_FEATURES) * PARAM_SCALES
    TRUE_SD_OO = np.ones(NUM_VOCALIC_FEATURES)
elif SHARE_PARAMS_ACROSS_GENDERS_GEN:
    TRUE_MEAN_AA = np.random.random((2, NUM_VOCALIC_FEATURES)) * PARAM_SCALES
    TRUE_SD_AA = np.random.random((2, NUM_VOCALIC_FEATURES)) * PARAM_SCALES
    TRUE_SD_OO = np.ones((2, NUM_VOCALIC_FEATURES))
else:
    TRUE_MEAN_AA = np.random.random((NUM_SUBJECTS, NUM_VOCALIC_FEATURES)) * PARAM_SCALES
    TRUE_SD_AA = np.random.random((NUM_SUBJECTS, NUM_VOCALIC_FEATURES)) * PARAM_SCALES
    TRUE_SD_OO = np.ones((NUM_SUBJECTS, NUM_VOCALIC_FEATURES))

if SHARE_PARAMS_ACROSS_SUBJECTS_INF:
    PARAM_ONES = np.ones(NUM_VOCALIC_FEATURES)
    PARAM_ZEROS = np.zeros(NUM_VOCALIC_FEATURES)
elif SHARE_PARAMS_ACROSS_GENDERS_INF:
    PARAM_ONES = np.ones((2, NUM_VOCALIC_FEATURES))
    PARAM_ZEROS = np.zeros((2, NUM_VOCALIC_FEATURES))
else:
    PARAM_ONES = np.ones((NUM_SUBJECTS, NUM_VOCALIC_FEATURES))
    PARAM_ZEROS = np.zeros((NUM_SUBJECTS, NUM_VOCALIC_FEATURES))

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
                                     mean_mean_a0_vocalic=PARAM_ZEROS,
                                     sd_mean_a0_vocalic=PARAM_ONES,
                                     sd_sd_aa_vocalic=PARAM_ONES,
                                     sd_sd_o_vocalic=PARAM_ONES,
                                     a_p_semantic_link=1,
                                     b_p_semantic_link=1,
                                     initial_coordination=INITIAL_COORDINATION,
                                     share_params_across_subjects=SHARE_PARAMS_ACROSS_SUBJECTS_INF,
                                     share_params_across_genders=SHARE_PARAMS_ACROSS_GENDERS_INF)

        model.semantic_link_cpn.parameters.p.value = 0.7
    else:
        model = VocalicModel(num_subjects=NUM_SUBJECTS,
                             vocalic_features=list(map(str, np.arange(NUM_VOCALIC_FEATURES))),
                             self_dependent=SELF_DEPENDENT,
                             sd_mean_uc0=1,
                             sd_sd_uc=1,
                             # sd_sd_c=1,
                             mean_mean_a0_vocalic=PARAM_ZEROS,
                             sd_mean_a0_vocalic=PARAM_ONES,
                             sd_sd_aa_vocalic=PARAM_ONES,
                             sd_sd_o_vocalic=PARAM_ONES,
                             initial_coordination=INITIAL_COORDINATION,
                             share_params_across_subjects=SHARE_PARAMS_ACROSS_SUBJECTS_INF,
                             share_params_across_genders=SHARE_PARAMS_ACROSS_GENDERS_INF)

    # Generate samples with different feature values per subject and different scales per feature
    model.coordination_cpn.parameters.sd_uc.value = np.ones(1)
    # model.coordination_cpn.parameters.sd_c.value = np.ones(1)
    model.latent_vocalic_cpn.parameters.mean_a0.value = TRUE_MEAN_AA  # np.array([[0.1, 2000], [0.5, 5000], [0.8, 9000]])
    model.latent_vocalic_cpn.parameters.sd_aa.value = TRUE_SD_AA  # np.array([[0.5, 1000], [0.5, 1000], [0.5, 1000]])
    model.obs_vocalic_cpn.parameters.sd_o.value = TRUE_SD_OO  # np.ones((NUM_SUBJECTS, NUM_VOCALIC_FEATURES))

    # Disable parameter sharing temporarily so we can generate samples. This is because we want to test whether our
    # normalization strategies work when observations came from parameters that are different per subject or gender.
    model.share_params_across_subjects = SHARE_PARAMS_ACROSS_SUBJECTS_GEN
    model.latent_vocalic_cpn.share_params_across_subjects = SHARE_PARAMS_ACROSS_SUBJECTS_GEN
    model.obs_vocalic_cpn.share_params_across_subjects = SHARE_PARAMS_ACROSS_SUBJECTS_GEN
    model.share_params_across_genders = SHARE_PARAMS_ACROSS_GENDERS_GEN
    model.latent_vocalic_cpn.share_params_across_genders = SHARE_PARAMS_ACROSS_GENDERS_GEN
    model.obs_vocalic_cpn.share_params_across_genders = SHARE_PARAMS_ACROSS_GENDERS_GEN
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
    model.share_params_across_subjects = SHARE_PARAMS_ACROSS_SUBJECTS_INF
    model.latent_vocalic_cpn.share_params_across_subjects = SHARE_PARAMS_ACROSS_SUBJECTS_INF
    model.obs_vocalic_cpn.share_params_across_subjects = SHARE_PARAMS_ACROSS_SUBJECTS_INF
    model.share_params_across_genders = SHARE_PARAMS_ACROSS_GENDERS_INF
    model.latent_vocalic_cpn.share_params_across_genders = SHARE_PARAMS_ACROSS_GENDERS_INF
    model.obs_vocalic_cpn.share_params_across_genders = SHARE_PARAMS_ACROSS_GENDERS_INF

    evidence = VocalicSeries(uuid="",
                             features=list(map(str, np.arange(NUM_VOCALIC_FEATURES))),
                             num_time_steps_in_coordination_scale=TIME_STEPS,
                             subjects_in_time=full_samples.latent_vocalic.subjects[0],
                             observation=full_samples.obs_vocalic.values[0],
                             previous_time_same_subject=full_samples.latent_vocalic.prev_time_same_subject[0],
                             previous_time_diff_subject=full_samples.latent_vocalic.prev_time_diff_subject[0],
                             time_steps_in_coordination_scale=
                             full_samples.latent_vocalic.time_steps_in_coordination_scale[0],
                             gender_map=full_samples.latent_vocalic.gender_map)

    if ADD_SEMANTIC_LINK:
        evidence = VocalicSemanticSeries(uuid="",
                                         vocalic_series=evidence,
                                         semantic_link_time_steps_in_coordination_scale=
                                         full_samples.semantic_link.time_steps_in_coordination_scale[0])

    evidence.normalize_across_subject()
    # evidence.standardize()

    model.clear_parameter_values()
    if not ESTIMATE_INITIAL_COORDINATION:
        model.coordination_cpn.parameters.mean_uc0.value = np.array([logit(INITIAL_COORDINATION)])
    model.prior_predictive(evidence, 2)

    pymc_model, idata = model.fit(evidence=evidence,
                                  burn_in=N,
                                  num_samples=N,
                                  num_chains=C,
                                  seed=SEED,
                                  num_jobs=C)

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
