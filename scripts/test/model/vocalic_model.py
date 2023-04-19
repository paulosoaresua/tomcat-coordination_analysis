import sys

import arviz as az
import matplotlib.pyplot as plt
import numpy as np

from coordination.common.utils import set_random_seed
from coordination.common.functions import logit
from coordination.model.coordination_model import CoordinationPosteriorSamples
from coordination.model.vocalic_model import VocalicModel, VocalicSeries
from coordination.model.vocalic_semantic_model import VocalicSemanticModel, VocalicSemanticSeries
from coordination.component.serialized_component import Mode

# Parameters
MODE = Mode.BLENDING
INITIAL_COORDINATION = 0.5
ESTIMATE_INITIAL_COORDINATION = True
TIME_STEPS = 200
NUM_SUBJECTS = 3
NUM_VOCALIC_FEATURES = 2
TIME_SCALE_DENSITY = 1
SEED = 1  # 1, 7
ADD_SEMANTIC_LINK = False
SELF_DEPENDENT = True

# Function f(.)
NUM_HIDDEN_LAYERS_F = 2
DIM_HIDDEN_LAYER_F = 4  # NUM_VOCALIC_FEATURES
if NUM_HIDDEN_LAYERS_F > 0:
    F = lambda x, d, s: x + 0.1
else:
    F = None
ACTIVATION_FUNCTION_F = "tanh"

# Emission function
NUM_LAYERS_EMISSION_NN = 0
ACTIVATIONS_EMISSION_NN = "linear"

N = 1000
C = 2
SHARE_PARAMS_ACROSS_SUBJECTS_GEN = False
SHARE_PARAMS_ACROSS_GENDERS_GEN = False
SHARE_PARAMS_ACROSS_FEATURES_GEN = False

SHARE_PARAMS_ACROSS_SUBJECTS_INF = True
SHARE_PARAMS_ACROSS_GENDERS_INF = False
SHARE_PARAMS_ACROSS_FEATURES_INF = False

# Different scales per features to test the model robustness
set_random_seed(SEED)
DIM = 1 if SHARE_PARAMS_ACROSS_FEATURES_GEN else NUM_VOCALIC_FEATURES

# PARAM_SCALES = [10 ** (3 * i) for i in range(DIM)]
PARAM_SCALES = [1 ** (3 * i) for i in range(DIM)]
if SHARE_PARAMS_ACROSS_SUBJECTS_GEN:
    TRUE_MEAN_AA = np.random.random(DIM) * PARAM_SCALES
    TRUE_SD_AA = np.random.random(DIM) * PARAM_SCALES
    TRUE_SD_OO = np.ones(DIM)
elif SHARE_PARAMS_ACROSS_GENDERS_GEN:
    TRUE_MEAN_AA = np.random.random((2, DIM)) * PARAM_SCALES
    TRUE_SD_AA = np.random.random((2, DIM)) * PARAM_SCALES
    TRUE_SD_OO = np.ones((2, DIM))
else:
    TRUE_MEAN_AA = np.random.random((NUM_SUBJECTS, DIM)) * PARAM_SCALES
    TRUE_SD_AA = np.random.random((NUM_SUBJECTS, DIM)) * PARAM_SCALES
    TRUE_SD_OO = np.ones((NUM_SUBJECTS, DIM))

DIM = 1 if SHARE_PARAMS_ACROSS_FEATURES_INF else NUM_VOCALIC_FEATURES
if SHARE_PARAMS_ACROSS_SUBJECTS_INF:
    PARAM_ONES = np.ones(DIM)
    PARAM_ZEROS = np.zeros(DIM)
elif SHARE_PARAMS_ACROSS_GENDERS_INF:
    PARAM_ONES = np.ones((2, DIM))
    PARAM_ZEROS = np.zeros((2, DIM))
else:
    PARAM_ONES = np.ones((NUM_SUBJECTS, DIM))
    PARAM_ZEROS = np.zeros((NUM_SUBJECTS, DIM))

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
                                     share_params_across_genders=SHARE_PARAMS_ACROSS_GENDERS_INF,
                                     share_params_across_features_latent=SHARE_PARAMS_ACROSS_FEATURES_INF,
                                     share_params_across_features_observation=SHARE_PARAMS_ACROSS_FEATURES_INF,
                                     mode=MODE)

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
                             share_params_across_genders=SHARE_PARAMS_ACROSS_GENDERS_INF,
                             share_params_across_features_latent=SHARE_PARAMS_ACROSS_FEATURES_INF,
                             share_params_across_features_observation=SHARE_PARAMS_ACROSS_FEATURES_INF,
                             mode=MODE,
                             f=F,
                             num_hidden_layers_f=NUM_HIDDEN_LAYERS_F,
                             activation_function_name_f=ACTIVATION_FUNCTION_F,
                             dim_hidden_layer_f=DIM_HIDDEN_LAYER_F,
                             nn_layers_emission=NUM_LAYERS_EMISSION_NN,
                             nn_activation_emission=ACTIVATIONS_EMISSION_NN)

    # Generate samples with different feature values per subject and different scales per feature
    model.coordination_cpn.parameters.sd_uc.value = np.ones(1)
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
    model.share_params_across_features = SHARE_PARAMS_ACROSS_FEATURES_GEN
    model.latent_vocalic_cpn.share_params_across_features = SHARE_PARAMS_ACROSS_FEATURES_GEN
    model.obs_vocalic_cpn.share_params_across_features = SHARE_PARAMS_ACROSS_FEATURES_GEN
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

    # plt.figure()
    # plt.plot(np.arange(TIME_STEPS), full_samples.coordination.coordination[0])
    # plt.show()

    model.share_params_across_subjects = SHARE_PARAMS_ACROSS_SUBJECTS_INF
    model.latent_vocalic_cpn.share_params_across_subjects = SHARE_PARAMS_ACROSS_SUBJECTS_INF
    model.obs_vocalic_cpn.share_params_across_subjects = SHARE_PARAMS_ACROSS_SUBJECTS_INF
    model.share_params_across_genders = SHARE_PARAMS_ACROSS_GENDERS_INF
    model.latent_vocalic_cpn.share_params_across_genders = SHARE_PARAMS_ACROSS_GENDERS_INF
    model.obs_vocalic_cpn.share_params_across_genders = SHARE_PARAMS_ACROSS_GENDERS_INF
    model.share_params_across_features = SHARE_PARAMS_ACROSS_FEATURES_INF
    model.latent_vocalic_cpn.share_params_across_features = SHARE_PARAMS_ACROSS_FEATURES_INF
    model.obs_vocalic_cpn.share_params_across_features = SHARE_PARAMS_ACROSS_FEATURES_INF

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

    # if SHARE_PARAMS_ACROSS_GENDERS_GEN:
    #     # Add different offsets to observations from different users' genders
    #     offsets = [10 ** i for i in range(2)]
    #     genders_in_time = np.array([evidence.gender_map[s] for s in evidence.subjects_in_time])
    #     for i, gender in enumerate([0, 1]):
    #         evidence.observation[:, genders_in_time == gender] += offsets[i]
    # elif SHARE_PARAMS_ACROSS_SUBJECTS_GEN:
    #     # Add different offsets to observations from different users
    #     all_subjects = set(evidence.subjects_in_time)
    #
    #     offsets = [10 ** i for i in range(NUM_SUBJECTS)]
    #     for i, subject in enumerate(all_subjects):
    #         evidence.observation[:, evidence.subjects_in_time == subject] += offsets[i]

    # if SHARE_PARAMS_ACROSS_GENDERS_INF:
    #     evidence.normalize_per_gender()
    # elif SHARE_PARAMS_ACROSS_SUBJECTS_INF:
    #     evidence.normalize_per_subject()

    # b00 = 5 / evidence.observation[0, evidence.subjects_in_time == 0].std()
    # b01 = 5 / evidence.observation[1, evidence.subjects_in_time == 0].std()
    # b10 = 5 / evidence.observation[0, evidence.subjects_in_time == 1].std()
    # b11 = 5 / evidence.observation[1, evidence.subjects_in_time == 1].std()
    # b20 = 5 / evidence.observation[0, evidence.subjects_in_time == 2].std()
    # b21 = 5 / evidence.observation[1, evidence.subjects_in_time == 2].std()

    model.clear_parameter_values()
    # model.latent_vocalic_cpn.parameters.weights_f = [
    #     # Input layer
    #     np.array([[1, 0],
    #               [0, 1],
    #               [0, 0],
    #               [0, 0],
    #               [0, 0],
    #               [0.1, 0.1]]),
    #
    #     # Hidden layers
    #     np.array([
    #         np.array([[1, 0],
    #                   [0, 1],
    #                   [0, 0]])
    #     ]),
    #
    #     # Output layer
    #     np.array([[1, 0],
    #               [0, 1],
    #               [0, 0]])
    # ]
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

    posterior_samples = CoordinationPosteriorSamples.from_inference_data(idata)
    estimated = posterior_samples.coordination.mean(dim=["chain", "draw"]).to_numpy()

    plt.figure(figsize=(15, 8))
    posterior_samples.plot(plt.gca(), line_width=5)
    plt.plot(range(TIME_STEPS), full_samples.coordination.coordination[0], label="Real", color="black", zorder=3)

    if ADD_SEMANTIC_LINK:
        time_steps = full_samples.semantic_link.time_steps_in_coordination_scale[0]
        plt.scatter(time_steps, estimated[time_steps] + 0.05, c="white", marker="*", s=10, zorder=4,
                    label="Semantic Link")
    plt.title("Coordination")
    plt.legend()
    plt.show()

    real = full_samples.coordination.coordination[0]

    print(f"Real Coordination Average: {real.mean()}")
    print(f"Estimated Coordination Average: {estimated.mean()}")

    mse = np.sqrt(np.square(real - estimated).sum())
    print(f"MSE = {mse}")

    print("")
    print("f(.)")
    print(idata.posterior[f"f_nn_weights_latent_vocalic_in"].mean(dim=["chain", "draw"]))
    print(idata.posterior[f"f_nn_weights_latent_vocalic_out"].mean(dim=["chain", "draw"]))
    print(idata.posterior[f"f_nn_weights_latent_vocalic_hidden"].mean(dim=["chain", "draw"]))
