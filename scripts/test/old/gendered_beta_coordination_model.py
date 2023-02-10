from typing import Optional

from copy import copy

import matplotlib.pyplot as plt
import numpy as np

from coordination.common.log import BaseLogger, TensorBoardLogger
from coordination.model.gendered_beta_coordination_blending_latent_vocalics import \
    GenderedBetaCoordinationBlendingLatentVocalics
from coordination.model.utils.beta_coordination_blending_latent_vocalics import BetaCoordinationLatentVocalicsDataset
from coordination.model.utils.gendered_beta_coordination_blending_latent_vocalics import \
    GenderedBetaCoordinationLatentVocalicsTrainingHyperParameters, GenderedBetaCoordinationLatentVocalicsModelParameters

# Parameters
TIME_STEPS = 50
NUM_SAMPLES = 100
NUM_FEATURES = 2
DATA_TIME_SCALE_DENSITY = 1
NUM_JOBS = 4

model_name = "gendered_beta_model"

VAR_UC = 0.25
VAR_CC = 0.01
VAR_A = 1
VAR_AA = 1
MEAN_O_MALE = np.array([10, 100])
VAR_O_MALE = np.array([5, 10])
MEAN_O_FEMALE = np.array([30, 80])
VAR_O_FEMALE = np.array([10, 20])

SAMPLE_TO_INFER = 8
BURN_IN = 100

train_hyper_parameters = GenderedBetaCoordinationLatentVocalicsTrainingHyperParameters(
    a_vu=1e-6,
    b_vu=1e-6,
    a_va=1e-6,
    b_va=1e-6,
    a_vaa=1e-6,
    b_vaa=1e-6,
    mu_mo_male=0,
    nu_mo_male=1e-6,
    a_vo_male=1e-6,
    b_vo_male=1e-6,
    mu_mo_female=0,
    nu_mo_female=1e-6,
    a_vo_female=1e-6,
    b_vo_female=1e-6,
    vu0=0.01,
    vc0=0.01,
    va0=1,
    vaa0=1,
    mo0_male=MEAN_O_MALE,#np.zeros(NUM_FEATURES),
    mo0_female=MEAN_O_FEMALE,#np.zeros(NUM_FEATURES),
    vo0_male=VAR_O_MALE,#np.ones(NUM_FEATURES),
    vo0_female=VAR_O_FEMALE,#np.ones(NUM_FEATURES),
    u_mcmc_iter=50,
    c_mcmc_iter=50,
    vu_mcmc_prop=0.001,
    vc_mcmc_prop=0.001
)


def estimate_parameters(model: GenderedBetaCoordinationBlendingLatentVocalics, evidence, burn_in: int, num_jobs: int,
                        logger: Optional[BaseLogger] = BaseLogger()):
    model.fit(evidence, train_hyper_parameters, burn_in=burn_in, seed=0, num_jobs=num_jobs, logger=logger)

    print(f"Estimated var_u / True var_uc = {model.parameters.var_u} / {VAR_UC}")
    print(f"Estimated var_c / True var_cc = {model.parameters.var_c} / {VAR_CC}")
    print(f"Estimated var_a / True var_a = {model.parameters.var_a} / {VAR_A}")
    print(f"Estimated var_aa / True var_aa = {model.parameters.var_aa} / {VAR_AA}")
    print(f"Estimated mu_o_male / True mu_o_male = {model.parameters.mean_o_male} / {MEAN_O_MALE}")
    print(f"Estimated var_o_male / True var_o_male = {model.parameters.var_o_male} / {VAR_O_MALE}")
    print(f"Estimated mu_o_female / True mu_o_female = {model.parameters.mean_o_female} / {MEAN_O_FEMALE}")
    print(f"Estimated var_o_female / True var_o_female = {model.parameters.var_o_female} / {VAR_O_FEMALE}")


# For parallelism to work, the script has to be called in a __main__ section
if __name__ == "__main__":
    model = GenderedBetaCoordinationBlendingLatentVocalics(
        initial_coordination=0.2,
        num_vocalic_features=NUM_FEATURES,
        num_speakers=2
    )

    model.parameters.set_var_u(VAR_UC)
    model.parameters.set_var_c(VAR_CC)
    model.parameters.set_var_a(VAR_A)
    model.parameters.set_var_aa(VAR_AA)
    model.parameters.set_mean_var_male(MEAN_O_MALE, VAR_O_MALE)
    model.parameters.set_mean_var_female(MEAN_O_FEMALE, VAR_O_FEMALE)

    samples = model.sample(NUM_SAMPLES, TIME_STEPS, seed=0, time_scale_density=DATA_TIME_SCALE_DENSITY)

    # Plot latent vocalics and observed vocalics
    # fig, axs = plt.subplots(1, 2, figsize=(15, 8))
    # ts = np.arange(TIME_STEPS)
    #
    # colors = ["tab:blue" if gender == 0 else "tab:pink" for gender in samples.genders[0]]
    # axs[0].plot(ts, samples.latent_vocalics[0].values[0], linewidth=0.1, linestyle="--")
    # axs[0].scatter(ts, samples.latent_vocalics[0].values[0], label="Latent", marker="o", color=colors)
    # axs[0].plot(ts, samples.observed_vocalics[0].values[0] / samples.latent_vocalics[0].values[0], linewidth=0.1, linestyle="--")
    # axs[0].scatter(ts, samples.observed_vocalics[0].values[0] / samples.latent_vocalics[0].values[0], label="Observed / Latent", marker="s", color=colors)
    # axs[0].set_xlabel("Time Step")
    # axs[0].set_ylabel("1st Feature")
    # axs[0].legend()
    #
    # axs[1].plot(ts, samples.latent_vocalics[0].values[1], linewidth=0.1, linestyle="--")
    # axs[1].scatter(ts, samples.latent_vocalics[0].values[1], label="Latent", marker="o", color=colors)
    # axs[1].plot(ts, samples.observed_vocalics[0].values[1] / samples.latent_vocalics[0].values[1], linewidth=0.1, linestyle="--")
    # axs[1].scatter(ts, samples.observed_vocalics[0].values[1] / samples.latent_vocalics[0].values[1], label="Observed / Latent", marker="s", color=colors)
    # axs[1].set_xlabel("Time Step")
    # axs[1].set_ylabel("2nd Feature")
    # axs[1].legend()
    # plt.show()

    full_evidence = BetaCoordinationLatentVocalicsDataset.from_samples(samples)
    #
    # tmp = copy(samples)
    # tmp.coordination = None
    # tmp.latent_vocalics = None
    # evidence_unbounded_coordination_only = BetaCoordinationLatentVocalicsDataset.from_samples(tmp)
    #
    # tmp = copy(samples)
    # tmp.unbounded_coordination = None
    # tmp.latent_vocalics = None
    # evidence_coordination_only = BetaCoordinationLatentVocalicsDataset.from_samples(tmp)
    #
    # tmp = copy(samples)
    # tmp.unbounded_coordination = None
    # tmp.coordination = None
    # evidence_latent_vocalics_only = BetaCoordinationLatentVocalicsDataset.from_samples(tmp)
    #
    # tmp = copy(samples)
    # tmp.unbounded_coordination = None
    # evidence_no_unbounded_coordination = BetaCoordinationLatentVocalicsDataset.from_samples(tmp)
    #
    # tmp = copy(samples)
    # tmp.coordination = None
    # evidence_no_coordination = BetaCoordinationLatentVocalicsDataset.from_samples(tmp)
    #
    tmp = copy(samples)
    tmp.latent_vocalics = None
    evidence_no_latent_vocalics = BetaCoordinationLatentVocalicsDataset.from_samples(tmp)
    #
    tmp = copy(samples)
    tmp.unbounded_coordination = None
    tmp.coordination = None
    tmp.latent_vocalics = None
    partial_evidence = BetaCoordinationLatentVocalicsDataset.from_samples(tmp)
    #
    # Provide complete data to estimate the true model negative-loglikelihood
    model.fit(full_evidence, train_hyper_parameters, burn_in=0, seed=0, num_jobs=1)
    true_nll = model.nll_[-1]

    print(f"True NLL = {true_nll}")

    # Check if we can estimate the parameters from the complete data
    print()
    print("Parameter estimation with full evidence")
    model.reset_parameters()
    estimate_parameters(model=model, evidence=full_evidence, burn_in=1, num_jobs=1)
    #
    # # No Unbounded Coordination
    # print()
    # print("Parameter estimation NO unbounded coordination")
    # tb_logger = TensorBoardLogger(
    #     f"/Users/paulosoares/code/tomcat-coordination/boards/{model_name}/evidence_no_unbounded_coordination")
    # tb_logger.add_info("data_time_scale_density", DATA_TIME_SCALE_DENSITY)
    # model.reset_parameters()
    # estimate_parameters(model=model, evidence=evidence_no_unbounded_coordination, burn_in=BURN_IN, num_jobs=NUM_JOBS,
    #                     logger=tb_logger)
    #
    # # No Coordination
    # print()
    # print("Parameter estimation NO coordination")
    # tb_logger = TensorBoardLogger(
    #     f"/Users/paulosoares/code/tomcat-coordination/boards/{model_name}/evidence_no_coordination")
    # tb_logger.add_info("data_time_scale_density", DATA_TIME_SCALE_DENSITY)
    # model.reset_parameters()
    # estimate_parameters(model=model, evidence=evidence_no_coordination, burn_in=BURN_IN, num_jobs=NUM_JOBS,
    #                     logger=tb_logger)
    #
    # No Latent Vocalics
    # print()
    # print("Parameter estimation NO latent vocalics")
    # tb_logger = TensorBoardLogger(
    #     f"/Users/paulosoares/code/tomcat-coordination/boards/{model_name}/evidence_no_latent_vocalics")
    # tb_logger.add_info("data_time_scale_density", DATA_TIME_SCALE_DENSITY)
    # model.reset_parameters()
    # estimate_parameters(model=model, evidence=evidence_no_latent_vocalics, burn_in=BURN_IN, num_jobs=NUM_JOBS,
    #                     logger=tb_logger)
    #
    # # With Unbounded Coordination only
    # print()
    # print("Parameter estimation with unbounded coordination only")
    # tb_logger = TensorBoardLogger(
    #     f"/Users/paulosoares/code/tomcat-coordination/boards/{model_name}/evidence_with_unbounded_coordination_only")
    # tb_logger.add_info("data_time_scale_density", DATA_TIME_SCALE_DENSITY)
    # model.reset_parameters()
    # estimate_parameters(model=model, evidence=evidence_unbounded_coordination_only, burn_in=BURN_IN, num_jobs=NUM_JOBS,
    #                     logger=tb_logger)
    #
    # # With Coordination only
    # print()
    # print("Parameter estimation with coordination only")
    # tb_logger = TensorBoardLogger(
    #     f"/Users/paulosoares/code/tomcat-coordination/boards/{model_name}/evidence_with_coordination_only")
    # tb_logger.add_info("data_time_scale_density", DATA_TIME_SCALE_DENSITY)
    # model.reset_parameters()
    # estimate_parameters(model=model, evidence=evidence_coordination_only, burn_in=BURN_IN, num_jobs=NUM_JOBS,
    #                     logger=tb_logger)
    #
    # # With Unbounded Latent Vocalics only
    # print()
    # print("Parameter estimation with latent vocalics only")
    # tb_logger = TensorBoardLogger(
    #     f"/Users/paulosoares/code/tomcat-coordination/boards/{model_name}/evidence_with_latent_vocalics_only")
    # tb_logger.add_info("data_time_scale_density", DATA_TIME_SCALE_DENSITY)
    # model.reset_parameters()
    # estimate_parameters(model=model, evidence=evidence_latent_vocalics_only, burn_in=BURN_IN, num_jobs=NUM_JOBS,
    #                     logger=tb_logger)

    # # Check if we can estimate the parameters if we do not observe latent vocalics and coordination
    # print()
    # print("Parameter estimation with partial evidence")
    # tb_logger = TensorBoardLogger(f"/Users/paulosoares/code/tomcat-coordination/boards/{model_name}/partial_evidence")
    # tb_logger.add_info("data_time_scale_density", DATA_TIME_SCALE_DENSITY)
    # model.reset_parameters()
    # estimate_parameters(model=model, evidence=partial_evidence, burn_in=BURN_IN, num_jobs=NUM_JOBS, logger=tb_logger)

    # Check if we can predict coordination over time for the 1st sample
    model.parameters.set_var_u(VAR_UC)
    model.parameters.set_var_c(VAR_CC)
    model.parameters.set_var_a(VAR_A)
    model.parameters.set_var_aa(VAR_AA)
    model.parameters.set_mean_var_male(MEAN_O_MALE, VAR_O_MALE)
    model.parameters.set_mean_var_female(MEAN_O_FEMALE, VAR_O_FEMALE)
    summary = model.predict(evidence=partial_evidence.get_subset([SAMPLE_TO_INFER]), num_particles=30000,
                            seed=0,
                            num_jobs=1)

    # Plot estimated unbounded coordination against the real coordination points
    plt.figure(figsize=(15, 8))
    means = summary[0].unbounded_coordination_mean
    stds = np.sqrt(summary[0].unbounded_coordination_var)
    ts = np.arange(TIME_STEPS)
    plt.plot(ts, means, color="tab:orange", marker="o")
    plt.fill_between(ts, means - stds, means + stds, color="tab:orange", alpha=0.5)
    plt.plot(ts, samples.unbounded_coordination[SAMPLE_TO_INFER], color="tab:blue", marker="o", alpha=0.5)
    plt.title("Unbounded Coordination")
    plt.show()

    # Plot estimated coordination against the real coordination points
    plt.figure(figsize=(15, 8))
    means = summary[0].coordination_mean
    stds = np.sqrt(summary[0].coordination_var)
    ts = np.arange(TIME_STEPS)
    plt.plot(ts, means, color="tab:orange", marker="o")
    plt.fill_between(ts, means - stds, means + stds, color="tab:orange", alpha=0.5)
    plt.plot(ts, samples.coordination[SAMPLE_TO_INFER], color="tab:blue", marker="o", alpha=0.5)
    plt.title("Coordination")
    plt.show()

    plt.figure(figsize=(15, 8))
    means = summary[0].coordination_mean
    stds = np.sqrt(summary[0].coordination_var)
    ts = np.arange(TIME_STEPS)
    for i in range(NUM_FEATURES):
        plt.plot(ts, samples.latent_vocalics[SAMPLE_TO_INFER].values[i], marker="o", alpha=0.5,
                 label=f"Feature {i + 1}")
    plt.title("Latent Vocalics")
    plt.legend()
    plt.show()
