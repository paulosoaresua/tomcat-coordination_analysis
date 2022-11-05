from typing import Optional

from copy import copy

import matplotlib.pyplot as plt
import numpy as np

from coordination.common.log import BaseLogger, TensorBoardLogger
from coordination.model.beta_coordination_blending_latent_vocalics import BetaCoordinationBlendingLatentVocalics
from coordination.model.beta_coordination_blending_latent_vocalics import BetaCoordinationLatentVocalicsDataset, \
    BetaCoordinationLatentVocalicsDataSeries

# Parameters
TIME_STEPS = 50
NUM_SAMPLES = 100
NUM_FEATURES = 2
DATA_TIME_SCALE_DENSITY = 1
NUM_JOBS = 1

model_name = "beta_model"

VAR_UC = 1
VAR_CC = 0.01
VAR_A = 1
VAR_AA = 0.5
VAR_O = 1


def estimate_parameters(evidence, burn_in: int, num_jobs: int, logger: Optional[BaseLogger] = BaseLogger()):
    model.reset_parameters()
    model.fit(evidence, burn_in=burn_in, seed=0, num_jobs=num_jobs, logger=logger)
    print(f"Estimated var_uc / True var_cc = {model.var_uc} / {VAR_UC}")
    print(f"Estimated var_cc / True var_cc = {model.var_cc} / {VAR_CC}")
    print(f"Estimated var_a / True var_a = {model.var_a} / {VAR_A}")
    print(f"Estimated var_aa / True var_aa = {model.var_aa} / {VAR_AA}")
    print(f"Estimated var_o / True var_o = {model.var_o} / {VAR_O}")


# For parallelism to work, the script has to be called in a __main__ section
if __name__ == "__main__":
    model = BetaCoordinationBlendingLatentVocalics(
        initial_coordination=0.5,
        num_vocalic_features=NUM_FEATURES,
        num_speakers=3,
        a_vcc=1,
        b_vcc=1,
        a_va=1,
        b_va=1,
        a_vaa=1,
        b_vaa=1,
        a_vo=1,
        b_vo=1,
        a_vuc=1,
        b_vuc=1
    )

    model.var_uc = VAR_UC
    model.var_cc = VAR_CC
    model.var_a = VAR_A
    model.var_aa = VAR_AA
    model.var_o = VAR_O

    samples = model.sample(NUM_SAMPLES, TIME_STEPS, seed=0, time_scale_density=DATA_TIME_SCALE_DENSITY)

    full_evidence = BetaCoordinationLatentVocalicsDataset(
        [BetaCoordinationLatentVocalicsDataSeries(f"{i}", samples.observed_vocalics[i],
                                                  samples.unbounded_coordination[i], samples.coordination[i],
                                                  samples.latent_vocalics[i]) for i in range(samples.size)])

    evidence_unbounded_coordination_only = BetaCoordinationLatentVocalicsDataset(
        [BetaCoordinationLatentVocalicsDataSeries(f"{i}", samples.observed_vocalics[i],
                                                  unbounded_coordination=samples.unbounded_coordination[i]) for i in
         range(samples.size)])

    evidence_coordination_only = BetaCoordinationLatentVocalicsDataset(
        [BetaCoordinationLatentVocalicsDataSeries(f"{i}", samples.observed_vocalics[i],
                                                  coordination=samples.coordination[i]) for i in range(samples.size)])

    evidence_latent_vocalics_only = BetaCoordinationLatentVocalicsDataset(
        [BetaCoordinationLatentVocalicsDataSeries(f"{i}", samples.observed_vocalics[i],
                                                  latent_vocalics=samples.latent_vocalics[i]) for i in
         range(samples.size)])

    evidence_no_unbounded_coordination = BetaCoordinationLatentVocalicsDataset(
        [BetaCoordinationLatentVocalicsDataSeries(f"{i}", samples.observed_vocalics[i],
                                                  coordination=samples.coordination[i],
                                                  latent_vocalics=samples.latent_vocalics[i]) for i in
         range(samples.size)])

    evidence_no_coordination = BetaCoordinationLatentVocalicsDataset(
        [BetaCoordinationLatentVocalicsDataSeries(f"{i}", samples.observed_vocalics[i],
                                                  unbounded_coordination=samples.unbounded_coordination[i],
                                                  latent_vocalics=samples.latent_vocalics[i]) for i in
         range(samples.size)])

    evidence_no_latent_vocalics = BetaCoordinationLatentVocalicsDataset(
        [BetaCoordinationLatentVocalicsDataSeries(f"{i}", samples.observed_vocalics[i],
                                                  unbounded_coordination=samples.unbounded_coordination[i],
                                                  coordination=samples.coordination[i]) for i in
         range(samples.size)])

    partial_evidence = BetaCoordinationLatentVocalicsDataset(
        [BetaCoordinationLatentVocalicsDataSeries(f"{i}", samples.observed_vocalics[i]) for i in range(samples.size)])

    # Provide complete data to estimate the true model negative-loglikelihood
    model.fit(full_evidence, burn_in=1, seed=0, num_jobs=1)
    true_nll = model.nll_[-1]

    print(f"True NLL = {true_nll}")

    # Check if we can estimate the parameters from the complete data
    print()
    print("Parameter estimation with full evidence")
    estimate_parameters(evidence=full_evidence, burn_in=1, num_jobs=1)

    # No Unbounded Coordination
    print()
    print("Parameter estimation NO unbounded coordination")
    tb_logger = TensorBoardLogger(
        f"/Users/paulosoares/code/tomcat-coordination/boards/{model_name}/evidence_no_unbounded_coordination")
    tb_logger.add_info("data_time_scale_density", DATA_TIME_SCALE_DENSITY)
    estimate_parameters(evidence=evidence_no_unbounded_coordination, burn_in=100, num_jobs=NUM_JOBS, logger=tb_logger)

    # No Coordination
    print()
    print("Parameter estimation NO coordination")
    tb_logger = TensorBoardLogger(
        f"/Users/paulosoares/code/tomcat-coordination/boards/{model_name}/evidence_no_coordination")
    tb_logger.add_info("data_time_scale_density", DATA_TIME_SCALE_DENSITY)
    estimate_parameters(evidence=evidence_no_coordination, burn_in=100, num_jobs=NUM_JOBS, logger=tb_logger)

    # No Latent Vocalics
    print()
    print("Parameter estimation NO latent vocalics")
    tb_logger = TensorBoardLogger(
        f"/Users/paulosoares/code/tomcat-coordination/boards/{model_name}/evidence_no_latent_vocalics")
    tb_logger.add_info("data_time_scale_density", DATA_TIME_SCALE_DENSITY)
    estimate_parameters(evidence=evidence_no_latent_vocalics, burn_in=100, num_jobs=NUM_JOBS, logger=tb_logger)

    # With Unbounded Coordination only
    print()
    print("Parameter estimation with unbounded coordination only")
    tb_logger = TensorBoardLogger(
        f"/Users/paulosoares/code/tomcat-coordination/boards/{model_name}/evidence_with_unbounded_coordination_only")
    tb_logger.add_info("data_time_scale_density", DATA_TIME_SCALE_DENSITY)
    estimate_parameters(evidence=evidence_unbounded_coordination_only, burn_in=100, num_jobs=NUM_JOBS, logger=tb_logger)

    # With Coordination only
    print()
    print("Parameter estimation with coordination only")
    tb_logger = TensorBoardLogger(
        f"/Users/paulosoares/code/tomcat-coordination/boards/{model_name}/evidence_with_coordination_only")
    tb_logger.add_info("data_time_scale_density", DATA_TIME_SCALE_DENSITY)
    estimate_parameters(evidence=evidence_coordination_only, burn_in=100, num_jobs=NUM_JOBS, logger=tb_logger)

    # With Unbounded Latent Vocalics only
    print()
    print("Parameter estimation with latent vocalics only")
    tb_logger = TensorBoardLogger(
        f"/Users/paulosoares/code/tomcat-coordination/boards/{model_name}/evidence_with_latent_vocalics_only")
    tb_logger.add_info("data_time_scale_density", DATA_TIME_SCALE_DENSITY)
    estimate_parameters(evidence=evidence_latent_vocalics_only, burn_in=100, num_jobs=NUM_JOBS, logger=tb_logger)

    # Check if we can estimate the parameters if we do not observe latent vocalics and coordination
    print()
    print("Parameter estimation with partial evidence")
    tb_logger = TensorBoardLogger(f"/Users/paulosoares/code/tomcat-coordination/boards/{model_name}/partial_evidence")
    tb_logger.add_info("data_time_scale_density", DATA_TIME_SCALE_DENSITY)
    estimate_parameters(evidence=partial_evidence, burn_in=100, num_jobs=NUM_JOBS, logger=tb_logger)

    # Check if we can predict coordination over time for the 1st sample
    model.var_uc = VAR_UC
    model.var_cc = VAR_CC
    model.var_a = VAR_A
    model.var_aa = VAR_AA
    model.var_o = VAR_O
    summary = model.predict(evidence=partial_evidence.get_subset([0]), num_particles=10000, seed=0, num_jobs=1)

    # Plot estimated unbounded coordination against the real coordination points
    plt.figure(figsize=(15, 8))
    means = summary[0].coordination_mean
    stds = np.sqrt(summary[0].coordination_var)
    ts = np.arange(TIME_STEPS)
    plt.plot(ts, means, color="tab:orange", marker="o")
    plt.fill_between(ts, means - stds, means + stds, color="tab:orange", alpha=0.5)
    plt.plot(ts, samples.coordination[0], color="tab:blue", marker="o", alpha=0.5)
    plt.title("Unbounded Coordination")
    plt.show()

    # Plot estimated coordination against the real coordination points
    plt.figure(figsize=(15, 8))
    means = summary[0].coordination_mean
    stds = np.sqrt(summary[0].coordination_var)
    ts = np.arange(TIME_STEPS)
    plt.plot(ts, means, color="tab:orange", marker="o")
    plt.fill_between(ts, means - stds, means + stds, color="tab:orange", alpha=0.5)
    plt.plot(ts, samples.coordination[0], color="tab:blue", marker="o", alpha=0.5)
    plt.title("Coordination")
    plt.show()

    model.fit(full_evidence.get_subset([0]), burn_in=1, seed=0, num_jobs=1)
    true_nll_1st_sample = model.nll_[-1]

    latent_vocalics = copy(samples.latent_vocalics[0])
    latent_vocalics.values = summary[0].latent_vocalics_mean
    estimated_dataset = BetaCoordinationLatentVocalicsDataset([
        BetaCoordinationLatentVocalicsDataSeries("0", samples.observed_vocalics[0],
                                                 summary[0].unbounded_coordination_mean, summary[0].coordination_mean,
                                                 latent_vocalics)])

    model.fit(estimated_dataset, burn_in=1, seed=0, num_jobs=1)
    print(f"True NLL 1st Sample = {true_nll_1st_sample}")
    print(f"Estimated NLL = {model.nll_[-1]}")
