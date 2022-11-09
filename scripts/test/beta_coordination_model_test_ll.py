from typing import Optional

from copy import copy

import matplotlib.pyplot as plt
import numpy as np

from coordination.common.log import BaseLogger, TensorBoardLogger
from coordination.model.beta_coordination_blending_latent_vocalics import BetaCoordinationBlendingLatentVocalics
from coordination.model.beta_coordination_blending_latent_vocalics import BetaCoordinationLatentVocalicsDataset, \
    BetaCoordinationLatentVocalicsDataSeries
from coordination.common.utils import logit
from coordination.component.speech.vocalics_component import VocalicsSparseSeries, SegmentedUtterance
from datetime import datetime
from coordination.common.distribution import beta
from scipy.stats import norm

VAR_UC = 0.01
VAR_CC = 0.0025
VAR_A = 1
VAR_AA = 1
VAR_O = 1


def estimate_parameters(model: BetaCoordinationBlendingLatentVocalics, evidence, burn_in: int, num_jobs: int,
                        logger: Optional[BaseLogger] = BaseLogger()):
    model.fit(evidence, burn_in=burn_in, seed=0, num_jobs=num_jobs, logger=logger)
    print(f"Estimated var_uc / True var_uc = {model.var_uc} / {VAR_UC}")
    print(f"Estimated var_cc / True var_cc = {model.var_cc} / {VAR_CC}")
    print(f"Estimated var_a / True var_a = {model.var_a} / {VAR_A}")
    print(f"Estimated var_aa / True var_aa = {model.var_aa} / {VAR_AA}")
    print(f"Estimated var_o / True var_o = {model.var_o} / {VAR_O}")


# For parallelism to work, the script has to be called in a __main__ section
if __name__ == "__main__":
    model = BetaCoordinationBlendingLatentVocalics(
        initial_coordination=0.3,
        num_vocalic_features=1,
        num_speakers=2,
        a_vcc=1,
        b_vcc=1,
        a_va=1,
        b_va=1,
        a_vaa=1,
        b_vaa=1,
        a_vo=1,
        b_vo=1,
        a_vuc=1,
        b_vuc=1,
        initial_var_unbounded_coordination=1e-4,
        initial_var_coordination=1e-6,
        var_unbounded_coordination_proposal=0.001,
        var_coordination_proposal=1e-8,
        unbounded_coordination_num_mcmc_iterations=50
    )

    utterances = [SegmentedUtterance("A", datetime.now(), datetime.now(), "")] * 3
    utterances[1].subject_id = "B"
    series = BetaCoordinationLatentVocalicsDataSeries(
        uuid="1",
        coordination=np.array([0.25, 0.4, 0.35]),
        unbounded_coordination=np.array([logit(0.3), logit(0.35), logit(0.4)]),
        latent_vocalics=VocalicsSparseSeries(utterances=utterances, previous_from_self=[None, None, 0],
                                             previous_from_other=[None, 0, 1],
                                             values=np.array([1, 0.5, 2]), mask=np.ones(3)),
        observed_vocalics=VocalicsSparseSeries(utterances=utterances, previous_from_self=[None, None, 0],
                                               previous_from_other=[None, 0, 1],
                                               values=np.array([2, 1, 1]), mask=np.ones(3))

    )
    evidence = BetaCoordinationLatentVocalicsDataset([series])

    model.var_uc = VAR_UC
    model.var_cc = VAR_CC
    model.var_a = VAR_A
    model.var_aa = VAR_AA
    model.var_o = VAR_O

    # Provide complete data to estimate the true model negative-loglikelihood
    model.fit(evidence, burn_in=1, seed=0, num_jobs=1)
    true_nll = model.nll_[-1]

    print(f"True NLL = {true_nll}")

    ll = norm(logit(0.3), np.sqrt(VAR_UC)).logpdf(logit(0.35)) + \
         norm(logit(0.35), np.sqrt(VAR_UC)).logpdf(logit(0.4)) + \
         beta(0.3, VAR_CC).logpdf(0.25) + \
         beta(0.35, VAR_CC).logpdf(0.4) + \
         beta(0.4, VAR_CC).logpdf(0.35) + \
         norm(0, 1).logpdf(1) + \
         norm(0.4, 1).logpdf(0.5) + \
         norm(-0.5 * 0.35 + 1, 1).logpdf(2) + \
         norm(1, 1).logpdf(2) + \
         norm(0.5, 1).logpdf(1) + \
         norm(2, 1).logpdf(1)

    print(-ll)
    # Check if we can estimate the parameters from the complete data
    # print()
    # print("Parameter estimation with full evidence")
    # model.reset_parameters()
    # estimate_parameters(model=model, evidence=full_evidence, burn_in=1, num_jobs=1)

    # # No Unbounded Coordination
    # print()
    # print("Parameter estimation NO unbounded coordination")
    # tb_logger = TensorBoardLogger(
    #     f"/Users/paulosoares/code/tomcat-coordination/boards/{model_name}/evidence_no_unbounded_coordination")
    # tb_logger.add_info("data_time_scale_density", DATA_TIME_SCALE_DENSITY)
    # model.reset_parameters()
    # estimate_parameters(model=model, evidence=evidence_no_unbounded_coordination, burn_in=100, num_jobs=NUM_JOBS,
    #                     logger=tb_logger)
    #
    # # No Coordination
    # print()
    # print("Parameter estimation NO coordination")
    # tb_logger = TensorBoardLogger(
    #     f"/Users/paulosoares/code/tomcat-coordination/boards/{model_name}/evidence_no_coordination")
    # tb_logger.add_info("data_time_scale_density", DATA_TIME_SCALE_DENSITY)
    # model.reset_parameters()
    # estimate_parameters(model=model, evidence=evidence_no_coordination, burn_in=100, num_jobs=NUM_JOBS,
    #                     logger=tb_logger)
    #
    # # No Latent Vocalics
    # print()
    # print("Parameter estimation NO latent vocalics")
    # tb_logger = TensorBoardLogger(
    #     f"/Users/paulosoares/code/tomcat-coordination/boards/{model_name}/evidence_no_latent_vocalics")
    # tb_logger.add_info("data_time_scale_density", DATA_TIME_SCALE_DENSITY)
    # model.reset_parameters()
    # estimate_parameters(model=model, evidence=evidence_no_latent_vocalics, burn_in=100, num_jobs=NUM_JOBS,
    #                     logger=tb_logger)
    #
    # # With Unbounded Coordination only
    # print()
    # print("Parameter estimation with unbounded coordination only")
    # tb_logger = TensorBoardLogger(
    #     f"/Users/paulosoares/code/tomcat-coordination/boards/{model_name}/evidence_with_unbounded_coordination_only")
    # tb_logger.add_info("data_time_scale_density", DATA_TIME_SCALE_DENSITY)
    # model.reset_parameters()
    # estimate_parameters(model=model, evidence=evidence_unbounded_coordination_only, burn_in=100, num_jobs=NUM_JOBS,
    #                     logger=tb_logger)
    #
    # # With Coordination only
    # print()
    # print("Parameter estimation with coordination only")
    # tb_logger = TensorBoardLogger(
    #     f"/Users/paulosoares/code/tomcat-coordination/boards/{model_name}/evidence_with_coordination_only")
    # tb_logger.add_info("data_time_scale_density", DATA_TIME_SCALE_DENSITY)
    # model.reset_parameters()
    # estimate_parameters(model=model, evidence=evidence_coordination_only, burn_in=100, num_jobs=NUM_JOBS,
    #                     logger=tb_logger)
    #
    # With Unbounded Latent Vocalics only
    # print()
    # print("Parameter estimation with latent vocalics only")
    # tb_logger = TensorBoardLogger(
    #     f"/Users/paulosoares/code/tomcat-coordination/boards/{model_name}/evidence_with_latent_vocalics_only")
    # tb_logger.add_info("data_time_scale_density", DATA_TIME_SCALE_DENSITY)
    # model.reset_parameters()
    model.var_uc = VAR_UC
    model.var_cc = VAR_CC
    model.var_a = VAR_A
    model.var_aa = VAR_AA
    model.var_o = VAR_O
    evidence.unbounded_coordination = None
    evidence.coordination = None
    estimate_parameters(model=model, evidence=evidence, burn_in=100, num_jobs=1)
    #
    # # Check if we can estimate the parameters if we do not observe latent vocalics and coordination
    # print()
    # print("Parameter estimation with partial evidence")
    # tb_logger = TensorBoardLogger(f"/Users/paulosoares/code/tomcat-coordination/boards/{model_name}/partial_evidence")
    # tb_logger.add_info("data_time_scale_density", DATA_TIME_SCALE_DENSITY)
    # model.reset_parameters()
    # estimate_parameters(model=model, evidence=partial_evidence, burn_in=100, num_jobs=NUM_JOBS, logger=tb_logger)

    # # Check if we can predict coordination over time for the 1st sample
    # model.var_uc = VAR_UC
    # model.var_cc = VAR_CC
    # model.var_a = VAR_A
    # model.var_aa = VAR_AA
    # model.var_o = VAR_O
    # summary = model.predict(evidence=partial_evidence.get_subset([SAMPLE_TO_INFER]), num_particles=10000,
    #                         seed=0,
    #                         num_jobs=1)
    #
    # # Plot estimated unbounded coordination against the real coordination points
    # plt.figure(figsize=(15, 8))
    # means = summary[0].unbounded_coordination_mean
    # stds = np.sqrt(summary[0].unbounded_coordination_var)
    # ts = np.arange(TIME_STEPS)
    # plt.plot(ts, means, color="tab:orange", marker="o")
    # plt.fill_between(ts, means - stds, means + stds, color="tab:orange", alpha=0.5)
    # plt.plot(ts, samples.unbounded_coordination[SAMPLE_TO_INFER], color="tab:blue", marker="o", alpha=0.5)
    # plt.title("Unbounded Coordination")
    # plt.show()
    #
    # # Plot estimated coordination against the real coordination points
    # plt.figure(figsize=(15, 8))
    # means = summary[0].coordination_mean
    # stds = np.sqrt(summary[0].coordination_var)
    # ts = np.arange(TIME_STEPS)
    # plt.plot(ts, means, color="tab:orange", marker="o")
    # plt.fill_between(ts, means - stds, means + stds, color="tab:orange", alpha=0.5)
    # plt.plot(ts, samples.coordination[SAMPLE_TO_INFER], color="tab:blue", marker="o", alpha=0.5)
    # plt.title("Coordination")
    # plt.show()
    #
    # plt.figure(figsize=(15, 8))
    # means = summary[0].coordination_mean
    # stds = np.sqrt(summary[0].coordination_var)
    # ts = np.arange(TIME_STEPS)
    # for i in range(NUM_FEATURES):
    #     plt.plot(ts, samples.latent_vocalics[SAMPLE_TO_INFER].values[i], marker="o", alpha=0.5,
    #              label=f"Feature {i + 1}")
    # plt.title("Latent Vocalics")
    # plt.legend()
    # plt.show()
    #
    # model.fit(full_evidence.get_subset([SAMPLE_TO_INFER]), burn_in=1, seed=0, num_jobs=1)
    # true_nll_1st_sample = model.nll_[-1]
    #
    # latent_vocalics = copy(samples.latent_vocalics[SAMPLE_TO_INFER])
    # latent_vocalics.values = summary[0].latent_vocalics_mean
    # estimated_dataset = BetaCoordinationLatentVocalicsDataset([
    #     BetaCoordinationLatentVocalicsDataSeries("0", samples.observed_vocalics[SAMPLE_TO_INFER],
    #                                              summary[0].unbounded_coordination_mean,
    #                                              summary[0].coordination_mean,
    #                                              latent_vocalics)])
    #
    # model.fit(estimated_dataset, burn_in=1, seed=0, num_jobs=1)
    # print(f"True NLL 1st Sample = {true_nll_1st_sample}")
    # print(f"Estimated NLL = {model.nll_[-1]}")
