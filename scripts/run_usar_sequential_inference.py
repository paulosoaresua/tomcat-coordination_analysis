import argparse
import logging
import os
import pickle
import sys
from typing import Any, List, Optional

import arviz as az
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from coordination.common.log import configure_log
from coordination.common.utils import set_random_seed
from coordination.model.coordination_model import CoordinationPosteriorSamples
from coordination.model.vocalic_model import (VOCALIC_FEATURES, VocalicModel,
                                              VocalicSeries)
from coordination.model.vocalic_semantic_model import (VocalicSemanticModel,
                                                       VocalicSemanticSeries)

"""
This script performs inferences in a subset of USAR experiments from a .csv file with vocalic and semantic link data. 
Inferences are performed sequentially, i.e., experiment by experiment until all experiments are covered. 
"""

# PyMC 5.0.2 prints some warnings when we use GaussianRandomWalk. The snippet below silences them.
if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")


def inference(
    out_dir: str,
    experiment_ids: List[str],
    evidence_filepath: str,
    model_name: str,
    burn_in: int,
    num_samples: int,
    num_chains: int,
    seed: int,
    num_inference_jobs: int,
    do_prior: bool,
    do_posterior: bool,
    initial_coordination: Optional[float],
    num_subjects: int,
    vocalic_features: List[str],
    self_dependent: bool,
    sd_mean_uc0: float,
    sd_sd_uc: float,
    mean_mean_a0_vocalic: np.ndarray,
    sd_mean_a0_vocalic: np.ndarray,
    sd_sd_aa_vocalic: np.ndarray,
    sd_sd_o_vocalic: np.ndarray,
    a_p_semantic_link: float,
    b_p_semantic_link: float,
    share_mean_a0_across_subjects: bool,
    share_mean_a0_across_features: bool,
    share_sd_aa_across_subjects: bool,
    share_sd_aa_across_features: bool,
    share_sd_o_across_subjects: bool,
    share_sd_o_across_features: bool,
    sd_uc: np.ndarray,
    mean_a0_vocalic: Optional[np.ndarray],
    sd_aa_vocalic: Optional[np.ndarray],
    sd_o_vocalic: Optional[np.ndarray],
    p_semantic_link: Optional[np.ndarray],
    nuts_init_method: str,
    target_accept: float = 0.8,
):
    if not do_prior and not do_posterior:
        raise Exception(
            "No inference to be performed. Choose either prior, posterior or both by setting the appropriate flags."
        )

    set_random_seed(seed)

    # Dataset in a .csv file with a series of experiments.
    evidence_df = pd.read_csv(evidence_filepath, index_col=0)

    evidence_df = evidence_df[evidence_df["experiment_id"].isin(experiment_ids)]

    # Non-interactive backend to make sure it works in a TMUX session when executed from PyCharm.
    mpl.use("Agg")

    print("")
    print(f"Experiment IDs: {experiment_ids}")

    for experiment_id in experiment_ids:
        results_dir = f"{out_dir}/{experiment_id}"
        os.makedirs(results_dir, exist_ok=True)

        # Write the logs to a file in the results directory
        configure_log(verbose=True, log_filepath=f"{results_dir}/log.txt")
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        print("")
        print(f"Processing {experiment_id}")
        print("")

        row_df = evidence_df[evidence_df["experiment_id"] == experiment_id]

        if model_name == "vocalic":
            evidence = VocalicSeries.from_data_frame(
                evidence_df=row_df, vocalic_features=vocalic_features
            )
        elif model_name == "vocalic_semantic":
            evidence = VocalicSemanticSeries.from_data_frame(
                evidence_df=row_df, vocalic_features=vocalic_features
            )
        else:
            raise Exception(f"Invalid model {model_name}.")

        # Models
        if model_name == "vocalic":
            model = VocalicModel(
                num_subjects=num_subjects,
                vocalic_features=vocalic_features,
                self_dependent=self_dependent,
                sd_mean_uc0=sd_mean_uc0,
                sd_sd_uc=sd_sd_uc,
                mean_mean_a0_vocalic=mean_mean_a0_vocalic,
                sd_mean_a0_vocalic=sd_mean_a0_vocalic,
                sd_sd_aa_vocalic=sd_sd_aa_vocalic,
                sd_sd_o_vocalic=sd_sd_o_vocalic,
                initial_coordination=initial_coordination,
                share_mean_a0_across_subjects=share_mean_a0_across_subjects,
                share_mean_a0_across_features=share_mean_a0_across_features,
                share_sd_aa_across_subjects=share_sd_aa_across_subjects,
                share_sd_aa_across_features=share_sd_aa_across_features,
                share_sd_o_across_subjects=share_sd_o_across_subjects,
                share_sd_o_across_features=share_sd_o_across_features,
            )

            # Fix model's parameters if requested. If None the parameter will be fit along with the other
            # latent variables in the model.
            model.coordination_cpn.parameters.sd_uc.value = sd_uc
            model.latent_vocalic_cpn.parameters.mean_a0.value = mean_a0_vocalic
            model.latent_vocalic_cpn.parameters.sd_aa.value = sd_aa_vocalic
            model.obs_vocalic_cpn.parameters.sd_o.value = sd_o_vocalic

        elif model_name == "vocalic_semantic":
            model = VocalicSemanticModel(
                num_subjects=num_subjects,
                vocalic_features=vocalic_features,
                self_dependent=self_dependent,
                sd_mean_uc0=sd_mean_uc0,
                sd_sd_uc=sd_sd_uc,
                mean_mean_a0_vocalic=mean_mean_a0_vocalic,
                sd_mean_a0_vocalic=sd_mean_a0_vocalic,
                sd_sd_aa_vocalic=sd_sd_aa_vocalic,
                sd_sd_o_vocalic=sd_sd_o_vocalic,
                a_p_semantic_link=a_p_semantic_link,
                b_p_semantic_link=b_p_semantic_link,
                initial_coordination=initial_coordination,
                share_mean_a0_across_subjects=share_mean_a0_across_subjects,
                share_mean_a0_across_features=share_mean_a0_across_features,
                share_sd_aa_across_subjects=share_sd_aa_across_subjects,
                share_sd_aa_across_features=share_sd_aa_across_features,
                share_sd_o_across_subjects=share_sd_o_across_subjects,
                share_sd_o_across_features=share_sd_o_across_features,
            )

            # Fix model's parameters if requested. If None the parameter will be fit along with the other
            # latent variables in the model.
            model.coordination_cpn.parameters.sd_uc.value = sd_uc
            model.latent_vocalic_cpn.parameters.mean_a0.value = mean_a0_vocalic
            model.latent_vocalic_cpn.parameters.sd_aa.value = sd_aa_vocalic
            model.obs_vocalic_cpn.parameters.sd_o.value = sd_o_vocalic
            model.semantic_link_cpn.parameters.p.value = p_semantic_link
        else:
            raise Exception(f"Invalid model {model_name}.")

        # Data transformation to correct biological differences captured in the signals from different participants.
        evidence.normalize_per_subject()

        idata = None
        if do_prior:
            # Do prior predictive checks and save plots to the results folder
            logger.info("Prior Predictive Check")
            _, idata = model.prior_predictive(
                evidence=evidence, num_samples=num_samples, seed=seed
            )
            save_predictive_prior_plots(
                f"{results_dir}/plots/predictive_prior", idata, evidence, model
            )

        if do_posterior:
            logger.info("Model fit")
            _, idata_posterior = model.fit(
                evidence=evidence,
                burn_in=burn_in,
                num_samples=num_samples,
                num_chains=num_chains,
                seed=seed,
                num_jobs=num_inference_jobs,
                init_method=nuts_init_method,
                target_accept=target_accept,
            )

            if idata is None:
                idata = idata_posterior
            else:
                idata.extend(idata_posterior)

            # Do posterior predictive checks and save plots to the results folder
            _, idata_posterior_predictive = model.posterior_predictive(
                evidence=evidence, trace=idata_posterior, seed=seed
            )
            idata.extend(idata_posterior_predictive)

            # Log the percentage of divergences
            num_divergences = float(
                idata.sample_stats.diverging.sum(dim=["chain", "draw"])
            )
            num_total_samples = num_samples * num_chains
            logger.info(
                f"{num_divergences} divergences in {num_total_samples} samples --> {100.0 * num_divergences / num_total_samples}%."
            )

            # Save the plots
            save_parameters_plot(f"{results_dir}/plots/posterior", idata, model)
            save_coordination_plots(
                f"{results_dir}/plots/posterior", idata, evidence, model
            )
            save_convergence_summary(f"{results_dir}", idata)
            save_predictive_posterior_plots(
                f"{results_dir}/plots/predictive_posterior", idata, evidence, model
            )

        # Save inference data
        with open(f"{results_dir}/inference_data.pkl", "wb") as f:
            pickle.dump(idata, f)


def save_predictive_prior_plots(
    out_dir: str, idata: az.InferenceData, single_evidence_series: Any, model: Any
):
    vocalic_evidence = (
        single_evidence_series
        if isinstance(model, VocalicModel)
        else single_evidence_series.vocalic
    )
    _plot_vocalic_predictive_plots(out_dir, idata, vocalic_evidence, model, True)


def save_predictive_posterior_plots(
    out_dir: str, idata: az.InferenceData, single_evidence_series: Any, model: Any
):
    vocalic_evidence = (
        single_evidence_series
        if isinstance(model, VocalicModel)
        else single_evidence_series.vocalic
    )
    _plot_vocalic_predictive_plots(out_dir, idata, vocalic_evidence, model, False)


def _plot_vocalic_predictive_plots(
    out_dir: str,
    idata: az.InferenceData,
    single_evidence_series: VocalicSeries,
    model: Any,
    prior: bool,
):
    if prior:
        vocalic_samples = idata.prior_predictive[model.obs_vocalic_variable_name].sel(
            chain=0
        )
    else:
        vocalic_samples = idata.posterior_predictive[
            model.obs_vocalic_variable_name
        ].sel(chain=0)

    logger = logging.getLogger()
    logger.info("Generating vocalic plots")

    vocalic_features = vocalic_samples.coords["vocalic_feature"].data
    for j, vocalic_feature in tqdm(
        enumerate(vocalic_features),
        total=len(vocalic_features),
        desc="Vocalic Features",
        leave=False,
        position=1,
    ):
        prior_samples = vocalic_samples.sel(vocalic_feature=vocalic_feature)

        T = prior_samples.sizes["vocalic_time"]
        N = prior_samples.sizes["draw"]

        fig = plt.figure(figsize=(15, 8))
        plt.plot(
            np.arange(T)[:, None].repeat(N, axis=1),
            prior_samples.T,
            color="tab:blue",
            alpha=0.3,
        )
        plt.plot(
            np.arange(T),
            single_evidence_series.observation[j],
            color="tab:pink",
            alpha=1,
            marker="o",
            markersize=5,
        )
        plt.title(f"Observed {vocalic_feature}")
        plt.xlabel(f"Time Step")
        plt.ylabel(vocalic_feature)

        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(
            f"{out_dir}/{vocalic_feature}.png", format="png", bbox_inches="tight"
        )
        plt.close(fig)


def save_coordination_plots(
    out_dir: str, idata: az.InferenceData, evidence: Any, model: Any
):
    os.makedirs(out_dir, exist_ok=True)

    posterior_samples = CoordinationPosteriorSamples.from_inference_data(idata)
    fig = plt.figure(figsize=(15, 8))
    posterior_samples.plot(fig.gca(), show_samples=False, line_width=5)

    if isinstance(model, VocalicModel) or isinstance(model, VocalicSemanticModel):
        # Mark points with vocalic
        if isinstance(model, VocalicModel):
            time_points = evidence.time_steps_in_coordination_scale
        else:
            time_points = evidence.vocalic.time_steps_in_coordination_scale
        y = posterior_samples.coordination.mean(dim=["chain", "draw"])[time_points]
        plt.scatter(time_points, y, c="white", alpha=1, marker="o", s=3, zorder=4)

    if isinstance(model, VocalicSemanticModel):
        # Mark points with semantic link
        time_points = evidence.semantic_link_time_steps_in_coordination_scale
        y = posterior_samples.coordination.mean(dim=["chain", "draw"])[time_points]
        plt.scatter(
            time_points, y + 0.05, c="white", alpha=1, marker="*", s=10, zorder=4
        )

    plt.title(f"Coordination")
    plt.xlabel(f"Time Step")
    plt.ylabel(f"Coordination")

    fig.savefig(f"{out_dir}/coordination.png", format="png", bbox_inches="tight")


def save_convergence_summary(out_dir: str, idata: az.InferenceData):
    os.makedirs(out_dir, exist_ok=True)

    header = ["variable", "mean_rhat", "std_rhat"]

    rhat = az.rhat(idata)
    data = []
    for var, values in rhat.data_vars.items():
        entry = [var, values.to_numpy().mean(), values.to_numpy().std()]
        data.append(entry)

    pd.DataFrame(data, columns=header).to_csv(f"{out_dir}/convergence_summary.csv")


def save_parameters_plot(out_dir: str, idata: az.InferenceData, model: Any):
    os.makedirs(out_dir, exist_ok=True)

    sampled_vars = set(idata.posterior.data_vars)
    var_names = sorted(list(set(model.parameter_names).intersection(sampled_vars)))

    if len(var_names) > 0:
        axes = az.plot_trace(idata, var_names=var_names)
        fig = axes.ravel()[0].figure
        plt.tight_layout()

        fig.savefig(f"{out_dir}/parameters.png", format="png", bbox_inches="tight")


def str_to_matrix(string: str):
    matrix = []
    for row in string.split(";"):
        matrix.append(list(map(float, row.split(","))))

    return np.array(matrix)


def matrix_to_size(matrix: np.ndarray, num_rows: int, num_cols: int) -> np.ndarray:
    if matrix.shape == (1, 1):
        matrix = matrix.repeat(num_rows, axis=0)
        matrix = matrix.repeat(num_cols, axis=1)
    else:
        raise Exception(
            f"It's not possible to adjust matrix {matrix} to the dimensions {num_rows} x {num_cols}."
        )

    return matrix


def str_to_array(string: str, size: int):
    array = np.array(list(map(float, string.split(","))))
    if array.shape == (1,):
        array = array.repeat(size)
    elif array.shape != (size,):
        raise Exception(
            f"It's not possible to adjust array {array} to the size ({size},)."
        )

    return array


def str_to_features(string: str, complete_list: List[str]):
    if string == "all":
        return complete_list
    else:

        def format_feature_name(name: str):
            return name.strip().lower()

        selected_features = list(map(format_feature_name, string.split(",")))

        valid_features = set(complete_list).intersection(set(selected_features))

        if len(selected_features) > len(valid_features):
            invalid_features = list(
                set(selected_features).difference(set(complete_list))
            )
            raise Exception(f"Invalid features: {invalid_features}")

        return valid_features


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Runs NUTS inference algorithm per experiment (in sequence) for a series of experiments."
        "Inference data and plots are artifacts generated by the execution of script. Typically, "
        "this script will be called by the run_usar_parallel_inference.py script, which will spawn a series "
        "of calls to this script such that inference of different experiments can be performed in "
        "parallel."
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Directory where artifacts must be saved.",
    )
    parser.add_argument(
        "--experiment_ids",
        type=str,
        required=True,
        help="A list of experiment ids for which we want to perform inference. If more than one "
        "experiment is provided, inference will be performed sequentially, i.e., for one "
        "experiment at a time. Experiment ids must be separated by comma.",
    )
    parser.add_argument(
        "--evidence_filepath",
        type=str,
        required=True,
        help="Path of the .csv file containing the evidence data.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["vocalic", "vocalic_semantic"],
        help="Model name.",
    )
    parser.add_argument(
        "--burn_in",
        type=int,
        required=False,
        default=1000,
        help="Number of samples to discard per chain during posterior inference.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        required=False,
        default=1000,
        help="Number of samples to keep per chain during posterior inference.",
    )
    parser.add_argument(
        "--num_chains",
        type=int,
        required=False,
        default=2,
        help="Number of chains to use during posterior inference.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=False,
        default=0,
        help="Random seed to use during inference.",
    )
    parser.add_argument(
        "--num_inference_jobs",
        type=int,
        required=False,
        default=4,
        help="Number of jobs to use per inference process.",
    )
    parser.add_argument(
        "--do_prior",
        type=int,
        required=False,
        default=1,
        help="Whether to perform prior predictive check. Use the value 0 to deactivate.",
    )
    parser.add_argument(
        "--do_posterior",
        type=int,
        required=False,
        default=1,
        help="Whether to perform posterior inference. Use the value 0 to deactivate.",
    )
    parser.add_argument(
        "--initial_coordination",
        type=float,
        required=False,
        help="Initial coordination value. If not provided, initial coordination will be fit "
        "along the other latent variables in the model",
    )
    parser.add_argument(
        "--num_subjects",
        type=int,
        required=False,
        default=3,
        help="Number of subjects in the experiment.",
    )
    parser.add_argument(
        "--vocalic_features",
        type=str,
        required=False,
        default="all",
        help="Vocalic features to use during inference. The features must be separated by comma.",
    )
    parser.add_argument(
        "--self_dependent",
        type=int,
        required=False,
        default=1,
        help="Whether subjects influence themselves over time.",
    )
    parser.add_argument(
        "--sd_mean_uc0",
        type=float,
        required=False,
        default=5,
        help="Standard deviation of mu_c",
    )
    parser.add_argument(
        "--sd_sd_uc",
        type=float,
        required=False,
        default=1,
        help="Standard deviation of sigma_c",
    )
    parser.add_argument(
        "--mean_mean_a0_vocalic",
        type=str,
        required=False,
        default="0",
        help="Mean of mu_a. If the parameters are different per subject and feature, it is possible to "
        "pass a matrix as a string in MATLAB format, i.e., with semi-colons delimiting rows and "
        "commas delimiting columns. If parameters are different per subject or feature but not "
        "both, pass an array with the values separated by comma. If a single number is passed, "
        "it will be replicated for all subjects and features according to the set parameter "
        "sharing option.",
    )
    parser.add_argument(
        "--sd_mean_a0_vocalic",
        type=str,
        required=False,
        default="1",
        help="Standard deviation of mu_a. If the parameters are different per subject and feature, "
        "it is possible to pass a matrix as a string in MATLAB format, i.e., with semi-colons "
        "delimiting rows and commas delimiting columns. If parameters are different per subject "
        "or feature but not both, pass an array with the values separated by comma. If a single "
        "number is passed, it will be replicated for all subjects and features according to the "
        "set parameter sharing option.",
    )
    parser.add_argument(
        "--sd_sd_aa_vocalic",
        type=str,
        required=False,
        default="1",
        help="Standard deviation of sigma_a. If the parameters are different per subject and feature, "
        "it is possible to pass a matrix as a string in MATLAB format, i.e., with semi-colons "
        "delimiting rows and commas delimiting columns. If parameters are different per subject "
        "or feature but not both, pass an array with the values separated by comma. If a single "
        "number is passed, it will be replicated for all subjects and features according to the "
        "set parameter sharing option.",
    )
    parser.add_argument(
        "--sd_sd_o_vocalic",
        type=str,
        required=False,
        default="1",
        help="Standard deviation of sigma_o. If the parameters are different per subject and feature, "
        "it is possible to pass a matrix as a string in MATLAB format, i.e., with semi-colons "
        "delimiting rows and commas delimiting columns. If parameters are different per subject "
        "or feature but not both, pass an array with the values separated by comma. If a single "
        "number is passed, it will be replicated for all subjects and features according to the "
        "set parameter sharing option.",
    )
    parser.add_argument(
        "--a_p_semantic_link",
        type=float,
        required=False,
        default="1",
        help="Parameter `a` of the prior distribution of p_semantic_link",
    )
    parser.add_argument(
        "--b_p_semantic_link",
        type=float,
        required=False,
        default="1",
        help="Parameter `b` of the prior distribution of p_semantic_link",
    )
    parser.add_argument(
        "--share_mean_a0_across_subjects",
        type=int,
        required=False,
        default=0,
        help="Whether to fit one mu_a for all subjects.",
    )
    parser.add_argument(
        "--share_mean_a0_across_features",
        type=int,
        required=False,
        default=0,
        help="Whether to fit one mu_a for all features.",
    )
    parser.add_argument(
        "--share_sd_aa_across_subjects",
        type=int,
        required=False,
        default=0,
        help="Whether to fit one sigma_a for all subjects.",
    )
    parser.add_argument(
        "--share_sd_aa_across_features",
        type=int,
        required=False,
        default=0,
        help="Whether to fit one sigma_a for all features.",
    )
    parser.add_argument(
        "--share_sd_o_across_subjects",
        type=int,
        required=False,
        default=0,
        help="Whether to fit one sigma_o for all subjects.",
    )
    parser.add_argument(
        "--share_sd_o_across_features",
        type=int,
        required=False,
        default=0,
        help="Whether to fit one sigma_o for all features.",
    )
    parser.add_argument(
        "--sd_uc", type=float, required=False, help="Fixed value for sigma_c."
    )
    parser.add_argument(
        "--mean_a0_vocalic",
        type=str,
        required=False,
        help="Fixed value for mu_a. If the parameters are different per subject and feature, "
        "it is possible to pass a matrix as a string in MATLAB format, i.e., with semi-colons "
        "delimiting rows and commas delimiting columns. If parameters are different per subject "
        "or feature but not both, pass an array with the values separated by comma. If a single "
        "number is passed, it will be replicated for all subjects and features according to the "
        "set parameter sharing option.",
    )
    parser.add_argument(
        "--sd_aa_vocalic",
        type=str,
        required=False,
        help="Fixed value for sigma_a. If the parameters are different per subject and feature, "
        "it is possible to pass a matrix as a string in MATLAB format, i.e., with semi-colons "
        "delimiting rows and commas delimiting columns. If parameters are different per subject "
        "or feature but not both, pass an array with the values separated by comma. If a single "
        "number is passed, it will be replicated for all subjects and features according to the "
        "set parameter sharing option.",
    )
    parser.add_argument(
        "--sd_o_vocalic",
        type=str,
        required=False,
        help="Fixed value for sigma_o. If the parameters are different per subject and feature, "
        "it is possible to pass a matrix as a string in MATLAB format, i.e., with semi-colons "
        "delimiting rows and commas delimiting columns. If parameters are different per subject "
        "or feature but not both, pass an array with the values separated by comma. If a single "
        "number is passed, it will be replicated for all subjects and features according to the "
        "set parameter sharing option.",
    )
    parser.add_argument(
        "--p_semantic_link",
        type=float,
        required=False,
        help="Fixed value for p_semantic_link.",
    )
    parser.add_argument(
        "--nuts_init_method",
        type=str,
        required=False,
        default="jitter+adapt_diag",
        help="NUTS initialization method.",
    )
    parser.add_argument(
        "--target_accept",
        type=float,
        required=False,
        default=0.8,
        help="Target acceptance probability used to control step size and reduce "
        "divergences during inference.",
    )

    args = parser.parse_args()

    arg_sd_uc = None if args.sd_uc is None else np.array([float(args.sd_uc)])

    # Determine parameters dimensions according to the parameter sharing configuration
    arg_vocalic_features = str_to_features(args.vocalic_features, VOCALIC_FEATURES)
    dim_mean_a0_features = (
        1 if bool(args.share_mean_a0_across_features) else len(arg_vocalic_features)
    )
    dim_sd_aa_features = (
        1 if bool(args.share_sd_aa_across_features) else len(arg_vocalic_features)
    )
    dim_sd_o_features = (
        1 if bool(args.share_sd_o_across_features) else len(arg_vocalic_features)
    )

    # Transform arguments to arrays and matrices according to the parameter sharing options.
    # mu_a
    arg_mean_a0_vocalic = None
    if bool(args.share_mean_a0_across_subjects):
        arg_mean_mean_a0_vocalic = str_to_array(
            args.mean_mean_a0_vocalic, dim_mean_a0_features
        )
        arg_sd_mean_a0_vocalic = str_to_array(
            args.sd_mean_a0_vocalic, dim_mean_a0_features
        )

        if args.mean_a0_vocalic is not None:
            arg_mean_a0_vocalic = str_to_array(
                args.mean_a0_vocalic, dim_mean_a0_features
            )
    else:
        arg_mean_mean_a0_vocalic = matrix_to_size(
            str_to_matrix(args.mean_mean_a0_vocalic),
            args.num_subjects,
            dim_mean_a0_features,
        )
        arg_sd_mean_a0_vocalic = matrix_to_size(
            str_to_matrix(args.sd_mean_a0_vocalic),
            args.num_subjects,
            dim_mean_a0_features,
        )

        if args.mean_a0_vocalic is not None:
            arg_mean_a0_vocalic = matrix_to_size(
                str_to_matrix(args.mean_a0_vocalic),
                args.num_subjects,
                dim_mean_a0_features,
            )

    # sigma_a
    arg_sd_aa_vocalic = None
    if bool(args.share_sd_aa_across_subjects):
        arg_sd_sd_aa_vocalic = str_to_array(args.sd_sd_aa_vocalic, dim_sd_aa_features)

        if args.sd_aa_vocalic is not None:
            arg_sd_aa_vocalic = str_to_array(args.sd_aa_vocalic, dim_sd_aa_features)
    else:
        arg_sd_sd_aa_vocalic = matrix_to_size(
            str_to_matrix(args.sd_sd_aa_vocalic), args.num_subjects, dim_sd_aa_features
        )

        if args.sd_aa_vocalic is not None:
            arg_sd_aa_vocalic = matrix_to_size(
                str_to_matrix(args.sd_aa_vocalic), args.num_subjects, dim_sd_aa_features
            )

    # sigma_o
    arg_sd_o_vocalic = None
    if bool(args.share_sd_o_across_subjects):
        arg_sd_sd_o_vocalic = str_to_array(args.sd_sd_o_vocalic, dim_sd_o_features)

        if args.sd_o_vocalic is not None:
            arg_sd_o_vocalic = str_to_array(args.sd_o_vocalic, dim_sd_o_features)

    else:
        arg_sd_sd_o_vocalic = matrix_to_size(
            str_to_matrix(args.sd_sd_o_vocalic), args.num_subjects, dim_sd_o_features
        )

        if args.sd_o_vocalic is not None:
            arg_sd_o_vocalic = matrix_to_size(
                str_to_matrix(args.sd_o_vocalic), args.num_subjects, dim_sd_o_features
            )

    arg_p_semantic_link = (
        None if args.p_semantic_link is None else np.array([args.p_semantic_link])
    )

    inference(
        out_dir=args.out_dir,
        experiment_ids=args.experiment_ids.split(","),
        evidence_filepath=args.evidence_filepath,
        model_name=args.model,
        burn_in=args.burn_in,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
        seed=args.seed,
        num_inference_jobs=args.num_inference_jobs,
        do_prior=bool(args.do_prior),
        do_posterior=bool(args.do_posterior),
        initial_coordination=args.initial_coordination,
        num_subjects=args.num_subjects,
        vocalic_features=arg_vocalic_features,
        self_dependent=args.self_dependent,
        sd_mean_uc0=args.sd_mean_uc0,
        sd_sd_uc=args.sd_sd_uc,
        mean_mean_a0_vocalic=arg_mean_mean_a0_vocalic,
        sd_mean_a0_vocalic=arg_sd_mean_a0_vocalic,
        sd_sd_aa_vocalic=arg_sd_sd_aa_vocalic,
        sd_sd_o_vocalic=arg_sd_sd_o_vocalic,
        a_p_semantic_link=args.a_p_semantic_link,
        b_p_semantic_link=args.b_p_semantic_link,
        share_mean_a0_across_subjects=bool(args.share_mean_a0_across_subjects),
        share_mean_a0_across_features=bool(args.share_mean_a0_across_features),
        share_sd_aa_across_subjects=bool(args.share_sd_aa_across_subjects),
        share_sd_aa_across_features=bool(args.share_sd_aa_across_features),
        share_sd_o_across_subjects=bool(args.share_sd_o_across_subjects),
        share_sd_o_across_features=bool(args.share_sd_o_across_features),
        sd_uc=arg_sd_uc,
        mean_a0_vocalic=arg_mean_a0_vocalic,
        sd_aa_vocalic=arg_sd_aa_vocalic,
        sd_o_vocalic=arg_sd_o_vocalic,
        p_semantic_link=arg_p_semantic_link,
        nuts_init_method=args.nuts_init_method,
        target_accept=args.target_accept,
    )
