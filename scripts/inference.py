import os
from typing import Any, List

import argparse
import pickle

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from coordination.model.brain_model import BrainModel, BrainSeries
from coordination.model.body_model import BodyModel, BodySeries
from coordination.model.brain_body_model import BrainBodyModel, BrainBodySeries
from coordination.model.vocalic_semantic_model import VocalicSemanticModel, VocalicSemanticSeries
from coordination.model.vocalic_model import VocalicModel, VocalicSeries


def inference(out_dir: str, experiment_ids: List[str], evidence_filepath: str, model_name: str,
              burn_in: int, num_samples: int, num_chains: int, seed: int, num_inference_jobs: int, do_posterior: bool,
              initial_coordination: float, num_subjects: int, brain_channels: List[str], vocalic_features: List[str],
              self_dependent: bool, sd_uc: float, sd_mean_a0_brain: np.ndarray, sd_sd_aa_brain: np.ndarray,
              sd_sd_o_brain: np.ndarray, sd_mean_a0_body: np.ndarray, sd_sd_aa_body: np.ndarray,
              sd_sd_o_body: np.ndarray, a_mixture_weights: np.ndarray, sd_mean_a0_vocalic: np.ndarray,
              sd_sd_aa_vocalic: np.ndarray, sd_sd_o_vocalic: np.ndarray, a_p_semantic_link: float,
              b_p_semantic_link: float):
    evidence_df = pd.read_csv(evidence_filepath, index_col=0)
    evidence_df = evidence_df[evidence_df["experiment_id"].isin(experiment_ids)]

    # Create correct model from the provided name
    if model_name == "brain":
        model = BrainModel(initial_coordination=initial_coordination,
                           num_subjects=num_subjects,
                           brain_channels=brain_channels,
                           self_dependent=self_dependent,
                           sd_uc=sd_uc,
                           sd_mean_a0=sd_mean_a0_brain,
                           sd_sd_aa=sd_sd_aa_brain,
                           sd_sd_o=sd_sd_o_brain,
                           a_mixture_weights=a_mixture_weights)
    elif model_name == "body":
        model = BodyModel(initial_coordination=initial_coordination,
                          num_subjects=num_subjects,
                          self_dependent=self_dependent,
                          sd_uc=sd_uc,
                          sd_mean_a0=sd_mean_a0_body,
                          sd_sd_aa=sd_sd_aa_body,
                          sd_sd_o=sd_sd_o_body,
                          a_mixture_weights=a_mixture_weights)
    elif model_name == "brain_body":
        model = BrainBodyModel(initial_coordination=initial_coordination,
                               num_subjects=num_subjects,
                               brain_channels=brain_channels,
                               self_dependent=self_dependent,
                               sd_uc=sd_uc,
                               sd_mean_a0_brain=sd_mean_a0_brain,
                               sd_sd_aa_brain=sd_sd_aa_brain,
                               sd_sd_o_brain=sd_sd_o_brain,
                               sd_mean_a0_body=sd_mean_a0_body,
                               sd_sd_aa_body=sd_sd_aa_body,
                               sd_sd_o_body=sd_sd_o_body,
                               a_mixture_weights=a_mixture_weights)
    elif model_name == "vocalic_semantic":
        model = VocalicSemanticModel(initial_coordination=initial_coordination,
                                     num_subjects=num_subjects,
                                     vocalic_features=vocalic_features,
                                     self_dependent=self_dependent,
                                     sd_uc=sd_uc,
                                     sd_mean_a0_vocalic=sd_mean_a0_vocalic,
                                     sd_sd_aa_vocalic=sd_sd_aa_vocalic,
                                     sd_sd_o_vocalic=sd_sd_o_vocalic,
                                     a_p_semantic_link=a_p_semantic_link,
                                     b_p_semantic_link=b_p_semantic_link)
    elif model_name == "vocalic":
        model = VocalicModel(initial_coordination=initial_coordination,
                             num_subjects=num_subjects,
                             vocalic_features=vocalic_features,
                             self_dependent=self_dependent,
                             sd_uc=sd_uc,
                             sd_mean_a0_vocalic=sd_mean_a0_vocalic,
                             sd_sd_aa_vocalic=sd_sd_aa_vocalic,
                             sd_sd_o_vocalic=sd_sd_o_vocalic)
    else:
        raise Exception(f"Invalid model {model_name}.")

    for experiment_id in experiment_ids:
        # Create evidence object from a data frame
        evidence = None
        if model_name == "brain":
            evidence = BrainSeries.from_data_frame(experiment_id, evidence_df, brain_channels)
        elif model_name == "body":
            evidence = BodySeries.from_data_frame(experiment_id, evidence_df)
        elif model_name == "brain_body":
            evidence = BrainBodySeries.from_data_frame(experiment_id, evidence_df, brain_channels)
        elif model_name == "vocalic_semantic":
            evidence = VocalicSemanticSeries.from_data_frame(experiment_id, evidence_df, vocalic_features)
        elif model_name == "vocalic":
            evidence = VocalicSeries.from_data_frame(experiment_id, evidence_df, vocalic_features)

        results_dir = f"{out_dir}/{evidence.uuid}"

        _, idata = model.prior_predictive(evidence=evidence, num_samples=num_samples, seed=seed)
        save_predictive_prior_plots(f"{results_dir}/plots/predictive_prior", idata, evidence, model_name)

        if do_posterior:
            pymc_model, idata_posterior = model.fit(evidence=evidence,
                                                    burn_in=burn_in,
                                                    num_samples=num_samples,
                                                    num_chains=num_chains,
                                                    seed=seed,
                                                    num_jobs=num_inference_jobs)
            idata.extend(idata_posterior)

            # Save pymc_model used for posterior inference
            with open(f"{results_dir}/pymc_model_posterior.pkl", "wb") as f:
                pickle.dump(pymc_model, f)

            save_parameters_plot(f"{results_dir}/plots/posterior", idata, model_name)
            save_coordination_plots(f"{results_dir}/plots/posterior", idata, model_name)

        # Save inference data
        with open(f"{results_dir}/inference_data.pkl", "wb") as f:
            pickle.dump(idata, f)


def save_predictive_prior_plots(out_dir: str, idata: az.InferenceData, single_evidence_series: Any, model: Any):
    if isinstance(model, BrainModel) or isinstance(model, BrainBodyModel):
        brain_posterior_samples = idata.prior_predictive[model.obs_brain_variable_name].sel(chain=0)

        for i, subject in enumerate(brain_posterior_samples.coords["subject"]):
            plot_dir = f"{out_dir}/{subject}"
            os.makedirs(plot_dir, exist_ok=True)

            for j, brain_channel in enumerate(brain_posterior_samples.coords["brain_channel"]):
                prior_samples = brain_posterior_samples.sel(subject=subject, brain_channel=brain_channel)

                T = prior_samples.coordination.sizes["brain_time"]
                N = prior_samples.coordination.sizes["draw"]

                fig = plt.figure(figsize=(15, 8))
                plt.plot(np.arange(T)[:, None].repeat(N, axis=1), prior_samples.T, color="tab:blue", alpha=0.3)
                plt.plot(np.arange(T), single_evidence_series.obs_brain[i, j], color="tab:pink", alpha=1, marker="o")
                plt.title(f"Observed Brain - Subject {subject}, Channel {brain_channel}")
                plt.xlabel(f"Time Step")
                plt.ylabel(f"Avg Hb Total")

                fig.savefig(f"{plot_dir}/{brain_channel}.png", format='png', bbox_inches='tight')

    # TODO - Continue


def save_coordination_plots(out_dir: str, idata: az.InferenceData, model: Any):
    posterior_samples = model.inference_data_to_posterior_samples(idata)

    T = posterior_samples.coordination.sizes["coordination_time"]
    N = posterior_samples.coordination.sizes["draw"]
    stacked_coordination_samples = posterior_samples.coordination.stack(chain_plus_draw=("chain", "draw"))

    fig = plt.figure(figsize=(15, 8))
    plt.plot(np.arange(T)[:, None].repeat(N, axis=1), stacked_coordination_samples.T, color="tab:blue", alpha=0.3)
    plt.plot(np.arange(T), posterior_samples.coordination.mean(dim=["chain", "draw"]), color="tab:pink", alpha=1,
             marker="o")
    plt.title(f"Coordination")
    plt.xlabel(f"Time Step")
    plt.ylabel(f"Coordination")

    fig.savefig(f"{out_dir}/coordination.png", format='png', bbox_inches='tight')


def save_parameters_plot(out_dir: str, idata: az.InferenceData, model: Any):
    fig = az.plot_trace(idata, var_names=[model.parameter_names])
    plt.tight_layout()

    fig.savefig(f"{out_dir}/parameters.png", format='png', bbox_inches='tight')


def str_to_matrix(string: str):
    matrix = []
    for row in string.split(";"):
        matrix.append(row.split(","))

    return np.array(matrix)


def str_to_brain_channels(string: str, evidence_filepath):
    evidence = pd.read_csv(evidence_filepath, index_col=0)

    all_channels = [col for col in evidence.columns if "avg_hb_total" in col]

    if string == "all":
        return all_channels
    else:
        def format_feature_name(name: str):
            return name.strip().lower()

        selected_channels = list(map(format_feature_name, string.split(",")))

        valid_channels = set(all_channels).intersection(set(selected_channels))

        if len(selected_channels) > len(valid_channels):
            invalid_channels = list(set(selected_channels).difference(set(all_channels)))
            raise Exception(f"Invalid brain channels: {invalid_channels}")

        return valid_channels


def str_to_vocalic_features(string: str, evidence_filepath):
    evidence = pd.read_csv(evidence_filepath, index_col=0)

    all_features = [col for col in evidence.columns if "feature" in col]

    if string == "all":
        return all_features
    else:
        def format_feature_name(name: str):
            return name.strip().lower()

        selected_features = list(map(format_feature_name, string.split(",")))

        valid_features = set(all_features).intersection(set(selected_features))

        if len(selected_features) > len(valid_features):
            invalid_channels = list(set(selected_features).difference(set(all_features)))
            raise Exception(f"Invalid brain channels: {invalid_channels}")

        return valid_features


def matrix_to_size(matrix: np.ndarray, num_rows: int, num_cols: int) -> np.ndarray:
    if matrix.shape == (1, 1):
        matrix = matrix.repeat(num_rows, axis=0)
        matrix = matrix.repeat(num_cols, axis=1)
    else:
        raise Exception(f"It's not possible to adjust matrix {matrix} to the dimensions {num_rows} x {num_cols}.")

    return matrix


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Infers coordination and model's parameters per experiment (in sequence) for a series of experiments."
                    "Inference data, pymc model definition, execution parameters and plots are artifacts generated "
                    "by the execution of script. Typically, this script will be called by the parallel_inference.py "
                    "script, which will spawn a series of calls to this script such that inference of different "
                    "experiments can be performed in parallel."
    )

    parser.add_argument("--out_dir", type=str, required=True,
                        help="Directory where artifacts must be saved.")
    parser.add_argument("--experiment_ids", type=str, required=True,
                        help="A list of experiment ids for which we want to perform inference. If more than one "
                             "experiment is provided, inference will be performed sequentially, i.e., for one "
                             "experiment at a time. Experiment ids must be separated commas.")
    parser.add_argument("--evidence_filepath", type=str, required=True,
                        help="Path of the csv file containing the evidence data.")
    parser.add_argument("--model", type=str, required=True,
                        choices=["brain", "body", "brain_body", "vocalic_semantic", "vocalic"],
                        help="Model name.")
    parser.add_argument("--burn_in", type=int, required=False, default=1000,
                        help="Number of samples to discard per chain during posterior inference.")
    parser.add_argument("--num_samples", type=int, required=False, default=1000,
                        help="Number of samples to keep per chain during posterior inference.")
    parser.add_argument("--num_chains", type=int, required=False, default=2,
                        help="Number of chains to use during posterior inference.")
    parser.add_argument("--seed", type=int, required=False, default=0,
                        help="Random seed to use during inference.")
    parser.add_argument("--num_inference_jobs", type=int, required=False, default=4,
                        help="Number of jobs to use per inference process.")
    parser.add_argument("--do_posterior", type=int, required=False, default=1,
                        help="Whether to perform posterior inference or not. Use the value 0 if only prior predictive "
                             "checks are necessary.")
    parser.add_argument("--initial_coordination", type=float, required=False, default=0.01,
                        help="Initial coordination value.")
    parser.add_argument("--num_subjects", type=int, required=False, default=3,
                        help="Number of subjects per experiment.")
    parser.add_argument("--brain_channels", type=str, required=False, default="all",
                        help="Brain channels to use during inference. The channels must be separated by commas.")
    parser.add_argument("--vocalic_features", type=str, required=False, default="all",
                        help="Vocalic features to use during inference. The features must be separated by commas.")
    parser.add_argument("--self_dependent", type=int, required=False, default=1,
                        help="Whether subjects influence themselves in the absense of coordination.")
    parser.add_argument("--sd_uc", type=float, required=False, default=1,
                        help="Standard deviation of the prior distribution of sigma_c")
    parser.add_argument("--sd_mean_a0_brain", type=str, required=False, default="1",
                        help="Standard deviation of the prior distribution of mu_brain_0. If the parameters are "
                             "different per subject and channels, it is possible to pass a matrix "
                             "(num_subjects x num_channels) in MATLAB style where rows are split by semi-colons "
                             "and columns by commas, e.g. 1,2;1,1;2,1  for 3 subjects and 2 channels.")
    parser.add_argument("--sd_sd_aa_brain", type=str, required=False, default="1",
                        help="Standard deviation of the prior distribution of sd_brain. If the parameters are "
                             "different per subject and channels, it is possible to pass a matrix "
                             "(num_subjects x num_channels) in MATLAB style where rows are split by semi-colons "
                             "and columns by commas, e.g. 1,2;1,1;2,1  for 3 subjects and 2 channels.")
    parser.add_argument("--sd_sd_o_brain", type=str, required=False, default="1",
                        help="Standard deviation of the prior distribution of sd_obs_brain. If the parameters are "
                             "different per subject and channels, it is possible to pass a matrix "
                             "(num_subjects x num_channels) in MATLAB style where rows are split by semi-colons "
                             "and columns by commas, e.g. 1,2;1,1;2,1  for 3 subjects and 2 channels.")
    parser.add_argument("--sd_mean_a0_body", type=str, required=False, default="1",
                        help="Standard deviation of the prior distribution of mu_body_0. If the parameters are "
                             "different per subjects, it is possible to pass a matrix "
                             "(num_subjects x 1) in MATLAB style where rows are split by semi-colons "
                             "and columns by commas, e.g. 1,2;1,1;2,1  for 3 subjects and 2 channels.")
    parser.add_argument("--sd_sd_aa_body", type=str, required=False, default="1",
                        help="Standard deviation of the prior distribution of sd_body. If the parameters are "
                             "different per subjects, it is possible to pass a matrix "
                             "(num_subjects x num_channels) in MATLAB style where rows are split by semi-colons "
                             "and columns by commas, e.g. 1,2;1,1;2,1  for 3 subjects and 2 channels.")
    parser.add_argument("--sd_sd_o_body", type=str, required=False, default="1",
                        help="Standard deviation of the prior distribution of sd_obs_body. If the parameters are "
                             "different per subjects, it is possible to pass a matrix "
                             "(num_subjects x num_channels) in MATLAB style where rows are split by semi-colons "
                             "and columns by commas, e.g. 1,2;1,1;2,1  for 3 subjects and 2 channels.")
    parser.add_argument("--a_mixture_weights", type=str, required=False, default="1",
                        help="Parameter `a` of the prior distribution of mixture_weights. If the parameters are "
                             "different per subject and their influencers, it is possible to pass a matrix "
                             "(num_subjects x num_subject - 1) in MATLAB style where rows are split by semi-colons "
                             "and columns by commas, e.g. 1,2;1,1;2,1  for 3 subjects.")
    parser.add_argument("--sd_mean_a0_vocalic", type=str, required=False, default="1",
                        help="Standard deviation of the prior distribution of mu_vocalic_0. If the parameters are "
                             "different per subject and features, it is possible to pass a matrix "
                             "(num_subjects x num_features) in MATLAB style where rows are split by semi-colons "
                             "and columns by commas, e.g. 1,2;1,1;2,1  for 3 subjects and 2 channels.")
    parser.add_argument("--sd_sd_aa_vocalic", type=str, required=False, default="1",
                        help="Standard deviation of the prior distribution of sd_vocalic. If the parameters are "
                             "different per subject and features, it is possible to pass a matrix "
                             "(num_subjects x num_features) in MATLAB style where rows are split by semi-colons "
                             "and columns by commas, e.g. 1,2;1,1;2,1  for 3 subjects and 2 channels.")
    parser.add_argument("--sd_sd_o_vocalic", type=str, required=False, default="1",
                        help="Standard deviation of the prior distribution of sd_obs_vocalic. If the parameters are "
                             "different per subject and features, it is possible to pass a matrix "
                             "(num_subjects x num_features) in MATLAB style where rows are split by semi-colons "
                             "and columns by commas, e.g. 1,2;1,1;2,1  for 3 subjects and 2 channels.")
    parser.add_argument("--a_p_semantic_link", type=float, required=False, default="1",
                        help="Parameter `a` of the prior distribution of p_link")
    parser.add_argument("--b_p_semantic_link", type=float, required=False, default="1",
                        help="Parameter `b` of the prior distribution of p_link")

    args = parser.parse_args()

    arg_brain_channels = str_to_brain_channels(args.brain_channels, args.evidence_filepath)
    arg_vocalic_features = str_to_vocalic_features(args.vocalic_features, args.evidence_filepath)
    arg_sd_mean_a0_brain = matrix_to_size(str_to_matrix(args.sd_mean_a0_brain), args.num_subjects,
                                          len(arg_brain_channels))
    arg_sd_sd_aa_brain = matrix_to_size(str_to_matrix(args.sd_sd_aa_brain), args.num_subjects, len(arg_brain_channels))
    arg_sd_sd_o_brain = matrix_to_size(str_to_matrix(args.sd_sd_o_brain), args.num_subjects, len(arg_brain_channels))
    arg_sd_mean_a0_body = matrix_to_size(str_to_matrix(args.sd_mean_a0_body), args.num_subjects, 1)
    arg_sd_sd_aa_body = matrix_to_size(str_to_matrix(args.sd_sd_aa_body), args.num_subjects, 1)
    arg_sd_sd_o_body = matrix_to_size(str_to_matrix(args.sd_sd_o_body), args.num_subjects, 1)
    arg_a_mixture_weights = matrix_to_size(str_to_matrix(args.a_mixture_weights), args.num_subjects,
                                           args.num_subjects - 1)
    arg_sd_mean_a0_vocalic = matrix_to_size(str_to_matrix(args.sd_mean_a0_vocalic), args.num_subjects,
                                            len(arg_vocalic_features))
    arg_sd_sd_aa_vocalic = matrix_to_size(str_to_matrix(args.sd_sd_aa_vocalic), args.num_subjects,
                                          len(arg_vocalic_features))
    arg_sd_sd_o_vocalic = matrix_to_size(str_to_matrix(args.sd_sd_o_vocalic), args.num_subjects,
                                         len(arg_vocalic_features))

    inference(out_dir=args.out_dir,
              experiment_ids=args.experiment_ids.split(","),
              evidence_filepath=args.evidence_filepath,
              model_name=args.model,
              burn_in=args.burn_in,
              num_samples=args.num_samples,
              num_chains=args.num_chains,
              seed=args.seed,
              num_inference_jobs=args.num_inference_jobs,
              do_posterior=args.do_posterior,
              initial_coordination=args.initial_coordination,
              num_subjects=args.num_subjects,
              brain_channels=arg_brain_channels,
              vocalic_features=arg_vocalic_features,
              self_dependent=args.self_dependent,
              sd_uc=args.sd_uc,
              sd_mean_a0_brain=arg_sd_mean_a0_brain,
              sd_sd_aa_brain=arg_sd_sd_aa_brain,
              sd_sd_o_brain=arg_sd_sd_o_brain,
              sd_mean_a0_body=arg_sd_mean_a0_body,
              sd_sd_aa_body=arg_sd_sd_aa_body,
              sd_sd_o_body=arg_sd_sd_o_body,
              a_mixture_weights=arg_a_mixture_weights,
              sd_mean_a0_vocalic=arg_sd_mean_a0_vocalic,
              sd_sd_aa_vocalic=arg_sd_sd_aa_vocalic,
              sd_sd_o_vocalic=arg_sd_sd_o_vocalic,
              a_p_semantic_link=args.a_p_semantic_link,
              b_p_semantic_link=args.b_p_semantic_link)
