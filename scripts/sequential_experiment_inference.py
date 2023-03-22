import os
import sys
from typing import Any, List, Optional, Union

import argparse
from ast import literal_eval
import pickle

import arviz as az
import logging
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tqdm import tqdm

from coordination.common.utils import set_random_seed
from coordination.model.brain_model import BrainModel, BrainSeries, BRAIN_CHANNELS
from coordination.model.body_model import BodyModel, BodySeries
from coordination.model.brain_body_model import BrainBodyModel, BrainBodySeries
from coordination.common.log import configure_log
from coordination.model.vocalic_semantic_model import VocalicSemanticModel, VocalicSemanticSeries
from coordination.model.vocalic_model import VocalicModel, VocalicSeries, VOCALIC_FEATURES
from coordination.model.coordination_model import CoordinationPosteriorSamples
from coordination.component.serialized_component import Mode

"""
This scripts performs inferences in a subset of experiments from a dataset. Inferences are performed sequentially. That
is , experiment by experiment until all experiments are covered. 
"""

if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")


def inference(out_dir: str, experiment_ids: List[str], evidence_filepath: str, model_name: str,
              burn_in: int, num_samples: int, num_chains: int, seed: int, num_inference_jobs: int, do_prior: bool,
              do_posterior: bool, initial_coordination: Optional[float], num_subjects: int, brain_channels: List[str],
              vocalic_features: List[str], self_dependent: bool, sd_mean_uc0: float, sd_sd_uc: float,
              sd_mean_a0_brain: np.ndarray, sd_sd_aa_brain: np.ndarray, sd_sd_o_brain: np.ndarray,
              sd_mean_a0_body: np.ndarray, sd_sd_aa_body: np.ndarray, sd_sd_o_body: np.ndarray,
              a_mixture_weights: np.ndarray, mean_mean_a0_vocalic: np.ndarray, sd_mean_a0_vocalic: np.ndarray,
              sd_sd_aa_vocalic: np.ndarray, sd_sd_o_vocalic: np.ndarray, a_p_semantic_link: float,
              b_p_semantic_link: float, ignore_bad_channels: bool, share_params_across_subjects: bool,
              share_params_across_genders: bool, share_params_across_features: bool, vocalic_mode: str):
    if not do_prior and not do_posterior:
        raise Exception(
            "No inference to be performed. Choose either prior, posterior or both by setting the appropriate flags.")

    if share_params_across_subjects and share_params_across_genders:
        raise Exception("Parameters can be shared only across subjects or genders, not both.")

    evidence_df = pd.read_csv(evidence_filepath, index_col=0)
    evidence_df = evidence_df[evidence_df["experiment_id"].isin(experiment_ids)]

    vocalic_mode = Mode.BLENDING if vocalic_mode == "blending" else Mode.MIXTURE

    # Non-interactive backend to make sure it works in a TMUX session when executed from PyCharm.
    mpl.use("Agg")

    print("")
    print(f"Experiment IDs: {experiment_ids}")

    for experiment_id in experiment_ids:
        results_dir = f"{out_dir}/{experiment_id}"
        os.makedirs(results_dir, exist_ok=True)

        configure_log(verbose=True, log_filepath=f"{results_dir}/log.txt")
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        print("")
        logger.info(f"Processing {experiment_id}")

        row_df = evidence_df[evidence_df["experiment_id"] == experiment_id]

        if ignore_bad_channels and "brain" in model_name:
            # Remove bad channels from the list and respective column in the parameter priors.
            bad_channels = set(literal_eval(row_df["bad_channels"].values[0]))
            removed_channels = []

            for i in range(len(brain_channels) - 1, -1, -1):
                if brain_channels[i] in bad_channels:
                    removed_channels.append(brain_channels[i])
                    del brain_channels[i]
                    sd_mean_a0_brain = np.delete(sd_mean_a0_brain, i, axis=1)
                    sd_sd_aa_brain = np.delete(sd_sd_aa_brain, i, axis=1)
                    sd_sd_o_brain = np.delete(sd_sd_o_brain, i, axis=1)

            logger.info(f"{len(removed_channels)} brain channels removed from the list. They are {removed_channels}")

        # Evidence according to the model
        if model_name == "brain":
            # We ignore channels globally instead of in the evidence object so we can adjust the prior
            # parameters and log that information.
            evidence = BrainSeries.from_data_frame(evidence_df=row_df, brain_channels=brain_channels,
                                                   ignore_bad_channels=False)
        elif model_name == "body":
            evidence = BodySeries.from_data_frame(row_df)

        elif model_name == "brain_body":
            evidence = BrainBodySeries.from_data_frame(row_df, brain_channels)

        elif model_name in ["vocalic", "vocalic_semantic"]:
            # The vocalic model samples missing genders uniformly. So we set the seed here for reproducibility.
            set_random_seed(seed)
            if model_name == "vocalic":
                evidence = VocalicSeries.from_data_frame(row_df, vocalic_features)
            else:
                evidence = VocalicSemanticSeries.from_data_frame(row_df, vocalic_features)

            if share_params_across_genders and evidence.num_genders < 2:
                # All subjects have the same gender, so sharing parameters across gender is the same as sharing them
                # across subjects since we won't be able to infer parameters for the missing gender.
                share_params_across_genders = False
                share_params_across_subjects = True
                gender = list(evidence.gender_map.values())[0]
                sd_mean_a0_vocalic = sd_mean_a0_vocalic[gender]
                sd_sd_aa_vocalic = sd_sd_aa_vocalic[gender]
                sd_sd_o_vocalic = sd_sd_o_vocalic[gender]
        else:
            raise Exception(f"Invalid model {model_name}.")

        # Models
        if model_name == "brain":
            model = BrainModel(subjects=evidence.subjects,
                               brain_channels=brain_channels,
                               self_dependent=self_dependent,
                               sd_mean_uc0=sd_mean_uc0,
                               sd_sd_uc=sd_sd_uc,
                               sd_mean_a0=sd_mean_a0_brain,
                               sd_sd_aa=sd_sd_aa_brain,
                               sd_sd_o=sd_sd_o_brain,
                               a_mixture_weights=a_mixture_weights,
                               initial_coordination=initial_coordination,
                               share_params_across_subjects=True)
        elif model_name == "body":
            model = BodyModel(subjects=evidence.subjects,
                              self_dependent=self_dependent,
                              sd_mean_uc0=sd_mean_uc0,
                              sd_sd_uc=sd_sd_uc,
                              sd_mean_a0=sd_mean_a0_body,
                              sd_sd_aa=sd_sd_aa_body,
                              sd_sd_o=sd_sd_o_body,
                              a_mixture_weights=a_mixture_weights,
                              initial_coordination=initial_coordination,
                              share_params_across_subjects=True)

        elif model_name == "brain_body":
            model = BrainBodyModel(subjects=evidence.subjects,
                                   brain_channels=brain_channels,
                                   self_dependent=self_dependent,
                                   sd_mean_uc0=sd_mean_uc0,
                                   sd_sd_uc=sd_sd_uc,
                                   sd_mean_a0_brain=sd_mean_a0_brain,
                                   sd_sd_aa_brain=sd_sd_aa_brain,
                                   sd_sd_o_brain=sd_sd_o_brain,
                                   sd_mean_a0_body=sd_mean_a0_body,
                                   sd_sd_aa_body=sd_sd_aa_body,
                                   sd_sd_o_body=sd_sd_o_body,
                                   a_mixture_weights=a_mixture_weights,
                                   initial_coordination=initial_coordination,
                                   share_params_across_subjects=True)

        elif model_name == "vocalic":
            model = VocalicModel(num_subjects=num_subjects,
                                 vocalic_features=vocalic_features,
                                 self_dependent=self_dependent,
                                 sd_mean_uc0=sd_mean_uc0,
                                 sd_sd_uc=sd_sd_uc,
                                 mean_mean_a0_vocalic=mean_mean_a0_vocalic,
                                 sd_mean_a0_vocalic=sd_mean_a0_vocalic,
                                 sd_sd_aa_vocalic=sd_sd_aa_vocalic,
                                 sd_sd_o_vocalic=sd_sd_o_vocalic,
                                 initial_coordination=initial_coordination,
                                 share_params_across_subjects=share_params_across_subjects,
                                 share_params_across_genders=share_params_across_genders,
                                 share_params_across_features=share_params_across_features,
                                 mode=vocalic_mode)

        elif model_name == "vocalic_semantic":
            model = VocalicSemanticModel(num_subjects=num_subjects,
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
                                         share_params_across_subjects=share_params_across_subjects,
                                         share_params_across_genders=share_params_across_genders,
                                         share_params_across_features=share_params_across_features,
                                         mode=vocalic_mode)
        else:
            raise Exception(f"Invalid model {model_name}.")

        # Data transformation to correct biological differences captured in the signals from different participant.
        evidence.normalize_per_subject()

        idata = None
        if do_prior:
            logger.info("Prior Predictive Check")
            _, idata = model.prior_predictive(evidence=evidence, num_samples=num_samples, seed=seed)
            save_predictive_prior_plots(f"{results_dir}/plots/predictive_prior", idata, evidence, model)

        if do_posterior:
            logger.info("Model fit")
            _, idata_posterior = model.fit(evidence=evidence,
                                           burn_in=burn_in,
                                           num_samples=num_samples,
                                           num_chains=num_chains,
                                           seed=seed,
                                           num_jobs=num_inference_jobs)

            if idata is None:
                idata = idata_posterior
            else:
                idata.extend(idata_posterior)

            _, idata_posterior_predictive = model.posterior_predictive(evidence=evidence, trace=idata_posterior,
                                                                       seed=seed)
            idata.extend(idata_posterior_predictive)

            num_divergences = float(idata.sample_stats.diverging.sum(dim=["chain", "draw"]))
            num_total_samples = num_samples * num_chains
            logger.info(
                f"{num_divergences} divergences in {num_total_samples} samples --> {100.0 * num_divergences / num_total_samples}%.")

            save_parameters_plot(f"{results_dir}/plots/posterior", idata, model)
            save_coordination_plots(f"{results_dir}/plots/posterior", idata, evidence, model)
            save_predictive_posterior_plots(f"{results_dir}/plots/predictive_posterior", idata, evidence, model)

        # Save inference data
        with open(f"{results_dir}/inference_data.pkl", "wb") as f:
            pickle.dump(idata, f)


def save_predictive_prior_plots(out_dir: str, idata: az.InferenceData, single_evidence_series: Any, model: Any):
    if isinstance(model, BrainModel) or isinstance(model, BrainBodyModel):
        _plot_brain_predictive_plots(out_dir, idata, single_evidence_series, model, True)

    if isinstance(model, BodyModel) or isinstance(model, BrainBodyModel):
        _plot_body_predictive_plots(out_dir, idata, single_evidence_series, model, True)

    if isinstance(model, VocalicModel) or isinstance(model, VocalicSemanticModel):
        _plot_vocalic_predictive_plots(out_dir, idata, single_evidence_series, model, True)


def save_predictive_posterior_plots(out_dir: str, idata: az.InferenceData, single_evidence_series: Any, model: Any):
    if isinstance(model, BrainModel) or isinstance(model, BrainBodyModel):
        _plot_brain_predictive_plots(out_dir, idata, single_evidence_series, model, False)

    if isinstance(model, BodyModel) or isinstance(model, BrainBodyModel):
        _plot_body_predictive_plots(out_dir, idata, single_evidence_series, model, False)

    if isinstance(model, VocalicModel) or isinstance(model, VocalicSemanticModel):
        _plot_vocalic_predictive_plots(out_dir, idata, single_evidence_series, model, False)


def _plot_brain_predictive_plots(out_dir: str,
                                 idata: az.InferenceData,
                                 single_evidence_series: Union[BrainSeries, BrainBodySeries],
                                 model: Any,
                                 prior: bool):
    if prior:
        brain_samples = idata.prior_predictive[model.obs_brain_variable_name].sel(chain=0)
    else:
        brain_samples = idata.posterior_predictive[model.obs_brain_variable_name].sel(chain=0)

    logger = logging.getLogger()
    logger.info("Generating brain plots")
    subjects = brain_samples.coords["subject"].data
    for i, subject in tqdm(enumerate(subjects), desc="Subject", total=len(subjects), position=0):
        plot_dir = f"{out_dir}/{subject}"
        os.makedirs(plot_dir, exist_ok=True)

        brain_channels = brain_samples.coords["brain_channel"].data
        for j, brain_channel in tqdm(enumerate(brain_channels),
                                     total=len(brain_channels),
                                     desc="Brain Channel", leave=False, position=1):
            prior_samples = brain_samples.sel(subject=subject, brain_channel=brain_channel)

            T = prior_samples.sizes["brain_time"]
            N = prior_samples.sizes["draw"]

            fig = plt.figure(figsize=(15, 8))
            plt.plot(np.arange(T)[:, None].repeat(N, axis=1), prior_samples.T, color="tab:blue", alpha=0.3)
            plt.plot(np.arange(T), single_evidence_series.observation[i, j], color="tab:pink", alpha=1, marker="o",
                     markersize=5)
            plt.title(f"Observed Brain - Subject {subject}, Channel {brain_channel}")
            plt.xlabel(f"Time Step")
            plt.ylabel(f"Avg Hb Total")

            fig.savefig(f"{plot_dir}/{brain_channel}.png", format='png', bbox_inches='tight')
            plt.close(fig)


def _plot_body_predictive_plots(out_dir: str,
                                idata: az.InferenceData,
                                single_evidence_series: Union[BodySeries, BrainBodySeries],
                                model: Any,
                                prior: bool):
    if prior:
        body_samples = idata.prior_predictive[model.obs_body_variable_name].sel(chain=0)
    else:
        body_samples = idata.posterior_predictive[model.obs_body_variable_name].sel(chain=0)

    logger = logging.getLogger()
    logger.info("Generating body plots")
    subjects = body_samples.coords["subject"].data
    for i, subject in tqdm(enumerate(subjects), desc="Subject", total=len(subjects), position=0):
        plot_dir = f"{out_dir}/{subject}"
        os.makedirs(plot_dir, exist_ok=True)

        prior_samples = body_samples.sel(subject=subject, body_feature="total_energy")

        T = prior_samples.sizes["body_time"]
        N = prior_samples.sizes["draw"]

        fig = plt.figure(figsize=(15, 8))
        plt.plot(np.arange(T)[:, None].repeat(N, axis=1), prior_samples.T, color="tab:blue", alpha=0.3)
        plt.plot(np.arange(T), single_evidence_series.observation[i, 0], color="tab:pink", alpha=1, marker="o",
                 markersize=5)
        plt.title(f"Observed Body - Subject {subject}")
        plt.xlabel(f"Time Step")
        plt.ylabel(f"Total Motion Energy")

        fig.savefig(f"{plot_dir}/body_motion.png", format='png', bbox_inches='tight')
        plt.close(fig)


def _plot_vocalic_predictive_plots(out_dir: str,
                                   idata: az.InferenceData,
                                   single_evidence_series: Union[VocalicSeries, VocalicSemanticSeries],
                                   model: Any,
                                   prior: bool):
    if prior:
        vocalic_samples = idata.prior_predictive[model.obs_vocalic_variable_name].sel(chain=0)
    else:
        vocalic_samples = idata.posterior_predictive[model.obs_vocalic_variable_name].sel(chain=0)

    logger = logging.getLogger()
    logger.info("Generating vocalic plots")

    vocalic_features = vocalic_samples.coords["vocalic_feature"].data
    for j, vocalic_feature in tqdm(enumerate(vocalic_features),
                                   total=len(vocalic_features),
                                   desc="Vocalic Features", leave=False, position=1):
        prior_samples = vocalic_samples.sel(vocalic_feature=vocalic_feature)

        T = prior_samples.sizes["vocalic_time"]
        N = prior_samples.sizes["draw"]

        fig = plt.figure(figsize=(15, 8))
        plt.plot(np.arange(T)[:, None].repeat(N, axis=1), prior_samples.T, color="tab:blue", alpha=0.3)

        if isinstance(model, VocalicSemanticModel):
            plt.plot(np.arange(T), single_evidence_series.vocalic.observation[j], color="tab:pink", alpha=1, marker="o",
                     markersize=5)
        else:
            plt.plot(np.arange(T), single_evidence_series.observation[j], color="tab:pink", alpha=1, marker="o",
                     markersize=5)
        plt.title(f"Observed {vocalic_feature}")
        plt.xlabel(f"Time Step")
        plt.ylabel(vocalic_feature)

        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(f"{out_dir}/{vocalic_feature}.png", format='png', bbox_inches='tight')
        plt.close(fig)


def save_coordination_plots(out_dir: str, idata: az.InferenceData, evidence: Any, model: Any):
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
        plt.scatter(time_points, y + 0.05, c="white", alpha=1, marker="*", s=10, zorder=4)

    plt.title(f"Coordination")
    plt.xlabel(f"Time Step")
    plt.ylabel(f"Coordination")

    fig.savefig(f"{out_dir}/coordination.png", format='png', bbox_inches='tight')


def save_parameters_plot(out_dir: str, idata: az.InferenceData, model: Any):
    os.makedirs(out_dir, exist_ok=True)

    sampled_vars = set(idata.posterior.data_vars)
    var_names = sorted(list(set(model.parameter_names).intersection(sampled_vars)))

    if len(var_names) > 0:
        axes = az.plot_trace(idata, var_names=var_names)
        fig = axes.ravel()[0].figure
        plt.tight_layout()

        fig.savefig(f"{out_dir}/parameters.png", format='png', bbox_inches='tight')


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
        raise Exception(f"It's not possible to adjust matrix {matrix} to the dimensions {num_rows} x {num_cols}.")

    return matrix


def str_to_array(string: str, size: int):
    array = np.array(list(map(float, string.split(","))))
    if array.shape == (1,):
        array = array.repeat(size)
    elif array.shape != (size,):
        raise Exception(f"It's not possible to adjust array {array} to the size ({size},).")

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
            invalid_features = list(set(selected_features).difference(set(complete_list)))
            raise Exception(f"Invalid features: {invalid_features}")

        return valid_features


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
    parser.add_argument("--do_prior", type=int, required=False, default=1,
                        help="Whether to perform prior predictive check or not. Use the value 0 to deactivate.")
    parser.add_argument("--do_posterior", type=int, required=False, default=1,
                        help="Whether to perform posterior inference or not. Use the value 0 to deactivate.")
    parser.add_argument("--initial_coordination", type=float, required=False,
                        help="Initial coordination value.")
    parser.add_argument("--num_subjects", type=int, required=False, default=3,
                        help="Number of subjects in the experiment.")
    parser.add_argument("--brain_channels", type=str, required=False, default="all",
                        help="Brain channels to use during inference. The channels must be separated by commas.")
    parser.add_argument("--vocalic_features", type=str, required=False, default="all",
                        help="Vocalic features to use during inference. The features must be separated by commas.")
    parser.add_argument("--self_dependent", type=int, required=False, default=1,
                        help="Whether subjects influence themselves in the absense of coordination.")
    parser.add_argument("--sd_mean_uc0", type=float, required=False, default=5,
                        help="Standard deviation of the prior distribution of mean_uc0")
    parser.add_argument("--sd_sd_uc", type=float, required=False, default=1,
                        help="Standard deviation of the prior distribution of sd_uc")
    parser.add_argument("--sd_mean_a0_brain", type=str, required=False, default="1",
                        help="Standard deviation of the prior distribution of mu_brain_0. If the parameters are "
                             "different per channel, it is possible to pass an array as a comma-separated list of."
                             "numbers."),
    parser.add_argument("--sd_sd_aa_brain", type=str, required=False, default="1",
                        help="Standard deviation of the prior distribution of sd_brain. If the parameters are "
                             "different per channel, it is possible to pass an array as a comma-separated list of."
                             "numbers."),
    parser.add_argument("--sd_sd_o_brain", type=str, required=False, default="1",
                        help="Standard deviation of the prior distribution of sd_obs_brain. If the parameters are "
                             "different per channel, it is possible to pass an array as a comma-separated list of."
                             "numbers."),
    parser.add_argument("--sd_mean_a0_body", type=str, required=False, default="1",
                        help="Standard deviation of the prior distribution of mu_body_0."),
    parser.add_argument("--sd_sd_aa_body", type=str, required=False, default="1",
                        help="Standard deviation of the prior distribution of sd_body."),
    parser.add_argument("--sd_sd_o_body", type=str, required=False, default="1",
                        help="Standard deviation of the prior distribution of sd_obs_body."),
    parser.add_argument("--a_mixture_weights", type=str, required=False, default="1",
                        help="Parameter `a` of the prior distribution of mixture_weights. If the parameters are "
                             "different per subject and their influencers, it is possible to pass a matrix "
                             "(num_subjects x num_subject - 1) in MATLAB style where rows are split by semi-colons "
                             "and columns by commas, e.g. 1,2;1,1;2,1  for 3 subjects.")
    parser.add_argument("--mean_mean_a0_vocalic", type=str, required=False, default="0",
                        help="Mean of the prior distribution of mu_vocalic_0. If the parameters are "
                             "different per feature, it is possible to pass an array as a comma-separated list of."
                             "numbers."),
    parser.add_argument("--sd_mean_a0_vocalic", type=str, required=False, default="1",
                        help="Standard deviation of the prior distribution of mu_vocalic_0. If the parameters are "
                             "different per feature, it is possible to pass an array as a comma-separated list of."
                             "numbers."),
    parser.add_argument("--sd_sd_aa_vocalic", type=str, required=False, default="1",
                        help="Standard deviation of the prior distribution of sd_vocalic. If the parameters are "
                             "different per feature, it is possible to pass an array as a comma-separated list of."
                             "numbers."),
    parser.add_argument("--sd_sd_o_vocalic", type=str, required=False, default="1",
                        help="Standard deviation of the prior distribution of sd_obs_vocalic. If the parameters are "
                             "different per feature, it is possible to pass an array as a comma-separated list of."
                             "numbers."),
    parser.add_argument("--a_p_semantic_link", type=float, required=False, default="1",
                        help="Parameter `a` of the prior distribution of p_link")
    parser.add_argument("--b_p_semantic_link", type=float, required=False, default="1",
                        help="Parameter `b` of the prior distribution of p_link")
    parser.add_argument("--ignore_bad_channels", type=int, required=False, default=0,
                        help="Whether to remove bad brain channels from the observations.")
    parser.add_argument("--share_params_across_subjects", type=int, required=False, default=0,
                        help="Whether to fit one parameter for all subjects.")
    parser.add_argument("--share_params_across_genders", type=int, required=False, default=0,
                        help="Whether to fit one parameter per subject's gender. If all subjects have the same gender, "
                             "only the parameters of that gender will be estimated.")
    parser.add_argument("--share_params_across_features", type=int, required=False, default=0,
                        help="Whether to fit one parameter per feature.")
    parser.add_argument("--vocalic_mode", type=str, required=False, default="blending", choices=["blending", "mixture"],
                        help="How coordination controls vocalics from different individuals.")

    args = parser.parse_args()

    # Brain parameters
    arg_brain_channels = str_to_features(args.brain_channels, BRAIN_CHANNELS)
    dim = len(arg_brain_channels)
    arg_a_mixture_weights = matrix_to_size(str_to_matrix(args.a_mixture_weights), args.num_subjects,
                                           args.num_subjects - 1)
    if bool(args.share_params_across_subjects):
        arg_sd_mean_a0_brain = str_to_array(args.sd_mean_a0_brain, dim)
        arg_sd_sd_aa_brain = str_to_array(args.sd_sd_aa_brain, dim)
        arg_sd_sd_o_brain = str_to_array(args.sd_sd_o_brain, dim)
    else:
        arg_sd_mean_a0_brain = matrix_to_size(str_to_matrix(args.sd_mean_a0_brain), args.num_subjects, dim)
        arg_sd_sd_aa_brain = matrix_to_size(str_to_matrix(args.sd_sd_aa_brain), args.num_subjects, dim)
        arg_sd_sd_o_brain = matrix_to_size(str_to_matrix(args.sd_sd_o_brain), args.num_subjects, dim)

    # Body parameters
    if bool(args.share_params_across_subjects):
        arg_sd_mean_a0_body = str_to_array(args.sd_mean_a0_body, 1)
        arg_sd_sd_aa_body = str_to_array(args.sd_sd_aa_body, 1)
        arg_sd_sd_o_body = str_to_array(args.sd_sd_o_body, 1)
    else:
        arg_sd_mean_a0_body = matrix_to_size(str_to_matrix(args.sd_mean_a0_body), args.num_subjects, dim)
        arg_sd_sd_aa_body = matrix_to_size(str_to_matrix(args.sd_sd_aa_body), args.num_subjects, dim)
        arg_sd_sd_o_body = matrix_to_size(str_to_matrix(args.sd_sd_o_body), args.num_subjects, dim)

    # Vocalic parameters
    arg_vocalic_features = str_to_features(args.vocalic_features, VOCALIC_FEATURES)
    dim = 1 if bool(args.share_params_across_features) else len(arg_vocalic_features)
    if bool(args.share_params_across_subjects):
        arg_mean_mean_a0_vocalic = str_to_array(args.mean_mean_a0_vocalic, dim)
        arg_sd_mean_a0_vocalic = str_to_array(args.sd_mean_a0_vocalic, dim)
        arg_sd_sd_aa_vocalic = str_to_array(args.sd_sd_aa_vocalic, dim)
        arg_sd_sd_o_vocalic = str_to_array(args.sd_sd_o_vocalic, dim)
    elif bool(args.share_params_across_genders):
        arg_mean_mean_a0_vocalic = matrix_to_size(str_to_matrix(args.mean_mean_a0_vocalic), 2, dim)
        arg_sd_mean_a0_vocalic = matrix_to_size(str_to_matrix(args.sd_mean_a0_vocalic), 2, dim)
        arg_sd_sd_aa_vocalic = matrix_to_size(str_to_matrix(args.sd_sd_aa_vocalic), 2, dim)
        arg_sd_sd_o_vocalic = matrix_to_size(str_to_matrix(args.sd_sd_o_vocalic), 2, dim)
    else:
        arg_mean_mean_a0_vocalic = matrix_to_size(str_to_matrix(args.mean_mean_a0_vocalic), args.num_subjects, dim)
        arg_sd_mean_a0_vocalic = matrix_to_size(str_to_matrix(args.sd_mean_a0_vocalic), args.num_subjects, dim)
        arg_sd_sd_aa_vocalic = matrix_to_size(str_to_matrix(args.sd_sd_aa_vocalic), args.num_subjects, dim)
        arg_sd_sd_o_vocalic = matrix_to_size(str_to_matrix(args.sd_sd_o_vocalic), args.num_subjects, dim)

    inference(out_dir=args.out_dir,
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
              brain_channels=arg_brain_channels,
              vocalic_features=arg_vocalic_features,
              self_dependent=args.self_dependent,
              sd_mean_uc0=args.sd_mean_uc0,
              sd_sd_uc=args.sd_sd_uc,
              sd_mean_a0_brain=arg_sd_mean_a0_brain,
              sd_sd_aa_brain=arg_sd_sd_aa_brain,
              sd_sd_o_brain=arg_sd_sd_o_brain,
              sd_mean_a0_body=arg_sd_mean_a0_body,
              sd_sd_aa_body=arg_sd_sd_aa_body,
              sd_sd_o_body=arg_sd_sd_o_body,
              a_mixture_weights=arg_a_mixture_weights,
              mean_mean_a0_vocalic=arg_mean_mean_a0_vocalic,
              sd_mean_a0_vocalic=arg_sd_mean_a0_vocalic,
              sd_sd_aa_vocalic=arg_sd_sd_aa_vocalic,
              sd_sd_o_vocalic=arg_sd_sd_o_vocalic,
              a_p_semantic_link=args.a_p_semantic_link,
              b_p_semantic_link=args.b_p_semantic_link,
              ignore_bad_channels=bool(args.ignore_bad_channels),
              share_params_across_subjects=bool(args.share_params_across_subjects),
              share_params_across_genders=bool(args.share_params_across_genders),
              share_params_across_features=bool(args.share_params_across_features),
              vocalic_mode=args.vocalic_mode)
