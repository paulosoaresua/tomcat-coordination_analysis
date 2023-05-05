import argparse
import json
from datetime import datetime
import os

import numpy as np
import pandas as pd

from coordination.common.tmux import TMUX

"""
This scripts uses divide the list of experiments in a data set such that they can be processed in parallel.
Since the parameters are not shared across experiments, we can perform inference in an experiment data independenly
of inferences using data from other experiments.

We split the experiments into different processes which are performed in different TMUX windows of the same TMUX 
session. If the number of experiments is bigger than the number of processes they should be split into, some processes
will be responsible for performing inference sequentially in the experiments assigned to them.
"""


def parallel_inference(out_dir: str, evidence_filepath: str, tmux_session_name: str,
                       num_parallel_processes: int, model: str, burn_in: int, num_samples: int, num_chains: int,
                       seed: int, num_inference_jobs: int, do_prior: int, do_posterior: int,
                       initial_coordination: str, num_subjects: int, brain_channels: str, vocalic_features: str,
                       self_dependent: int, sd_mean_uc0: float, sd_sd_uc: float, sd_mean_a0_brain: str,
                       sd_sd_aa_brain: str, sd_sd_o_brain: str, sd_mean_a0_body: str, sd_sd_aa_body: str,
                       sd_sd_o_body: str, a_mixture_weights: str, mean_mean_a0_vocalic: str, sd_mean_a0_vocalic: str,
                       sd_sd_aa_vocalic: str, sd_sd_o_vocalic: str, a_p_semantic_link: float, b_p_semantic_link: float,
                       ignore_bad_channels: int, share_mean_a0_across_subjects: int, share_mean_a0_across_features: int,
                       share_sd_aa_across_subjects: int, share_sd_aa_across_features: int,
                       share_sd_o_across_subjects: int, share_sd_o_across_features: int, vocalic_mode: str, sd_uc: str,
                       mean_a0_brain: str, sd_aa_brain: str, sd_o_brain: str, mean_a0_body: str, sd_aa_body: str,
                       sd_o_body: str, mixture_weights: str, mean_a0_vocalic: str, sd_aa_vocalic: str,
                       sd_o_vocalic: str, p_semantic_link: str, num_layers_f: int,
                       dim_hidden_layer_f: int, activation_function_name_f: str, mean_weights_f: float,
                       sd_weights_f: float, max_lag: int, nuts_init_method: str):
    # Parameters passed to this function relevant for post-analysis.
    execution_params = locals().copy()
    del execution_params["out_dir"]

    execution_time = datetime.now().strftime("%Y.%m.%d--%H.%M.%S")
    results_folder = f"{out_dir}/{model}/{execution_time}"
    os.makedirs(results_folder, exist_ok=True)

    # Save arguments passed to the function

    with open(f"{results_folder}/execution_params.json", "w") as f:
        json.dump(execution_params, f, indent=4)

    # Set environmental variables that will be used by the shell script that calls the sequential inference
    # script
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    evidence_df = pd.read_csv(evidence_filepath, index_col=0)
    experiments = sorted(list(evidence_df["experiment_id"].unique()))
    experiment_blocks = np.array_split(experiments, min(num_parallel_processes, len(experiments)))
    tmux = TMUX(tmux_session_name)
    for i, experiments_per_process in enumerate(experiment_blocks):
        # Start a new process in a tmux window for the experiments to be processed.
        experiment_ids = ",".join(experiments_per_process)
        tmux_window_name = experiment_ids

        # Call the actual inference script
        initial_coordination_arg = f'--initial_coordination={initial_coordination} ' if initial_coordination != "" else ""
        sd_uc_arg = f'--sd_uc={sd_uc} ' if sd_uc != "" else ""
        mean_a0_brain_arg = f'--mean_a0_brain={mean_a0_brain} ' if mean_a0_brain != "" else ""
        sd_aa_brain_arg = f'--sd_aa_brain={sd_aa_brain} ' if sd_aa_brain != "" else ""
        mixture_weights_arg = f'--mixture_weights={mixture_weights} ' if mixture_weights != "" else ""
        sd_o_brain_arg = f'--sd_o_brain={sd_o_brain} ' if sd_o_brain != "" else ""
        mean_a0_body_arg = f'--mean_a0_body={mean_a0_body} ' if mean_a0_body != "" else ""
        sd_aa_body_arg = f'--sd_aa_body={sd_aa_body} ' if sd_aa_body != "" else ""
        sd_o_body_arg = f'--sd_o_body={sd_o_body} ' if sd_o_body != "" else ""
        mean_a0_vocalic_arg = f'--mean_a0_vocalic={mean_a0_vocalic} ' if mean_a0_vocalic != "" else ""
        sd_aa_vocalic_arg = f'--sd_aa_vocalic={sd_aa_vocalic} ' if sd_aa_vocalic != "" else ""
        sd_o_vocalic_arg = f'--sd_o_vocalic={sd_o_vocalic} ' if sd_o_vocalic != "" else ""
        p_semantic_link_arg = f'--p_semantic_link={p_semantic_link} ' if p_semantic_link != "" else ""
        call_python_script_command = f'python3 "{project_dir}/scripts/sequential_experiment_inference.py" ' \
                                     f'--out_dir="{results_folder}" ' \
                                     f'--experiment_ids="{experiment_ids}" ' \
                                     f'--evidence_filepath="{evidence_filepath}" ' \
                                     f'--model="{model}" ' \
                                     f'--burn_in={burn_in} ' \
                                     f'--num_samples={num_samples} ' \
                                     f'--num_chains={num_chains} ' \
                                     f'--seed={seed} ' \
                                     f'--num_inference_jobs={num_inference_jobs} ' \
                                     f'--do_prior={do_prior} ' \
                                     f'--do_posterior={do_posterior} ' \
                                     f'{initial_coordination_arg} ' \
                                     f'{sd_uc_arg} ' \
                                     f'{mean_a0_brain_arg} ' \
                                     f'{sd_aa_brain_arg} ' \
                                     f'{mixture_weights_arg} ' \
                                     f'{sd_o_brain_arg} ' \
                                     f'{mean_a0_body_arg} ' \
                                     f'{sd_aa_body_arg} ' \
                                     f'{sd_o_body_arg} ' \
                                     f'{mean_a0_vocalic_arg} ' \
                                     f'{sd_aa_vocalic_arg} ' \
                                     f'{sd_o_vocalic_arg} ' \
                                     f'{p_semantic_link_arg} ' \
                                     f'--num_subjects={num_subjects} ' \
                                     f'--brain_channels="{brain_channels}" ' \
                                     f'--vocalic_features="{vocalic_features}" ' \
                                     f'--self_dependent={self_dependent} ' \
                                     f'--sd_mean_uc0={sd_mean_uc0} ' \
                                     f'--sd_sd_uc={sd_sd_uc} ' \
                                     f'--sd_mean_a0_brain="{sd_mean_a0_brain}" ' \
                                     f'--sd_sd_aa_brain="{sd_sd_aa_brain}" ' \
                                     f'--sd_sd_o_brain="{sd_sd_o_brain}" ' \
                                     f'--sd_mean_a0_body="{sd_mean_a0_body}" ' \
                                     f'--sd_sd_aa_body="{sd_sd_aa_body}" ' \
                                     f'--sd_sd_o_body="{sd_sd_o_body}" ' \
                                     f'--a_mixture_weights="{a_mixture_weights}" ' \
                                     f'--mean_mean_a0_vocalic="{mean_mean_a0_vocalic}" ' \
                                     f'--sd_mean_a0_vocalic="{sd_mean_a0_vocalic}" ' \
                                     f'--sd_sd_aa_vocalic="{sd_sd_aa_vocalic}" ' \
                                     f'--sd_sd_o_vocalic="{sd_sd_o_vocalic}" ' \
                                     f'--a_p_semantic_link={a_p_semantic_link} ' \
                                     f'--b_p_semantic_link={b_p_semantic_link} ' \
                                     f'--ignore_bad_channels={ignore_bad_channels} ' \
                                     f'--share_mean_a0_across_subjects={share_mean_a0_across_subjects} ' \
                                     f'--share_mean_a0_across_features={share_mean_a0_across_features} ' \
                                     f'--share_sd_aa_across_subjects={share_sd_aa_across_subjects} ' \
                                     f'--share_sd_aa_across_features={share_sd_aa_across_features} ' \
                                     f'--share_sd_o_across_subjects={share_sd_o_across_subjects} ' \
                                     f'--share_sd_o_across_features={share_sd_o_across_features} ' \
                                     f'--vocalic_mode={vocalic_mode} ' \
                                     f'--num_layers_f={num_layers_f} ' \
                                     f'--dim_hidden_layer_f={dim_hidden_layer_f} ' \
                                     f'--activation_function_name_f="{activation_function_name_f}" ' \
                                     f'--mean_weights_f={mean_weights_f} ' \
                                     f'--sd_weights_f={sd_weights_f} ' \
                                     f'--max_lag={max_lag} ' \
                                     f'--nuts_init_method={nuts_init_method}'

        tmux.create_window(tmux_window_name)
        # The user has to make sure tmux initializes conda when a new session or window is created.
        # This is accomplished by copying the conda initialization script (typically saved in the shell .rc file)
        # to their shell .profile file.
        tmux.run_command("conda activate coordination")
        tmux.run_command(call_python_script_command)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Infers coordination and model's parameters for different experiments in parallel. This script will"
                    "create a new tmux session and each inference process will run in a separate tmux window."
    )

    parser.add_argument("--out_dir", type=str, required=True,
                        help="Directory where artifacts must be saved.")
    parser.add_argument("--evidence_filepath", type=str, required=True,
                        help="Path of the csv file containing the evidence data.")
    parser.add_argument("--tmux_session_name", type=str, required=True,
                        help="Name of the tmux session to be created.")
    parser.add_argument("--num_parallel_processes", type=int, required=False, default=1,
                        help="Number of processes to split the experiments into.")
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
    parser.add_argument("--initial_coordination", type=str, required=False, default="",
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
    parser.add_argument("--a_p_semantic_link", type=float, required=False, default=1,
                        help="Parameter `a` of the prior distribution of p_link")
    parser.add_argument("--b_p_semantic_link", type=float, required=False, default=1,
                        help="Parameter `b` of the prior distribution of p_link")
    parser.add_argument("--ignore_bad_channels", type=int, required=False, default=0,
                        help="Whether to remove bad brain channels from the observations.")
    parser.add_argument("--share_mean_a0_across_subjects", type=int, required=False, default=0,
                        help="Whether to fit one mean_a0 for all subjects.")
    parser.add_argument("--share_mean_a0_across_features", type=int, required=False, default=0,
                        help="Whether to fit one mean_a0 for all features.")
    parser.add_argument("--share_sd_aa_across_subjects", type=int, required=False, default=0,
                        help="Whether to fit one sd_aa for all subjects.")
    parser.add_argument("--share_sd_aa_across_features", type=int, required=False, default=0,
                        help="Whether to fit one sd_aa for all features.")
    parser.add_argument("--share_sd_o_across_subjects", type=int, required=False, default=0,
                        help="Whether to fit one sd_o for all subjects.")
    parser.add_argument("--share_sd_o_across_features", type=int, required=False, default=0,
                        help="Whether to fit one sd_o for all features.")
    parser.add_argument("--vocalic_mode", type=str, required=False, default="blending", choices=["blending", "mixture"],
                        help="How coordination controls vocalics from different individuals.")
    parser.add_argument("--sd_uc", type=str, required=False, default="",
                        help="Fixed value for sd_uc. It can be passed in single number, array or matrix form "
                             "depending on how parameters are shared.")
    parser.add_argument("--mean_a0_brain", type=str, required=False, default="",
                        help="Fixed value for mean_a0_brain. It can be passed in single number, array or matrix form "
                             "depending on how parameters are shared.")
    parser.add_argument("--sd_aa_brain", type=str, required=False, default="",
                        help="Fixed value for sd_aa_brain. It can be passed in single number, array or matrix form "
                             "depending on how parameters are shared.")
    parser.add_argument("--sd_o_brain", type=str, required=False, default="",
                        help="Fixed value for sd_o_brain. It can be passed in single number, array or matrix form "
                             "depending on how parameters are shared.")
    parser.add_argument("--mixture_weights", type=str, required=False, default="",
                        help="Fixed value for mixture_weights. It can be passed in single number, array or matrix form "
                             "depending on how parameters are shared.")
    parser.add_argument("--mean_a0_body", type=str, required=False, default="",
                        help="Fixed value for mean_a0_body. It can be passed in single number, array or matrix form "
                             "depending on how parameters are shared.")
    parser.add_argument("--sd_aa_body", type=str, required=False, default="",
                        help="Fixed value for sd_aa_body. It can be passed in single number, array or matrix form "
                             "depending on how parameters are shared.")
    parser.add_argument("--sd_o_body", type=str, required=False, default="",
                        help="Fixed value for sd_o_body. It can be passed in single number, array or matrix form "
                             "depending on how parameters are shared.")
    parser.add_argument("--mean_a0_vocalic", type=str, required=False, default="",
                        help="Fixed value for mean_a0_vocalic. It can be passed in single number, array form "
                             "depending on how parameters are shared.")
    parser.add_argument("--sd_aa_vocalic", type=str, required=False, default="",
                        help="Fixed value for sd_aa_vocalic. It can be passed in single number, array form "
                             "depending on how parameters are shared.")
    parser.add_argument("--sd_o_vocalic", type=str, required=False, default="",
                        help="Fixed value for sd_o_vocalic. It can be passed in single number, array form "
                             "depending on how parameters are shared.")
    parser.add_argument("--p_semantic_link", type=str, required=False, default="",
                        help="Fixed value for p_semantic_link.")
    parser.add_argument("--num_layers_f", type=int, required=False, default=0,
                        help="Number of hidden layers in function f(.) if f is to be fitted.")
    parser.add_argument("--dim_hidden_layer_f", type=int, required=False, default=0,
                        help="Number of units in the hidden layers of f(.) if f is to be fitted.")
    parser.add_argument("--activation_function_name_f", type=str, required=False, default="linear",
                        help="Activation function for f(.) if f is to be fitted.")
    parser.add_argument("--mean_weights_f", type=float, required=False, default=0,
                        help="Mean of the weights (prior)for fitting f(.).")
    parser.add_argument("--sd_weights_f", type=float, required=False, default=1,
                        help="Standard deviation of the weights (prior) for fitting f(.).")
    parser.add_argument("--max_lag", type=int, required=False, default=0,
                        help="Maximum lag to the vocalic component if lags are to be fitted.")
    parser.add_argument("--nuts_init_method", type=str, required=False, default="jitter+adapt_diag",
                        help="NUTS initialization method.")

    args = parser.parse_args()

    parallel_inference(out_dir=args.out_dir,
                       evidence_filepath=args.evidence_filepath,
                       model=args.model,
                       tmux_session_name=args.tmux_session_name,
                       num_parallel_processes=args.num_parallel_processes,
                       burn_in=args.burn_in,
                       num_samples=args.num_samples,
                       num_chains=args.num_chains,
                       seed=args.seed,
                       num_inference_jobs=args.num_inference_jobs,
                       do_prior=args.do_prior,
                       do_posterior=args.do_posterior,
                       initial_coordination=args.initial_coordination,
                       num_subjects=args.num_subjects,
                       brain_channels=args.brain_channels,
                       vocalic_features=args.vocalic_features,
                       self_dependent=args.self_dependent,
                       sd_mean_uc0=args.sd_mean_uc0,
                       sd_sd_uc=args.sd_sd_uc,
                       sd_mean_a0_brain=args.sd_mean_a0_brain,
                       sd_sd_aa_brain=args.sd_sd_aa_brain,
                       sd_sd_o_brain=args.sd_sd_o_brain,
                       sd_mean_a0_body=args.sd_mean_a0_body,
                       sd_sd_aa_body=args.sd_sd_aa_body,
                       sd_sd_o_body=args.sd_sd_o_body,
                       a_mixture_weights=args.a_mixture_weights,
                       mean_mean_a0_vocalic=args.mean_mean_a0_vocalic,
                       sd_mean_a0_vocalic=args.sd_mean_a0_vocalic,
                       sd_sd_aa_vocalic=args.sd_sd_aa_vocalic,
                       sd_sd_o_vocalic=args.sd_sd_o_vocalic,
                       a_p_semantic_link=args.a_p_semantic_link,
                       b_p_semantic_link=args.b_p_semantic_link,
                       ignore_bad_channels=args.ignore_bad_channels,
                       share_mean_a0_across_subjects=args.share_mean_a0_across_subjects,
                       share_mean_a0_across_features=args.share_mean_a0_across_features,
                       share_sd_aa_across_subjects=args.share_sd_aa_across_subjects,
                       share_sd_aa_across_features=args.share_sd_aa_across_features,
                       share_sd_o_across_subjects=args.share_sd_o_across_subjects,
                       share_sd_o_across_features=args.share_sd_o_across_features,
                       vocalic_mode=args.vocalic_mode,
                       sd_uc=args.sd_uc,
                       mean_a0_brain=args.mean_a0_brain,
                       sd_aa_brain=args.sd_aa_brain,
                       sd_o_brain=args.sd_o_brain,
                       mean_a0_body=args.mean_a0_body,
                       sd_aa_body=args.sd_aa_body,
                       sd_o_body=args.sd_o_body,
                       mixture_weights=args.mixture_weights,
                       mean_a0_vocalic=args.mean_a0_vocalic,
                       sd_aa_vocalic=args.sd_aa_vocalic,
                       sd_o_vocalic=args.sd_o_vocalic,
                       p_semantic_link=args.p_semantic_link,
                       num_layers_f=args.num_layers_f,
                       dim_hidden_layer_f=args.dim_hidden_layer_f,
                       activation_function_name_f=args.activation_function_name_f,
                       mean_weights_f=args.mean_weights_f,
                       sd_weights_f=args.sd_weights_f,
                       max_lag=args.max_lag,
                       nuts_init_method=args.nuts_init_method)
