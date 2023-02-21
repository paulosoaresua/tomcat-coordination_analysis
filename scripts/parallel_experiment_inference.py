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
                       initial_coordination: float, num_subjects: int, brain_channels: str, vocalic_features: str,
                       self_dependent: bool, sd_uc: float, sd_mean_a0_brain: str, sd_sd_aa_brain: str,
                       sd_sd_o_brain: str, sd_mean_a0_body: str, sd_sd_aa_body: str, sd_sd_o_body: str,
                       a_mixture_weights: str, sd_mean_a0_vocalic: str, sd_sd_aa_vocalic: str, sd_sd_o_vocalic: str,
                       a_p_semantic_link: float, b_p_semantic_link: float):

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
    experiments = sorted(list(evidence_df["experiment_id"].unique()))[:3]
    experiment_blocks = np.array_split(experiments, min(num_parallel_processes, len(experiments)))
    tmux = TMUX(tmux_session_name)
    for i, experiments_per_process in enumerate(experiment_blocks):
        # Start a new process in a tmux window for the experiments to be processed.
        experiment_ids = ",".join(experiments_per_process)
        tmux_window_name = experiment_ids

        # Call the actual inference script
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
                                     f'--initial_coordination={initial_coordination} ' \
                                     f'--num_subjects={num_subjects} ' \
                                     f'--brain_channels="{brain_channels}" ' \
                                     f'--vocalic_features="{vocalic_features}" ' \
                                     f'--self_dependent={self_dependent} ' \
                                     f'--sd_uc={sd_uc} ' \
                                     f'--sd_mean_a0_brain="{sd_mean_a0_brain}" ' \
                                     f'--sd_sd_aa_brain="{sd_sd_aa_brain}" ' \
                                     f'--sd_sd_o_brain="{sd_sd_o_brain}" ' \
                                     f'--sd_mean_a0_body="{sd_mean_a0_body}" ' \
                                     f'--sd_sd_aa_body="{sd_sd_aa_body}" ' \
                                     f'--sd_sd_o_body="{sd_sd_o_body}" ' \
                                     f'--a_mixture_weights="{a_mixture_weights}" ' \
                                     f'--sd_mean_a0_vocalic="{sd_mean_a0_vocalic}" ' \
                                     f'--sd_sd_aa_vocalic="{sd_sd_aa_vocalic}" ' \
                                     f'--sd_sd_o_vocalic="{sd_sd_o_vocalic}" ' \
                                     f'--a_p_semantic_link={a_p_semantic_link} ' \
                                     f'--b_p_semantic_link={b_p_semantic_link}'

        tmux.create_window(tmux_window_name)
        tmux.run_command(f"source {project_dir}/.venv/bin/activate")
        tmux.run_command(f"export PYTHONPATH={project_dir}")
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
    parser.add_argument("--initial_coordination", type=float, required=False, default=0.01,
                        help="Initial coordination value.")
    parser.add_argument("--num_subjects", type=int, required=False, default=3,
                        help="Number of subjects in the experiment.")
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
                       sd_uc=args.sd_uc,
                       sd_mean_a0_brain=args.sd_mean_a0_brain,
                       sd_sd_aa_brain=args.sd_sd_aa_brain,
                       sd_sd_o_brain=args.sd_sd_o_brain,
                       sd_mean_a0_body=args.sd_mean_a0_body,
                       sd_sd_aa_body=args.sd_sd_aa_body,
                       sd_sd_o_body=args.sd_sd_o_body,
                       a_mixture_weights=args.a_mixture_weights,
                       sd_mean_a0_vocalic=args.sd_mean_a0_vocalic,
                       sd_sd_aa_vocalic=args.sd_sd_aa_vocalic,
                       sd_sd_o_vocalic=args.sd_sd_o_vocalic,
                       a_p_semantic_link=args.a_p_semantic_link,
                       b_p_semantic_link=args.b_p_semantic_link)
