import argparse
import json
from datetime import datetime
import os

import numpy as np
import pandas as pd

from coordination.common.tmux import TMUX

"""
This script divides the list of USAR experiments in a .csv dataset such that they can be processed in parallel.
Since the parameters are not shared across experiments, we can perform inferences in multiple experiments in
parallel if we have computing resources.

We split the experiments into different processes which are performed in different TMUX windows of the same TMUX 
session. If the tmux session does not exist, this script will create one with the name provided. If the number of 
experiments is bigger than the number of processes they should be split into, some processes will be responsible for 
performing inference sequentially in the experiments assigned to them.

Note: 
1. This script requires TMUX to be installed in the machine.
2. This script requires Conda to be installed in the machine and have an environment called 'coordination' with the 
dependencies listed in requirements.txt installed.
"""


def parallel_inference(out_dir: str,
                       evidence_filepath: str,
                       tmux_session_name: str,
                       num_parallel_processes: int,
                       model: str,
                       burn_in: int,
                       num_samples: int,
                       num_chains: int,
                       seed: int,
                       num_inference_jobs: int,
                       do_prior: int,
                       do_posterior: int,
                       initial_coordination: str,
                       num_subjects: int,
                       vocalic_features: str,
                       self_dependent: int,
                       sd_mean_uc0: float,
                       sd_sd_uc: float,
                       mean_mean_a0_vocalic: str,
                       sd_mean_a0_vocalic: str,
                       sd_sd_aa_vocalic: str,
                       sd_sd_o_vocalic: str,
                       a_p_semantic_link: float,
                       b_p_semantic_link: float,
                       share_mean_a0_across_subjects: int,
                       share_mean_a0_across_features: int,
                       share_sd_aa_across_subjects: int,
                       share_sd_aa_across_features: int,
                       share_sd_o_across_subjects: int,
                       share_sd_o_across_features: int,
                       sd_uc: str,
                       mean_a0_vocalic: str,
                       sd_aa_vocalic: str,
                       sd_o_vocalic: str,
                       p_semantic_link: str,
                       nuts_init_method: str,
                       target_accept: float):

    # Parameters passed to this function relevant for post-analysis. We will save them to a file.
    execution_params = locals().copy()
    del execution_params["out_dir"]

    execution_time = datetime.now().strftime("%Y.%m.%d--%H.%M.%S")
    results_folder = f"{out_dir}/{model}/{execution_time}"
    os.makedirs(results_folder, exist_ok=True)

    print("")
    print(f"Inferences will be saved in {results_folder}")
    print("")

    # Save arguments passed to the function
    with open(f"{results_folder}/execution_params.json", "w") as f:
        json.dump(execution_params, f, indent=4)

    # Get absolute path to the project. We will use it to execute run_sequential_inference.py from a tmux window.
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    evidence_df = pd.read_csv(evidence_filepath, index_col=0)
    experiments = sorted(list(evidence_df["experiment_id"].unique()))
    experiment_blocks = np.array_split(experiments, min(num_parallel_processes, len(experiments)))
    tmux = TMUX(tmux_session_name)
    for i, experiments_per_process in enumerate(experiment_blocks):
        # Start a new process in a tmux window for the experiments to be processed.
        experiment_ids = ",".join(experiments_per_process)
        tmux_window_name = experiment_ids

        # Call the actual inference script (run_usar_sequential.inference.py)
        initial_coordination_arg = f'--initial_coordination={initial_coordination} ' if initial_coordination != "" else ""
        sd_uc_arg = f'--sd_uc={sd_uc} ' if sd_uc != "" else ""
        mean_a0_vocalic_arg = f'--mean_a0_vocalic={mean_a0_vocalic} ' if mean_a0_vocalic != "" else ""
        sd_aa_vocalic_arg = f'--sd_aa_vocalic={sd_aa_vocalic} ' if sd_aa_vocalic != "" else ""
        sd_o_vocalic_arg = f'--sd_o_vocalic={sd_o_vocalic} ' if sd_o_vocalic != "" else ""
        p_semantic_link_arg = f'--p_semantic_link={p_semantic_link} ' if p_semantic_link != "" else ""
        call_python_script_command = f'python3 "{project_dir}/scripts/run_usar_sequential_inference.py" ' \
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
                                     f'{mean_a0_vocalic_arg} ' \
                                     f'{sd_aa_vocalic_arg} ' \
                                     f'{sd_o_vocalic_arg} ' \
                                     f'{p_semantic_link_arg} ' \
                                     f'--num_subjects={num_subjects} ' \
                                     f'--vocalic_features="{vocalic_features}" ' \
                                     f'--self_dependent={self_dependent} ' \
                                     f'--sd_mean_uc0={sd_mean_uc0} ' \
                                     f'--sd_sd_uc={sd_sd_uc} ' \
                                     f'--mean_mean_a0_vocalic="{mean_mean_a0_vocalic}" ' \
                                     f'--sd_mean_a0_vocalic="{sd_mean_a0_vocalic}" ' \
                                     f'--sd_sd_aa_vocalic="{sd_sd_aa_vocalic}" ' \
                                     f'--sd_sd_o_vocalic="{sd_sd_o_vocalic}" ' \
                                     f'--a_p_semantic_link={a_p_semantic_link} ' \
                                     f'--b_p_semantic_link={b_p_semantic_link} ' \
                                     f'--share_mean_a0_across_subjects={share_mean_a0_across_subjects} ' \
                                     f'--share_mean_a0_across_features={share_mean_a0_across_features} ' \
                                     f'--share_sd_aa_across_subjects={share_sd_aa_across_subjects} ' \
                                     f'--share_sd_aa_across_features={share_sd_aa_across_features} ' \
                                     f'--share_sd_o_across_subjects={share_sd_o_across_subjects} ' \
                                     f'--share_sd_o_across_features={share_sd_o_across_features} ' \
                                     f'--nuts_init_method={nuts_init_method} ' \
                                     f'--target_accept={target_accept}'

        tmux.create_window(tmux_window_name)

        # Before running this script for the first time, the user has to check if tmux initializes conda when
        # a new session or window is created. This is accomplished by copying the conda initialization script,
        # typically saved in the shell rc file (e.g., .bashrc), to their shell profile file (e.g., .bash_profile).
        tmux.run_command("conda activate coordination")
        tmux.run_command(call_python_script_command)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Infers coordination and model's parameters for different experiments in parallel. This script will"
                    "create a new tmux session and each inference process will run in a separate tmux window."
    )

    parser.add_argument("--tmux_session_name", type=str, required=True,
                        help="Name of the tmux session to be created.")
    parser.add_argument("--num_parallel_processes", type=int, required=False, default=1,
                        help="Number of processes to split the experiments into.")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Directory where artifacts must be saved.")
    parser.add_argument("--evidence_filepath", type=str, required=True,
                        help="Path of the csv file containing the evidence data.")
    # Arguments below will be passes to run_sequential_inference and should match the arguments in that
    # script.
    parser.add_argument("--model", type=str, required=True,
                        choices=["vocalic", "vocalic_semantic"],
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
                        help="Whether to perform prior predictive check. Use the value 0 to deactivate.")
    parser.add_argument("--do_posterior", type=int, required=False, default=1,
                        help="Whether to perform posterior inference. Use the value 0 to deactivate.")
    parser.add_argument("--initial_coordination", type=float, required=False,
                        help="Initial coordination value. If not provided or < 0, initial coordination will be fit "
                             "along the other latent variables in the model")
    parser.add_argument("--num_subjects", type=int, required=False, default=3,
                        help="Number of subjects in the experiment.")
    parser.add_argument("--vocalic_features", type=str, required=False, default="all",
                        help="Vocalic features to use during inference. The features must be separated by comma.")
    parser.add_argument("--self_dependent", type=int, required=False, default=1,
                        help="Whether subjects influence themselves over time.")
    parser.add_argument("--sd_mean_uc0", type=float, required=False, default=5,
                        help="Standard deviation of mu_c")
    parser.add_argument("--sd_sd_uc", type=float, required=False, default=1,
                        help="Standard deviation of sigma_c")
    parser.add_argument("--mean_mean_a0_vocalic", type=str, required=False, default="0",
                        help="Mean of mu_a. If the parameters are different per subject and feature, it is possible to "
                             "pass a matrix as a string in MATLAB format, i.e., with semi-colons delimiting rows and "
                             "commas delimiting columns. If parameters are different per subject or feature but not "
                             "both, pass an array with the values separated by comma. If a single number is passed, "
                             "it will be replicated for all subjects and features according to the set parameter "
                             "sharing option.")
    parser.add_argument("--sd_mean_a0_vocalic", type=str, required=False, default="1",
                        help="Standard deviation of mu_a. If the parameters are different per subject and feature, "
                             "it is possible to pass a matrix as a string in MATLAB format, i.e., with semi-colons "
                             "delimiting rows and commas delimiting columns. If parameters are different per subject "
                             "or feature but not both, pass an array with the values separated by comma. If a single "
                             "number is passed, it will be replicated for all subjects and features according to the "
                             "set parameter sharing option.")
    parser.add_argument("--sd_sd_aa_vocalic", type=str, required=False, default="1",
                        help="Standard deviation of sigma_a. If the parameters are different per subject and feature, "
                             "it is possible to pass a matrix as a string in MATLAB format, i.e., with semi-colons "
                             "delimiting rows and commas delimiting columns. If parameters are different per subject "
                             "or feature but not both, pass an array with the values separated by comma. If a single "
                             "number is passed, it will be replicated for all subjects and features according to the "
                             "set parameter sharing option.")
    parser.add_argument("--sd_sd_o_vocalic", type=str, required=False, default="1",
                        help="Standard deviation of sigma_o. If the parameters are different per subject and feature, "
                             "it is possible to pass a matrix as a string in MATLAB format, i.e., with semi-colons "
                             "delimiting rows and commas delimiting columns. If parameters are different per subject "
                             "or feature but not both, pass an array with the values separated by comma. If a single "
                             "number is passed, it will be replicated for all subjects and features according to the "
                             "set parameter sharing option.")
    parser.add_argument("--a_p_semantic_link", type=float, required=False, default="1",
                        help="Parameter `a` of the prior distribution of p_semantic_link")
    parser.add_argument("--b_p_semantic_link", type=float, required=False, default="1",
                        help="Parameter `b` of the prior distribution of p_semantic_link")
    parser.add_argument("--share_mean_a0_across_subjects", type=int, required=False, default=0,
                        help="Whether to fit one mu_a for all subjects.")
    parser.add_argument("--share_mean_a0_across_features", type=int, required=False, default=0,
                        help="Whether to fit one mu_a for all features.")
    parser.add_argument("--share_sd_aa_across_subjects", type=int, required=False, default=0,
                        help="Whether to fit one sigma_a for all subjects.")
    parser.add_argument("--share_sd_aa_across_features", type=int, required=False, default=0,
                        help="Whether to fit one sigma_a for all features.")
    parser.add_argument("--share_sd_o_across_subjects", type=int, required=False, default=0,
                        help="Whether to fit one sigma_o for all subjects.")
    parser.add_argument("--share_sd_o_across_features", type=int, required=False, default=0,
                        help="Whether to fit one sigma_o for all features.")
    parser.add_argument("--sd_uc", type=float, required=False, help="Fixed value for sigma_c.")
    parser.add_argument("--mean_a0_vocalic", type=str, required=False,
                        help="Fixed value for mu_a. If the parameters are different per subject and feature, "
                             "it is possible to pass a matrix as a string in MATLAB format, i.e., with semi-colons "
                             "delimiting rows and commas delimiting columns. If parameters are different per subject "
                             "or feature but not both, pass an array with the values separated by comma. If a single "
                             "number is passed, it will be replicated for all subjects and features according to the "
                             "set parameter sharing option.")
    parser.add_argument("--sd_aa_vocalic", type=str, required=False,
                        help="Fixed value for sigma_a. If the parameters are different per subject and feature, "
                             "it is possible to pass a matrix as a string in MATLAB format, i.e., with semi-colons "
                             "delimiting rows and commas delimiting columns. If parameters are different per subject "
                             "or feature but not both, pass an array with the values separated by comma. If a single "
                             "number is passed, it will be replicated for all subjects and features according to the "
                             "set parameter sharing option.")
    parser.add_argument("--sd_o_vocalic", type=str, required=False,
                        help="Fixed value for sigma_o. If the parameters are different per subject and feature, "
                             "it is possible to pass a matrix as a string in MATLAB format, i.e., with semi-colons "
                             "delimiting rows and commas delimiting columns. If parameters are different per subject "
                             "or feature but not both, pass an array with the values separated by comma. If a single "
                             "number is passed, it will be replicated for all subjects and features according to the "
                             "set parameter sharing option.")
    parser.add_argument("--p_semantic_link", type=float, required=False,
                        help="Fixed value for p_semantic_link.")
    parser.add_argument("--nuts_init_method", type=str, required=False, default="jitter+adapt_diag",
                        help="NUTS initialization method.")
    parser.add_argument("--target_accept", type=float, required=False, default=0.8,
                        help="Target acceptance probability used to control step size and reduce "
                             "divergences during inference.")

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
                       vocalic_features=args.vocalic_features,
                       self_dependent=args.self_dependent,
                       sd_mean_uc0=args.sd_mean_uc0,
                       sd_sd_uc=args.sd_sd_uc,
                       mean_mean_a0_vocalic=args.mean_mean_a0_vocalic,
                       sd_mean_a0_vocalic=args.sd_mean_a0_vocalic,
                       sd_sd_aa_vocalic=args.sd_sd_aa_vocalic,
                       sd_sd_o_vocalic=args.sd_sd_o_vocalic,
                       a_p_semantic_link=args.a_p_semantic_link,
                       b_p_semantic_link=args.b_p_semantic_link,
                       share_mean_a0_across_subjects=args.share_mean_a0_across_subjects,
                       share_mean_a0_across_features=args.share_mean_a0_across_features,
                       share_sd_aa_across_subjects=args.share_sd_aa_across_subjects,
                       share_sd_aa_across_features=args.share_sd_aa_across_features,
                       share_sd_o_across_subjects=args.share_sd_o_across_subjects,
                       share_sd_o_across_features=args.share_sd_o_across_features,
                       sd_uc=args.sd_uc,
                       mean_a0_vocalic=args.mean_a0_vocalic,
                       sd_aa_vocalic=args.sd_aa_vocalic,
                       sd_o_vocalic=args.sd_o_vocalic,
                       p_semantic_link=args.p_semantic_link,
                       nuts_init_method=args.nuts_init_method,
                       target_accept=args.target_accept)
