from typing import List

import json
from datetime import datetime
import os
import uuid

import numpy as np
import pandas as pd


def tmux(command: str):
    os.system(f"tmux {command}")


def tmux_shell(command: str):
    tmux(f'send-keys "{command}" "C-m"')


def parallel_inference(out_dir: str, evidence_filepath: str, conda_env_name: str, shell_source: str,
                       num_parallel_processes: int, model: str, burn_in: int, num_samples: int, num_chains: int,
                       seed: int, num_inference_jobs: int, do_posterior: bool, initial_coordination: float, num_subjects: int,
                       brain_channels: str, vocalic_features: str, self_dependent: bool, sd_uc: float,
                       sd_mean_a0_brain: str, sd_sd_aa_brain: str, sd_sd_o_brain: str, sd_mean_a0_body: str,
                       sd_sd_aa_body: str, sd_sd_o_body: str, a_mixture_weights: str, sd_mean_a0_vocalic: str,
                       sd_sd_aa_vocalic: str, sd_sd_o_vocalic: str, a_p_semantic_link: float, b_p_semantic_link: float):
    # Save execution parameters
    execution_time = datetime.now().strftime("%Y.%m.%d--%H.%M.%S")
    results_folder = f"{out_dir}/{model}/{execution_time}"
    execution_params = {
        "burn_in": burn_in,
        "num_samples": num_samples,
        "num_chains": num_chains,
        "seed": seed,
        "num_inference_jobs": num_inference_jobs,
        "do_posterior": do_posterior,
        "initial_coordination": initial_coordination,
        "num_subjects": num_subjects,
        "brain_channels": brain_channels,
        "vocalic_features": vocalic_features,
        "self_dependent": self_dependent,
        "sd_uc": sd_uc,
        "sd_mean_a0_brain": sd_mean_a0_brain,
        "sd_sd_aa_brain": sd_sd_aa_brain,
        "sd_sd_o_brain": sd_sd_o_brain,
        "sd_mean_a0_body": sd_mean_a0_body,
        "sd_sd_aa_body": sd_sd_aa_body,
        "sd_sd_o_body": sd_sd_o_body,
        "a_mixture_weights": a_mixture_weights,
        "sd_mean_a0_vocalic": sd_mean_a0_vocalic,
        "sd_sd_aa_vocalic": sd_sd_aa_vocalic,
        "sd_sd_o_vocalic": sd_sd_o_vocalic,
        "a_p_semantic_link": a_p_semantic_link,
        "b_p_semantic_link": b_p_semantic_link
    }

    with open(f"{results_folder}/execution_params.json", "w") as f:
        json.dump(execution_params, f, indent=4)

    evidence_df = pd.read_csv(evidence_filepath, index_col=0)
    experiments = sorted(list(evidence_df["experiment_id"].unique()))
    tmux_session_name = f"{model}_{uuid.uuid1()}"
    for i, experiments_per_process in enumerate(np.array_split(experiments, num_parallel_processes)):
        # Start a new process in a tmux window for the experiments to be processed.
        experiment_ids = ",".join(experiments_per_process)
        tmux_window_name = experiment_ids

        if i == 0:
            tmux(f"new-session -s{tmux_session_name} -n{tmux_window_name}")
        else:
            tmux(f"new_window -n{tmux_window_name}")

        # We source the interactive shell script and activate the proper conda environment with the required libraries
        # to execute the script.
        tmux_shell(f"source ~/.{shell_source}")
        tmux_shell(f"conda activate {conda_env_name}")

        # Point PYTHONPATH to the project directory so we can execute the python script from the terminal
        project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        tmux_shell(f'export PYTHONPATH="{project_dir}"')

        # Call the actual inference script
        call_python_script_command = f'python3 "{project_dir}/scripts/inference.py" ' \
                                     f'--out_dir="{results_folder}" ' \
                                     f'--experiment_ids="{experiment_ids}" ' \
                                     f'--evidence_filepath="{evidence_filepath}" ' \
                                     f'--model="{model}" ' \
                                     f'--burn_in="{burn_in}" ' \
                                     f'--num_samples="{num_samples}" ' \
                                     f'--num_chains="{num_chains}" ' \
                                     f'--seed="{seed}" ' \
                                     f'--num_inference_jobs="{num_inference_jobs}" ' \
                                     f'--initial_coordination="{initial_coordination}" ' \
                                     f'--num_subjects="{num_subjects}" ' \
                                     f'--brain_channels="{brain_channels}" ' \
                                     f'--vocalic_features="{vocalic_features}" ' \
                                     f'--self_dependent="{self_dependent}" ' \
                                     f'--sd_uc="{sd_uc}" ' \
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
                                     f'--a_p_semantic_link="{a_p_semantic_link}" ' \
                                     f'--b_p_semantic_link="{b_p_semantic_link}"'

        tmux_shell(call_python_script_command)


if __name__ == "__main__":
    pass
