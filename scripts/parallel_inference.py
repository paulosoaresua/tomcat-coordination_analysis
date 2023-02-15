import os
import uuid

import numpy as np
import pandas as pd


def tmux(command: str):
    os.system(f"tmux {command}")


def tmux_shell(command: str):
    tmux(f'send-keys "{command}" "C-m"')

def parallel_inference(evidence_filepath: str, conda_env_name: str, shell_source: str, num_parallel_processes: int,
                       model: str, burn_in: int, num_samples: int, num_chains: int, seed: int, num_inference_jobs: int,
                       initial_coordination: float, num_subjects: int, num_brain_channels: int, self_dependent: bool,
                       sd_uc: float, sd_mean_a0_brain: np.ndarray, sd_sd_aa_brain: np.ndarray,
                       sd_sd_o_brain: np.ndarray, sd_mean_a0_body: np.ndarray, sd_sd_aa_body: np.ndarray,
                       sd_sd_o_body: np.ndarray, a_mixture_weights: np.ndarray, sd_mean_a0_vocalic: np.ndarray,
                       sd_sd_aa_vocalic: np.ndarray, sd_sd_o_vocalic: np.ndarray, a_p_semantic_link: float,
                       b_p_semantic_link: float):
    evidence_df = pd.read_csv(evidence_filepath, index_col=0)

    experiments = sorted(list(evidence_df["experiment_id"].unique()))

    experiments_per_process = np.array_split(experiments, num_parallel_processes)

    tmux_session_name = f"{model}_{uuid.uuid1()}"

    for i, experiments_per_process in enumerate(experiments):
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
                                     f'--num_brain_channels="{num_brain_channels}" ' \
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
