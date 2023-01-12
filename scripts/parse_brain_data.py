import os
import pickle

from dateutil.parser import parse
from datetime import timedelta
import json
from glob import glob

import numpy as np
import pandas as pd
from tqdm import tqdm

COLUMNS = [
    "S1-D1",
    "S1-D2",
    "S2-D1",
    "S2-D3",
    "S3-D1",
    "S3-D3",
    "S3-D4",
    "S4-D2",
    "S4-D4",
    "S4-D5",
    "S5-D3",
    "S5-D4",
    "S5-D6",
    "S6-D4",
    "S6-D6",
    "S6-D7",
    "S7-D5",
    "S7-D7",
    "S8-D6",
    "S8-D7"
]

SUBJECTS = ["lion", "tiger", "leopard"]

def parse(data_dir: str, task: str, num_time_steps: int, step_size_in_seconds: float, out_dir: str):
    experiment_dirs = glob(f"{data_dir}/exp_*")
    for experiment_dir in tqdm(experiment_dirs, desc="Experiment"):
        hb_total = np.zeros((len(SUBJECTS), len(COLUMNS), 120))

        for i, subject in enumerate(SUBJECTS):
            nirs_df = pd.read_csv(f"{experiment_dir}/{subject}/eeg_fnirs_pupil/NIRS_filtered.csv", delimiter="\t",
                                  parse_dates=["human_readable_time"])

            initial_timestamp_json = load_json(f"{experiment_dir}/initial_timestamp.json")
            initial_timestamp = parse(initial_timestamp_json[task])

            nirs_df_task = nirs_df[nirs_df["event_type"] == task]
            nirs_df_task = nirs_df_task[
                nirs_df_task["human_readable_time"] >= initial_timestamp].reset_index(drop=True)

            hbO_columns = list(map(lambda x: f"{x}_HbO", COLUMNS))
            hbR_columns = list(map(lambda x: f"{x}_HbR", COLUMNS))

            prev_timestamp = initial_timestamp
            curr_timestamp = initial_timestamp + timedelta(seconds=step_size_in_seconds)

            for t in range(num_time_steps):
                tmp = nirs_df_task[(nirs_df_task["human_readable_time"] >= prev_timestamp) & (
                            nirs_df_task["human_readable_time"] < curr_timestamp)]

                hb_total_in_window = tmp[hbO_columns].values + tmp[hbR_columns].values
                hb_total[i, :, t] = np.mean(hb_total_in_window, axis=0)

                prev_timestamp = curr_timestamp
                curr_timestamp = curr_timestamp + timedelta(seconds=step_size_in_seconds)

        out_dir = f"{out_dir}/{task}"
        os.makedirs(out_dir, exist_ok=True)
        with open(f"{out_dir}/brain_signals.pkl", "wb") as f:
            pickle.dump(hb_total, f)


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


if __name__ == "__main__":
