import argparse
from datetime import timedelta
from dateutil.parser import parse
import json
import logging
from glob import glob
import os
import pytz

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger()

SUBJECTS = ["lion", "tiger", "leopard"]


def parse_body_data(data_dir: str, task: str, num_time_steps: int, frequency: float, out_dir: str):
    experiment_dirs = glob(f"{data_dir}/exp_*")

    table = []
    for experiment_dir in tqdm(experiment_dirs, desc="Experiment"):
        experiment_id = experiment_dir[experiment_dir.rfind("/") + 1:]

        # The ping-pong task's timestamps are in UTC and fNIRS in MST. We convert the original time to MST
        # for alignment purposes.
        initial_timestamp_json = load_json(f"{experiment_dir}/baseline_tasks/metadata.json")
        initial_timestamp = parse(initial_timestamp_json[task]["initial_timestamp_utc"])
        # Remove time zone info after conversion to allow direct usage with a pandas DataFrame.
        # initial_timestamp = change_time_zone(initial_timestamp, "MST").replace(tzinfo=None)

        for i, subject in enumerate(SUBJECTS):
            if not os.path.exists(f"{experiment_dir}/{subject}/motion_energy.csv"):
                continue

            body_df = pd.read_csv(f"{experiment_dir}/{subject}/motion_energy.csv", delimiter=",",
                                  parse_dates=["timestamp"])

            body_df_task = body_df[body_df["timestamp"] >= initial_timestamp].reset_index(drop=True)
            body_df_task["total_energy"] = body_df_task.sum(axis="columns", numeric_only=True)

            # We move a window of size 1/ freq, and collect the average of measurements in that window if any.
            window_start_ts = initial_timestamp

            avg_total_energy = np.zeros((1, num_time_steps))
            measurements_per_window = np.zeros(num_time_steps)
            for t in range(num_time_steps):
                window_end_ts = window_start_ts + timedelta(seconds=1.0 / frequency)
                if t < num_time_steps - 1:
                    tmp = body_df_task[(body_df_task["timestamp"] >= window_start_ts) &
                                       (body_df_task["timestamp"] < window_end_ts)]
                else:
                    tmp = body_df_task[(body_df_task["timestamp"] >= window_start_ts) &
                                       (body_df_task["timestamp"] <= window_end_ts)]

                if len(tmp) == 0:
                    # No measurement in the window
                    msg = f"No measurement find for time step {t} in experiment {experiment_id} and subject {subject} " \
                          f"Between {window_start_ts.isoformat()} and {window_end_ts.isoformat()}"
                    logger.warning(msg)
                else:
                    avg_total_energy[:, t] = tmp["total_energy"].mean()

                measurements_per_window[t] = len(tmp)

                window_start_ts = window_start_ts + timedelta(seconds=1.0 / frequency)

            table.append(
                [experiment_id, subject, frequency, avg_total_energy.tolist(), (measurements_per_window == 0).sum(),
                 measurements_per_window.tolist(), initial_timestamp.isoformat()])

    df = pd.DataFrame(table, columns=["experiment_id", "subject", "frequency_hz", "total_energy", "no_measurement",
                                      "measurements_per_window", "initial_timestamp"])
    out_dir = f"{out_dir}/{task}"
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(f"{out_dir}/body_movements.csv")


def change_time_zone(dt, new_time_zone: str):
    new_time_zone = pytz.timezone(new_time_zone)
    dt = dt.astimezone(new_time_zone)

    return dt


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parses NIRS data and produces a time series of HbTotal measurements per subject over time for a "
                    "given baseline task."
    )

    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing experiment folders.")
    parser.add_argument("--task", type=str, required=True,
                        help="Name of the baseline task.")
    parser.add_argument("--n_time_steps", type=int, required=True,
                        help="Number of time steps.")
    parser.add_argument("--freq", type=float, required=True,
                        help="Frequency in Hz.")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Directory where to save the parsed data")

    args = parser.parse_args()

    parse_body_data(args.data_dir, args.task, args.n_time_steps, args.freq, args.out_dir)
