import argparse
from datetime import datetime
import json
from glob import glob
import os

import pandas as pd
from tqdm import tqdm


def collect_baseline_tasks_metadata(data_dir: str):
    experiment_dirs = glob(f"{data_dir}/exp_*")
    for experiment_dir in tqdm(experiment_dirs, desc="Experiment"):
        cooperative_pp_filepath = list(glob(f"{experiment_dir}/baseline_tasks/ping_pong/cooperative_*.csv"))[0]
        cooperative_ping_pong_df = pd.read_csv(cooperative_pp_filepath, delimiter=";",
                                               parse_dates=["human_readable_time"])

        initial_timestamp_utc = cooperative_ping_pong_df[cooperative_ping_pong_df["started"]][
            "human_readable_time"].min()
        team_score = cooperative_ping_pong_df.iloc[-1]["score_left"]
        ai_score = cooperative_ping_pong_df.iloc[-1]["score_right"]

        metadata = {
            "ping_pong_cooperative_0": {
                "initial_timestamp_utc": initial_timestamp_utc.isoformat(),
                "team_score": int(team_score),
                "ai_score": int(ai_score)
            }
        }

        # TODO: add other baseline tasks later. For the moment, we only need cooperative pong-pong.

        out_dir = f"{experiment_dir}/baseline_tasks"
        os.makedirs(out_dir, exist_ok=True)
        with open(f"{out_dir}/metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parses Baseline tasks to extract metadata info."
    )

    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing experiment folders.")

    args = parser.parse_args()

    collect_baseline_tasks_metadata(args.data_dir)
