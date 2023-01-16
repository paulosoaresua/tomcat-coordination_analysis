import argparse
from glob import glob
import os
from dateutil.parser import parse

import pandas as pd
import numpy as np

import json

from tqdm import tqdm

TRIAL_TOPIC = "trial"
SCOREBOARD_TOPIC = "observations/events/scoreboard"
TOPIC_ASI_M2_M3 = "agent/measures/AC_Aptima_TA3_measures"
TOPIC_ASI_M5_THREAT = "agent/ac/threat_room_communication"
TOPIC_ASI_M5_VICTIM = "agent/ac/victim_type_communication"
TOPIC_ASI_M5_BELIEF = "agent/ac/belief_diff"
TOTAL_ROOMS = 64


def serialize_trials(metadata_dir: str, out_dir: str):
    if not os.path.exists(metadata_dir):
        raise Exception(f"Directory {metadata_dir} does not exist.")

    filepaths = glob(f"{metadata_dir}/*.metadata")

    table = []
    for filepath in tqdm(filepaths):
        with open(filepath, 'r') as f:
            messages = []
            for line in f:
                try:
                    json_message = json.loads(line)
                    topic = json_message.get("topic", "")

                    if topic in (
                            "trial", TOPIC_ASI_M2_M3, TOPIC_ASI_M5_THREAT, TOPIC_ASI_M5_VICTIM, TOPIC_ASI_M5_BELIEF,
                            SCOREBOARD_TOPIC):
                        messages.append(json_message)
                except:
                    pass

            messages = sorted(
                messages, key=lambda x: parse(x["header"]["timestamp"])
            )

            m2 = {}
            m5_threat = 0
            m5_victim = 0
            m5_belief = 0
            m3 = 1
            trial_number = None
            for json_message in messages:
                topic = json_message.get("topic", "")

                if topic == TRIAL_TOPIC:
                    trial_number = json_message["data"]["trial_number"]
                elif topic == TOPIC_ASI_M2_M3:
                    if json_message["data"]["event_properties"][
                        "qualifying_event_sub_type"] == "Event:VictimEvacuated":

                        for measure_data in json_message["data"]["measure_data"]:
                            if measure_data["measure_id"] == "ASI-M2":
                                for data in measure_data["additional_data"]:
                                    if data["rescued"]:
                                        if data["unique_id"] not in m2:
                                            m2[data["unique_id"]] = data["rescue_time"] - data["discovery_time"]
                            elif measure_data["measure_id"] == "ASI-M3":
                                m3 = measure_data["measure_value"]

                elif topic == TOPIC_ASI_M5_THREAT:
                    rooms = set()
                    for i, room in enumerate(json_message["data"]["nearest_room"]):
                        if json_message["data"]["is_observed_threat_room"][i]:
                            rooms.add(room)
                    m5_threat = len(rooms) / TOTAL_ROOMS

                elif topic == TOPIC_ASI_M5_VICTIM:
                    for i, marker in enumerate(json_message["data"]["marker_block_type"]):
                        if marker in ["A", "B"] and json_message["data"]["victims_match_marker_block"][i] != "":
                            m5_victim += 1

                elif topic == TOPIC_ASI_M5_BELIEF:
                    pass

                elif topic == SCOREBOARD_TOPIC:
                    score = int(json_message["data"]["scoreboard"]["TeamScore"])

            table.append([trial_number, np.mean(list(m2.values())), m3, m5_threat, m5_victim, m5_belief, score])

    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(table, columns=["Trial", "M2", "M3", "M5_Threat", "M5_Victim", "M5_Belief", "Score"])
    df.to_csv(f"{out_dir}/outcome_measures.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parses a Minecraft .metadata file to extract relevant data to the coordination model and "
                    "saves the post processed trial structures to a folder."
    )

    parser.add_argument("--metadata_dir", type=str, required=True,
                        help="Directory where the metadata files are located.")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Directory directory with serialized trial must be saved.")

    args = parser.parse_args()

    serialize_trials(args.metadata_dir, args.out_dir)
