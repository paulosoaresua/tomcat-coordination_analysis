from typing import Dict, Tuple, Any, List
from dateutil.parser import parse
import json


class Trial:
    TRIAL_TOPIC = "trial"
    ASR_TOPIC = "agent/asr/final"
    MISSION_STATE_TOPIC = "observations/events/mission"

    def __init__(self):
        self.utterance_intervals: Dict[str, List[Tuple[Any, Any]]] = {}
        self.player_ids = []
        self.player_id_to_color = []
        self.start_timestamp = None
        self.end_timestamp = None
        self.id = ""

    def parse(self, filepath: str):
        asr_messages = self._get_sorted_asr_messages(filepath)

        self.utterance_intervals = {player: [] for player in self.player_ids}

        for asr_message in asr_messages:
            timestamp = parse(asr_message["header"]["timestamp"])
            if timestamp < self.start_timestamp:
                continue
            if timestamp > self.end_timestamp:
                break

            start_timestamp = max(self.start_timestamp, parse(asr_message["data"]["start_timestamp"]))
            end_timestamp = min(self.end_timestamp, parse(asr_message["data"]["end_timestamp"]))

            player_id = asr_message["data"]["participant_id"]
            if player_id in self.utterance_intervals:
                self.utterance_intervals[player_id].append((start_timestamp, end_timestamp))

    def _get_sorted_asr_messages(self, filepath: str):
        asr_messages = []

        with open(filepath, 'r') as f:
            for line in f:
                try:
                    json_message = json.loads(line)

                    topic = json_message.get("topic", "")
                    if topic == Trial.TRIAL_TOPIC:
                        self._store_trial_info(json_message)
                    if topic == Trial.ASR_TOPIC:
                        asr_messages.append(json_message)
                    elif topic == Trial.MISSION_STATE_TOPIC:
                        self._store_mission_state(json_message)
                except:
                    print(f"Bad json line of len: {len(line)}, {line}")

        sorted_messages = sorted(
            asr_messages, key=lambda x: parse(x["header"]["timestamp"])
        )

        return sorted_messages

    def _store_trial_info(self, json_message: Any):
        self.id = json_message["msd"]["trial_id"]
        self.number = json_message["data"]["trial_number"]

        # Stores the list of player ids in the game and the map between ID and color
        self.player_ids = [playerId.strip() for playerId in json_message["data"]["subjects"]]
        for info in json_message["data"]["client_info"]:
            player_color = info["callsign"].lower()
            player_id = info["participant_id"]
            self.player_id_to_color[player_id] = player_color

    def _store_mission_state(self, json_message: Any):
        # Stores the initial and final timestamp of a mission
        state = json_message["data"]["mission_state"].lower()
        if state == "start":
            self.start_timestamp = parse(json_message["header"]["timestamp"])
        else:
            self.end_timestamp = parse(json_message["header"]["timestamp"])


if __name__ =='__main__':
    trial = Trial()
    trial.parse("../data/study3/Test_A_Thon/TrialMessages_CondBtwn-1_CondWin-Saturn-StaticMap_Trial-1_Team-55cd3a31_Member-Aptiomiomer1_Vers-3.5.1-dev.84-1c1474c.metadata")

    print(f"{trial.start_timestamp} - {trial.end_timestamp}")
    for player_id in trial.player_ids:
        print(f"{player_id} has {len(trial.utterance_intervals[player_id])} utterances in the trial")