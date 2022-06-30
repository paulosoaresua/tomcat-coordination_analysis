import json
from typing import Tuple


def utterance_metadata_parser(metadata_path: str) -> Tuple[str, str, str, str, str]:
    file = open(metadata_path, 'r')

    while True:
        line = file.readline()

        if not line:
            break
        else:
            message = json.loads(line)
            if "topic" in message and message["topic"] == "agent/asr/final":
                trial_id = message["msg"]["trial_id"]
                participant_id = message["data"]["participant_id"]
                text = message["data"]["text"]
                start_timestamp = message["data"]["start_timestamp"]
                end_timestamp = message["data"]["end_timestamp"]
                yield trial_id, participant_id, text, start_timestamp, end_timestamp

    file.close()
