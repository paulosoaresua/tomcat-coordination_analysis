import os
import csv
from typing import Tuple, Any, List, Dict
import numpy as np
import random
import uuid

from src.trial import Trial
from src.vocalics import Vocalics


class SpeechSegmentation:

    def __init__(self, feature_names=None):
        if feature_names is None:
            self._feature_names = ["pitch", "intensity"]
        else:
            self._feature_names = feature_names

        self.red_to_green: List[Dict[str, List[Any]]] = [{}, {}]
        self.red_to_blue: List[Dict[str, List[Any]]] = [{}, {}]
        self.green_to_red: List[Dict[str, List[Any]]] = [{}, {}]
        self.green_to_blue: List[Dict[str, List[Any]]] = [{}, {}]
        self.blue_to_red: List[Dict[str, List[Any]]] = [{}, {}]
        self.blue_to_green: List[Dict[str, List[Any]]] = [{}, {}]

        self.trial = None

    def generate_random_series(self, min_seg_size: int = 10, max_seg_size: int = 40, min_num_seg: int = 10,
                               max_num_seg: int = 100):
        """
        Generates random time series for all features and dyads
        """

        player_colors = ["red", "green", "blue"]
        for player1_color in player_colors:
            for player2_color in player_colors:
                if player1_color != player2_color:
                    num_segments = random.randint(min_num_seg, max_num_seg)

                    # Source and Target
                    vocalics = [{}, {}]
                    for i in range(2):
                        vocalics[i] = {"timestamp": []}
                        for feature_name in self._feature_names:
                            vocalics[i][feature_name] = []

                            for _ in range(num_segments):
                                seg_size = random.randint(min_seg_size, max_seg_size)
                                vocalics[i][feature_name].append(np.random.randn(seg_size).tolist())

                    setattr(self, f"{player1_color}_to_{player2_color}", vocalics)

    def save(self, out_dir: str):
        data_id = self.trial.id if self.trial is not None else str(uuid.uuid1())
        out_dir = os.path.join(out_dir, data_id)
        os.makedirs(out_dir, exist_ok=True)

        series = [
            self.red_to_green,
            self.red_to_blue,
            self.green_to_red,
            self.green_to_blue,
            self.blue_to_red,
            self.blue_to_green
        ]

        dir_names = [
            "red_to_green",
            "red_to_blue",
            "green_to_red",
            "green_to_blue",
            "blue_to_red",
            "blue_to_green",
        ]

        for i in range(len(dir_names)):
            dir_name = os.path.join(out_dir, dir_names[i])
            os.makedirs(dir_name, exist_ok=True)

            with open(os.path.join(dir_name, "source_timestamp.txt"), "w", newline="") as f:
                wr = csv.writer(f)
                wr.writerows(series[i][0]["timestamp"])

            with open(os.path.join(dir_name, "target_timestamp.txt"), "w", newline="") as f:
                wr = csv.writer(f)
                wr.writerows(series[i][1]["timestamp"])

            for feature_name in self._feature_names:
                with open(os.path.join(dir_name, f"source_{feature_name}.txt"), "w", newline="") as f:
                    wr = csv.writer(f)
                    wr.writerows(series[i][0][feature_name])

                with open(os.path.join(dir_name, f"target_{feature_name}.txt"), "w", newline="") as f:
                    wr = csv.writer(f)
                    wr.writerows(series[i][1][feature_name])

    def load(self, in_dir: str):
        series = [
            self.red_to_green,
            self.red_to_blue,
            self.green_to_red,
            self.green_to_blue,
            self.blue_to_red,
            self.blue_to_green
        ]

        dir_names = [
            "red_to_green",
            "red_to_blue",
            "green_to_red",
            "green_to_blue",
            "blue_to_red",
            "blue_to_green",
        ]

        for i in range(len(dir_names)):
            dir_name = os.path.join(in_dir, dir_names[i])

            with open(os.path.join(dir_name, "source_timestamp.txt"), newline="") as f:
                reader = csv.reader(f)
                series[i][0]["timestamp"] = list(reader)

            with open(os.path.join(dir_name, "target_timestamp.txt"), newline="") as f:
                reader = csv.reader(f)
                series[i][1]["timestamp"] = list(reader)

            for feature_name in self._feature_names:
                with open(os.path.join(dir_name, f"source_{feature_name}.txt"), newline="") as f:
                    reader = csv.reader(f)
                    series[i][0][feature_name] = [list(map(float, row)) for row in list(reader)]

                with open(os.path.join(dir_name, f"target_{feature_name}.txt"), newline="") as f:
                    reader = csv.reader(f)
                    series[i][1][feature_name] = [list(map(float, row)) for row in list(reader)]

    def create_dyadic_vocalics_segment_series(self, trial_filepath: str):
        self.trial = Trial()
        self.trial.parse(trial_filepath)
        vocalics = Vocalics()
        vocalics.read_features(self.trial, self._feature_names)

        for player1_id in self.trial.player_ids:
            for player2_id in self.trial.player_ids:
                if player1_id == player2_id:
                    continue

                self._store_segments(self.trial, vocalics, player1_id, player2_id)

    def _store_segments(self, trial: Trial, vocalics: Vocalics, source_player_id: str, target_player_id: str):
        source_vocalics, target_vocalics = SpeechSegmentation._segment(trial, vocalics, source_player_id,
                                                                       target_player_id)

        source_player_color = trial.player_id_to_color[source_player_id]
        target_player_color = trial.player_id_to_color[target_player_id]

        if source_player_color == "red":
            if target_player_color == "green":
                self.red_to_green[0] = source_vocalics
                self.red_to_green[1] = target_vocalics
            else:
                self.red_to_blue[0] = source_vocalics
                self.red_to_blue[1] = target_vocalics
        elif source_player_color == "green":
            if target_player_color == "red":
                self.green_to_red[0] = source_vocalics
                self.green_to_red[1] = target_vocalics
            else:
                self.green_to_blue[0] = source_vocalics
                self.green_to_blue[1] = target_vocalics
        else:
            if target_player_color == "red":
                self.blue_to_red[0] = source_vocalics
                self.blue_to_red[1] = target_vocalics
            else:
                self.blue_to_green[0] = source_vocalics
                self.blue_to_green[1] = target_vocalics

    @staticmethod
    def _segment(trial: Trial, vocalics: Vocalics, source_player_id: str, target_player_id: str) -> Tuple[
        Dict[str, List[Any]], Dict[str, List[Any]]]:

        source_utterances = trial.utterance_intervals[source_player_id]
        target_utterances = trial.utterance_intervals[target_player_id]

        [source_segments, target_segments] = remove_overlapping_segments(source_utterances, target_utterances)
        [source_segments, target_segments] = equalize_segments(source_segments, target_segments)

        source_vocalics = vocalics.get_segmented_vocalics(source_player_id, source_segments)
        target_vocalics = vocalics.get_segmented_vocalics(target_player_id, target_segments)

        return source_vocalics, target_vocalics


def remove_overlapping_segments(source_utterances: List[Tuple[Any, Any]],
                                target_utterances: List[Tuple[Any, Any]]) -> Tuple[
    List[Tuple[Any, Any]], List[Tuple[Any, Any]]]:
    """
    Removes target utterances that overlap with the source ones.
    """

    source_idx = 0
    target_idx = 0
    source_segments = []
    target_segments = []

    while source_idx < len(source_utterances) or target_idx < len(target_utterances):
        while source_idx < len(source_utterances) and to_the_left_of(source_utterances[source_idx],
                                                                     target_utterances[target_idx]):
            # |---|
            #        |-----|
            source_segments.append(source_utterances[source_idx])
            source_idx += 1

        while target_idx < len(target_utterances) and to_the_left_of(target_utterances[target_idx],
                                                                     source_utterances[source_idx]):
            #        |-----|
            # |---|
            target_segments.append(target_utterances[target_idx])
            target_idx += 1

        if source_idx < len(source_utterances) and target_idx < len(target_utterances):
            if overlaps(target_utterances[target_idx], source_utterances[source_idx]):
                if inside_of(source_utterances[source_idx], target_utterances[target_idx]):
                    #    |-----|
                    # |-----------|
                    # The target player is not paying attention to what the source player is saying because he is
                    # talking at the same time.
                    source_idx += 1
                else:
                    if source_utterances[source_idx][1] < target_utterances[target_idx][1]:
                        # |-----------|
                        #           |-----|
                        segment = (source_utterances[source_idx][0], target_utterances[target_idx][0])
                        source_segments.append(segment)
                        source_idx += 1
                    elif inside_of(target_utterances[target_idx], source_utterances[source_idx]):
                        # |-----------|
                        #    |-----|
                        left_segment = (source_utterances[source_idx][0], target_utterances[target_idx][0])
                        source_segments.append(left_segment)
                        target_segments.append(target_utterances[target_idx])

                        # Update current source utterance
                        right_segment = (target_utterances[target_idx][1], source_utterances[source_idx][1])
                        source_utterances[source_idx] = right_segment
                        target_idx += 1
                    else:
                        #         |-----------|
                        # |-----------|
                        target_segments.append(target_utterances[target_idx])

                        # Update current source utterance
                        right_segment = (target_utterances[target_idx][1], source_utterances[source_idx][1])
                        source_utterances[source_idx] = right_segment
                        target_idx += 1

        if source_idx == len(source_utterances) and target_idx < len(target_utterances):
            target_segments.extend(target_utterances[target_idx:])
            target_idx = len(target_utterances)
        if target_idx == len(target_utterances) and source_idx < len(source_utterances):
            source_segments.extend(source_utterances[source_idx:])
            source_idx = len(source_utterances)

    return source_segments, target_segments


def to_the_left_of(interval1: Tuple[Any, Any], interval2: Tuple[Any, Any]):
    return interval1[1] <= interval2[0]


def to_the_right_of(interval1: Tuple[Any, Any], interval2: Tuple[Any, Any]):
    return interval1[0] >= interval2[1]


def inside_of(interval1: Tuple[Any, Any], interval2: Tuple[Any, Any]):
    return interval1[0] >= interval2[0] and interval1[1] <= interval2[1]


def overlaps(interval1: Tuple[Any, Any], interval2: Tuple[Any, Any]):
    return not (interval1[1] <= interval2[0] or interval1[0] >= interval2[1])


def equalize_segments(source_segments: List[Tuple[Any, Any]],
                      target_segments: List[Tuple[Any, Any]]) -> Tuple[List[Tuple[Any, Any]], List[Tuple[Any, Any]]]:
    """
    It keeps only immediate segments from the source and the target and equalizes time steps. This function assumes
    there are no overlapping segments between source and target interval series.

    For example,
    source: |----| |-------| |----|        |----|
    target:                         |----|        |-------| |----|

    results in

    source: |----|        |----|
    target:        |----|        |-------|
    """

    source_idx = 0
    target_idx = 0

    reduced_source_segments = []
    reduced_target_segments = []

    while source_idx < len(source_segments) and target_idx < len(target_segments):
        while target_idx < len(target_segments) and to_the_left_of(target_segments[target_idx],
                                                                   source_segments[source_idx]):
            # Target segments have to come after source ones. We discard all target segments that come before the
            # source segment.
            # source:                |-----|
            # target: |-x-| |---x---|        |---|
            target_idx += 1

        if target_idx < len(target_segments):
            while source_idx < len(source_segments) and to_the_left_of(source_segments[source_idx],
                                                                       target_segments[target_idx]):
                # Discard source segments not immediately adjacent to the next target segment
                # |-x-| |--x--| |--x--|         |----|
                #                       |-----|
                source_idx += 1

            reduced_source_segments.append(source_segments[source_idx - 1])
            reduced_target_segments.append(target_segments[target_idx])

    return reduced_source_segments, reduced_target_segments


if __name__ == "__main__":
    ss = SpeechSegmentation()
    ss.generate_random_series()
    ss.save("../data/study3/vocalic_series")
