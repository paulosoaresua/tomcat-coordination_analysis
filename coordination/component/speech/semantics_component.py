from __future__ import annotations

import json
import logging
import os
import pickle
from datetime import datetime
from typing import List, Optional

import numpy as np
from pkg_resources import resource_string

from coordination.component.speech.common import SparseSeries
from coordination.entity.vocalics import Utterance, Vocalics

UTTERANCE_MISSING_VOCALICS_DURATION_THRESHOLD = 1

logger = logging.getLogger()


class SemanticLink:

    def __init__(self, source: Utterance, target: Utterance, source_label: str, target_label: str):
        self.source = source
        self.target = target
        self.source_label = source_label
        self.target_label = target_label


class SemanticsComponent:
    def __init__(self, semantic_links: List[SemanticLink], window_size: int = 5):
        self.semantic_links = semantic_links
        self.window_size = window_size  # Number of utterances apart to consider a link exists

    @classmethod
    def from_vocalics(cls, vocalics: Vocalics, window_size: int):
        # Sorted list with the utterances from all the subjects
        utterances: List[Utterance] = []
        for u in vocalics.utterances_per_subject.values():
            utterances.extend(u)
        utterances.sort(key=lambda utterance: utterance.start)

        link_config = json.loads(
            resource_string("coordination.resources.conf.speech_semantics", "linked_labels.json").decode())

        target_labels = set([link["target"] for link in link_config["linked_labels"]])

        target_2_source = {}
        for link in link_config["linked_labels"]:
            if link["target"] not in target_2_source:
                target_2_source[link["target"]] = set()
            target_2_source[link["target"]].add(link["source"])

        semantic_links = []
        queue = []
        for utterance in utterances:
            target_labels = utterance.labels.intersection(target_labels)
            if len(utterance.labels.intersection(target_labels)) > 0:
                # Look for any link with utterances in the queue (from the most recent to the oldest)
                semantic_link = None
                for source_utterance in reversed(queue):
                    if utterance.subject_id == source_utterance.subject_id:
                        continue

                    for target_label in target_labels:
                        linked_source_labels = source_utterance.labels.intersection(target_2_source[target_label])
                        if len(linked_source_labels) > 0:
                            semantic_link = SemanticLink(source_utterance, utterance, linked_source_labels.pop(),
                                                         target_label)
                            break

                    if semantic_link is not None:
                        semantic_links.append(semantic_link)
                        break

                queue.append(utterance)

                if len(queue) > window_size:
                    queue.pop(0)

        return cls(semantic_links, window_size)

    @classmethod
    def from_trial_directory(cls, trial_dir: str) -> SemanticsComponent:
        semantics_component_path = f"{trial_dir}/semantic_links.pkl"
        window_size_path = f"{trial_dir}/semantic_link_window_size.txt"

        if not os.path.exists(semantics_component_path):
            raise Exception(f"Could not find the file semantic_links.pkl in {trial_dir}.")

        if not os.path.exists(window_size_path):
            raise Exception(f"Could not find the file semantic_link_window_size.txt in {trial_dir}.")

        with open(semantics_component_path, "rb") as f:
            semantic_links = pickle.load(f)

        with open(window_size_path, "r") as f:
            window_size = json.load(f)

        return cls(semantic_links, window_size)

    def to_array(self, num_time_steps: int, mission_start: datetime, ) -> SparseSeries:
        values = np.zeros(num_time_steps)

        for i, semantic_link in enumerate(self.semantic_links):
            # We consider that the observation is available at the end of an utterance. We take the average vocalics
            # per feature within the utterance as a measurement at the respective time step.
            time_step = int((semantic_link.target.end - mission_start).total_seconds())

            if time_step >= num_time_steps:
                msg = f"Time step {time_step} exceeds the number of time steps {num_time_steps} at " \
                      f"utterance link {i} out of {len(self.semantic_links)} ending at " \
                      f"{semantic_link.target.end.isoformat()} considering an initial timestamp of {mission_start.isoformat()}."
                logger.warning(msg)
                break

            values[time_step] = 1

        return values

    def save(self, out_dir: str):
        with open(f"{out_dir}/semantic_links.pkl", "wb") as f:
            pickle.dump(self.semantic_links, f)

        with open(f"{out_dir}/semantic_link_window_size.txt", "w") as f:
            json.dump(self.window_size, f)
