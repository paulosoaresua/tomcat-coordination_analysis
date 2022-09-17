from typing import Any, Dict, List, Union

import matplotlib.pyplot as plt

from coordination.component.speech.common import SegmentedUtterance, VocalicsSparseSeries
from coordination.entity.vocalics import Utterance


def plot_vocalic_features(axs: List[Any], series: VocalicsSparseSeries, feature_names=List[str],
                          timestamp_as_index: bool = False):
    times_per_subject = {}
    time_labels_per_subject = {}

    for t, mask in enumerate(series.mask):
        if mask == 0:
            continue

        subject = series.utterances[t].subject_id
        if subject not in times_per_subject:
            times_per_subject[subject] = [t]
            time_labels_per_subject[subject] = [series.timestamps[t] if timestamp_as_index else t]
        else:
            times_per_subject[subject].append(t)
            time_labels_per_subject[subject].append(series.timestamps[t] if timestamp_as_index else t)

    for i, feature_name in enumerate(feature_names):
        axs[i].set_ylabel("Average Value in Utterance")
        axs[i].set_title(feature_name.capitalize(), fontsize=14, weight="bold")
        for subject in sorted(times_per_subject.keys()):
            axs[i].plot(time_labels_per_subject[subject],
                        series.values[i, times_per_subject[subject]],
                        marker='o',
                        label=subject.capitalize())
        axs[i].legend()
        axs[i].set_xlabel("Time" if timestamp_as_index else "Time Step (seconds)")


def plot_utterance_durations(ax: Any,
                             utterances_per_subject: Dict[str, List[Union[Utterance, SegmentedUtterance]]]) -> None:
    ax.margins(x=0)
    subjects = sorted(utterances_per_subject.keys())
    num_subjects = len(subjects)

    for i, subject in enumerate(subjects):
        utterances = utterances_per_subject[subject]
        durations = [[utterance.start, utterance.end - utterance.start] for utterance in utterances]
        ax.broken_barh(durations, ((num_subjects - i - 1) * 5, 4.9), facecolors='tab:blue')

    ax.set_yticks([(num_subjects - i - 1) * 5 + 2.5 for i in range(num_subjects)])
    ax.set_yticklabels(subjects)

    ax.set_title("Utterance Durations", fontsize=14, weight="bold")
    plt.xlabel("Timestamps")
