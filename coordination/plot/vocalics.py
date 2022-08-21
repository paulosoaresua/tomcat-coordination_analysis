from typing import Any, List

import matplotlib.pyplot as plt

from coordination.common.sparse_series import SparseSeries


def plot_vocalic_features(axs: List[Any], series_a: SparseSeries, series_b: SparseSeries, feature_names=List[str],
                          timestamp_as_index: bool = False):
    time_steps_a = [t for t, mask in enumerate(series_a.mask) if mask == 1]
    time_steps_b = [t for t, mask in enumerate(series_b.mask) if mask == 1]

    time_labels_a = [series_a.timestamps[t] for t in time_steps_a] if timestamp_as_index else time_steps_a
    time_labels_b = [series_b.timestamps[t] for t in time_steps_b] if timestamp_as_index else time_steps_b

    # plot
    for i, feature_name in enumerate(feature_names):
        axs[i].set_ylabel("Average Value in Utterance")
        axs[i].set_title(feature_name.capitalize(), fontsize=14, weight="bold")
        axs[i].plot(time_labels_a,
                    series_a.values[i, time_steps_a],
                    color='tab:red',
                    marker='o',
                    label="Subject A")
        axs[i].plot(time_labels_b,
                    series_b.values[i, time_steps_b],
                    color='tab:blue',
                    marker='o',
                    label="Subject B")
        axs[i].legend()
        axs[i].set_xlabel("Time" if timestamp_as_index else "Time Step (seconds)")
        # axs[i].set_xlim([0, max(series_a.num_time_steps, series_b.num_time_steps)])


def plot_utterance_durations(ax: Any, utterances_per_subject) -> None:
    """Plot durations of utterances

    Args:
        ax (Any): axis
        utterances_per_subject (Dictionary[str, Utterance or SegmentedUtterance]): utterances or 
            segmented utterances per subject.
    """
    ax.margins(x=0)
    labels = list(utterances_per_subject.keys())
    num_labels = len(labels)

    for i, utterances in enumerate(utterances_per_subject.values()):
        durations = [[utterance.start, utterance.end - utterance.start] for utterance in utterances]
        ax.broken_barh(durations, ((num_labels - i - 1) * 5, 4.9), facecolors='tab:blue')

    ax.set_yticks([(num_labels - i - 1) * 5 + 2.5 for i in range(num_labels)])
    ax.set_yticklabels(labels)

    ax.set_title("Utterance Durations", fontsize=14, weight="bold")
    plt.xlabel("Timestamps")
