from typing import Any, List

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
        axs[i].set_title(feature_name.capitalize(), fontsize=14)
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
