import matplotlib.pyplot as plt

from .common import VocalicsComponent


def plot_vocalics_series(vocalics_component: VocalicsComponent) -> None:
    """Plot vocalics series, with x axis the end timestamp of utterance and y axis the average value of utterance

    Args:
        vocalics_component (VocalicsComponent): vocalics component object
    """
    # prepare data for plot
    series_A_data = {}
    series_B_data = {}

    for feature_name in vocalics_component.feature_names:
        series_A_data[feature_name] = {}
        series_B_data[feature_name] = {}

    for segmented_utterance in vocalics_component.series_a:
        for feature_name, average_value in segmented_utterance.average_vocalics.items():
            series_A_data[feature_name][segmented_utterance.end] = average_value

    for segmented_utterance in vocalics_component.series_b:
        for feature_name, average_value in segmented_utterance.average_vocalics.items():
            series_B_data[feature_name][segmented_utterance.end] = average_value

    # plot
    fig, axes = plt.subplots(len(series_A_data), figsize=(25, 10))

    for i, feature_name in enumerate(series_A_data.keys()):
        axes[i].set_ylabel("average value")
        axes[i].set_title(feature_name, fontsize=16)
        axes[i].scatter(series_A_data[feature_name].keys(), series_A_data[feature_name].values(), color='r', label="series A")
        axes[i].scatter(series_B_data[feature_name].keys(), series_B_data[feature_name].values(), color='b', label="series B")

    axes[len(series_A_data.keys()) - 1].set_xlabel("Timestamp")

    fig.suptitle("Vocalics series", fontsize=32)
    plt.legend()
    plt.show()
