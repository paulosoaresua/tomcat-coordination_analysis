from typing import Dict, List, Tuple

from src.components.speech.common import VocalicsComponent


def vocalics_component_to_series(vocalics_component: VocalicsComponent) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    series_a = {}
    series_b = {}

    for feature_name in vocalics_component.feature_names:
        series_a[feature_name] = []
        series_b[feature_name] = []

    for segmented_utterance in vocalics_component.series_a:
        for feature_name, average_value in segmented_utterance.average_vocalics.items():
            series_a[feature_name].append(average_value)
            series_a[feature_name].append(None)

    for segmented_utterance in vocalics_component.series_b:
        for feature_name, average_value in segmented_utterance.average_vocalics.items():
            series_b[feature_name].append(None)
            series_b[feature_name].append(average_value)

    return series_a, series_b
