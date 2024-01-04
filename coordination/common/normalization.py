import numpy as np


def normalize_serialized_data_per_subject_and_feature(
        data: np.ndarray,
        subject_indices: np.ndarray,
        num_subjects: int) -> np.ndarray:
    """
    Normalize data (feature x time) to have mean 0 and standard deviation 1 across time. The
    normalization is done individually per subject and feature.

    @param data: data to be normalized.
    @param subject_indices: an array of indices indicating the subject associated with the data
        value at each time point.
    @param num_subjects: number of subjects in the experiment.
    @return: normalized data.
    """

    normalized_values = np.zeros_like(data)
    for subject in range(num_subjects):
        # Get values for a specific subject across time.
        idx = np.array(subject_indices) == subject
        data_per_subject = data[:, idx]
        mean = np.mean(data_per_subject, axis=-1, keepdims=True)  # mean across time
        std = np.std(data_per_subject, axis=-1, keepdims=True)
        normalized_values[:, idx] = (data_per_subject - mean) / std

    return normalized_values


def normalize_serialized_data_per_feature(data: np.ndarray) -> np.ndarray:
    """
    Normalize data to have mean 0 and standard deviation 1 across time and subject. The
    normalization is done individually per feature.

    @param data: data to be normalized.
    @return: normalized data.
    """

    # Mean across time and subject
    mean = np.mean(data, axis=-1, keepdims=True)
    std = np.std(data, axis=-1, keepdims=True)
    normalized_values = (data - mean) / std
    return normalized_values
