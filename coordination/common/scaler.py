import numpy as np

from coordination.common.normalization import (
    NORMALIZATION_PER_FEATURE, NORMALIZATION_PER_SUBJECT_AND_FEATURE)


class Scaler:
    """
    Creates a scaler to normalize data.
    """

    def __init__(self, normalization_method: str):
        """
        Creates a scaler.

        @param normalization_method: normalization method to be used.
        """
        self.normalization_method = normalization_method
        self._mean = None
        self._std = None

    def fit(self, X: np.ndarray):
        """
        Fits the data by computing and storing means and standard deviation.

        @param X: data.
        @return:
        """
        if X is None:
            return

        if self.normalization_method == NORMALIZATION_PER_FEATURE:
            # Mean across time and subject
            self._mean = np.mean(X, axis=(0, -1), keepdims=True)
            self._std = np.std(X, axis=(0, -1), keepdims=True)

        if self.normalization_method == NORMALIZATION_PER_SUBJECT_AND_FEATURE:
            self._mean = np.mean(X, axis=-1, keepdims=True)  # mean across time
            self._std = np.std(X, axis=-1, keepdims=True)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transforms the data with the stored mean and standard deviation from a previous fit.

        @param X: data.
        @return: normalized data.
        """
        if X is None:
            return X

        if self._mean is None:
            raise ValueError("Scaler hasn't been fit yet.")

        return (np.array(X) - self._mean) / self._std


class SerialScaler(Scaler):
    """
    Creates a scaler to normalize serial data.
    """

    def __init__(self, normalization_method: str):
        """
        Creates a scaler.

        @param normalization_method: normalization method to be used.
        """
        super().__init__(normalization_method)
        self._mean = None
        self._std = None

    def fit(
        self,
        X: np.ndarray,
        subject_indices: np.ndarray = None,
        num_subjects: int = None,
    ):
        """
        Fits the data by computing and storing means and standard deviation.

        @param X: data.
        @param subject_indices: an array of indices indicating the subject associated with the data
                value at each time point.
            @param num_subjects: number of subjects in the experiment.
        @return:
        """
        if X is None:
            return

        if self.normalization_method == NORMALIZATION_PER_FEATURE:
            # Mean across time and subject
            self._mean = np.mean(X, axis=-1, keepdims=True)
            self._std = np.std(X, axis=-1, keepdims=True)

        if self.normalization_method == NORMALIZATION_PER_SUBJECT_AND_FEATURE:
            self._mean = np.zeros((num_subjects, X.shape[0]))
            self._std = np.zeros((num_subjects, X.shape[0]))
            for subject in range(num_subjects):
                # Get values for a specific subject across time.
                idx = np.array(subject_indices[: X.shape[-1]]) == subject
                data_per_subject = np.array(X)[:, idx]
                self._mean[subject] = np.mean(
                    data_per_subject, axis=-1
                )  # mean across time
                self._std[subject] = np.std(data_per_subject, axis=-1)

    def transform(
        self, X: np.ndarray, subject_indices: np.ndarray = None
    ) -> np.ndarray:
        """
        Transforms the data with the stored mean and standard deviation from a previous fit.

        @param X: data.
        @param subject_indices: an array of indices indicating the subject associated with the data
                value at each time point.
        @return: normalized data.
        """
        if X is None:
            return None

        if self._mean is None:
            raise ValueError("Scaler hasn't been fit yet.")

        if self.normalization_method == NORMALIZATION_PER_FEATURE:
            return (np.array(X) - self._mean) / self._std

        if self.normalization_method == NORMALIZATION_PER_SUBJECT_AND_FEATURE:
            # Get values for a specific subject across time.
            m = self._mean[subject_indices].T
            s = self._std[subject_indices].T
            return (X - m) / s
