from typing import Any, List

import numpy as np
import pymc as pm
import pytensor.tensor as ptt

from coordination.common.activation_function import ActivationFunction


class NeuralNetworkParameters:

    def __init__(self, num_layers: int):
        # One per layer
        self.weights: List[np.array] = [None] * num_layers

    def clear_values(self):
        self.weights = [None] * len(self.weights)


class NeuralNetwork:

    def __init__(self, uuid: str, units_per_layer: List[int], activations: List[str]):
        self.uuid = uuid
        self.units_per_layer = units_per_layer
        self.activations = activations

        self.parameters = NeuralNetworkParameters(len(units_per_layer))

    @property
    def parameter_names(self) -> List[str]:
        # Fixed parameters, mean 0 and standard deviation 1. We don't fit them.
        return []

    @property
    def weights_name(self) -> str:
        return f"{self.uuid}_weights"

    def predict(self, input_data: np.ndarray) -> np.array:
        X = input_data
        for layer, W in enumerate(self.parameters.weights):
            activation = ActivationFunction.from_name(self.activations[layer])
            X = activation(np.dot(X, W))

        return X

    def update_pymc_model(self, input_data: Any) -> Any:
        weights = []
        outputs = []
        X = input_data
        # Number of features + bias
        size_in = input_data.shape[-1] + 1
        for layer, num_units in enumerate(self.units_per_layer):
            activation = ActivationFunction.from_name(self.activations[layer])
            W = pm.Normal(f"{self.weights_name}_{layer}", mu=0, sigma=1, size=(size_in, num_units),
                                observed=self.parameters.weights[layer])
            X = pm.Deterministic(f"a_{layer}", activation(pm.math.dot(NeuralNetwork._add_bias(X), W)))
            weights.append(W)
            outputs.append(X)
            # Units + bias
            size_in = num_units + 1

        return weights, outputs

    @staticmethod
    def _add_bias(X: Any):
        return pm.math.concatenate([X, ptt.ones((X.shape[0], 1))], axis=1)
