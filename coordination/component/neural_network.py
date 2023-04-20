from typing import Any, List, Optional

import numpy as np
import pymc as pm
import pytensor.tensor as ptt

from coordination.common.activation_function import ActivationFunction


def add_bias(X: Any):
    if isinstance(X, np.ndarray):
        return np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    else:
        return ptt.concatenate([X, ptt.ones((X.shape[0], 1))], axis=1)


class NeuralNetworkParameters:

    def __init__(self):
        # One per layer
        self.weights: Optional[List[np.array]] = None

    def clear_values(self):
        self.weights = None


class NeuralNetwork:

    def __init__(self, uuid: str, num_hidden_layers: int, dim_hidden_layer: int, activation_function_name: str):
        self.uuid = uuid
        self.num_hidden_layers = num_hidden_layers
        self.dim_hidden_layer = dim_hidden_layer
        self.activation_function_name = activation_function_name

        self.parameters = NeuralNetworkParameters()

    @property
    def parameter_names(self) -> List[str]:
        # Fixed parameters, mean 0 and standard deviation 1. We don't fit them.
        return []

    @property
    def weights_name(self) -> str:
        return f"{self.uuid}_weights"

    def predict(self, input_data: np.ndarray) -> np.array:
        activation = ActivationFunction.from_numpy_name(self.activation_function_name)

        # Input layer
        a0 = activation(np.dot(add_bias(input_data), self.parameters.weights[0]))

        # Hidden layers
        h = a0
        for W in enumerate(self.parameters.weights[1]):
            h = activation(np.dot(add_bias(h), W))

        # Output layer
        output = np.dot(add_bias(h), self.parameters.weights[2])

        return output

    def update_pymc_model(self, input_data: Any, output_dim: int) -> Any:
        if self.parameters.weights is None:
            observed_weights = [None] * 3
        else:
            observed_weights = self.parameters.weights

        # Features + bias term
        input_layer_dim_in = input_data.shape[1] + 1
        input_layer_dim_out = self.dim_hidden_layer

        hidden_layer_dim_in = self.dim_hidden_layer + 1
        hidden_layer_dim_out = self.dim_hidden_layer

        output_layer_dim_in = self.dim_hidden_layer + 1
        output_layer_dim_out = output_dim

        input_layer = pm.Normal(f"{self.uuid}_in", size=(input_layer_dim_in, input_layer_dim_out),
                                observed=observed_weights[0])

        hidden_layers = pm.Normal(f"{self.uuid}_hidden", mu=0, sigma=1,
                                  size=(self.n, hidden_layer_dim_in, hidden_layer_dim_out),
                                  observed=observed_weights_f[1])

        # There's a bug in PyMC 5.0.2 that we cannot pass an argument with more dimensions than the
        # dimension of CustomDist. To work around it, I will join the layer dimension with the input dimension for
        # the hidden layers. Inside the logp function, I will reshape the layers back to their original 3 dimensions:
        # num_layers x in_dim x out_dim, so we can perform the feed-forward step.
        hidden_layers = pm.Deterministic(f"{self.f_nn_weights_name}_hidden_reshaped", hidden_layers.reshape(
            (num_hidden_layers * hidden_layer_dim_in, hidden_layer_dim_out)))

        output_layer = pm.Normal(f"{self.f_nn_weights_name}_out", size=(output_layer_dim_in, output_layer_dim_out),
                                 observed=observed_weights_f[2])

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
