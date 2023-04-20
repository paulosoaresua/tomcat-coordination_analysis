from typing import Any, List, Optional

import numpy as np
import pymc as pm
import pytensor as pt
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

        activation = ActivationFunction.from_pytensor_name(self.activation_function_name)

        # Input activation
        input_layer = pm.Normal(f"{self.weights_name}_in", size=(input_layer_dim_in, input_layer_dim_out),
                                observed=observed_weights[0])
        a0 = activation(pm.math.dot(add_bias(input_data), input_layer))

        # Hidden layers activations
        hidden_layers = pm.Normal(f"{self.weights_name}_hidden", mu=0, sigma=1,
                                      size=(self.num_hidden_layers, hidden_layer_dim_in, hidden_layer_dim_out),
                                      observed=observed_weights[1])

        def forward(W, X, act_number):
            fn = ActivationFunction.from_pytensor_number(act_number.eval())
            z = pm.math.dot(add_bias(X), W)
            return fn(z)

        # Feed-Forward through the hidden layers
        res, updates = pt.scan(forward,
                               outputs_info=a0,
                               sequences=[hidden_layers],
                               non_sequences=[ActivationFunction.NAME_TO_NUMBER[self.activation_function_name]])

        h = res[-1]

        # Output value
        output_layer = pm.Normal(f"{self.weights_name}_out", size=(output_layer_dim_in, output_layer_dim_out),
                                 observed=observed_weights[2])

        output = pm.math.dot(add_bias(h), output_layer)

        return output, input_layer, hidden_layers, output_layer
