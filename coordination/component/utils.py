from typing import Any, Callable

import numpy as np
import pymc as pm
import pytensor as pt
import pytensor.tensor as ptt

from coordination.common.activation_function import ActivationFunction


def add_bias(X: Any):
    if isinstance(X, np.ndarray):
        return np.concatenate([X, np.ones((1, X.shape[-1]))], axis=0)
    else:
        return ptt.concatenate([X, ptt.ones((1, X.shape[-1]))], axis=0)


def feed_forward_logp_f(input_data: Any,
                        input_layer_f: Any,
                        hidden_layers_f: Any,
                        output_layer_f: Any,
                        activation_function_number_f: ptt.TensorConstant):
    def forward(W, X, act_number):
        fn = ActivationFunction.from_pytensor_number(act_number.eval())
        z = pm.math.dot(W.transpose(), add_bias(X))
        return fn(z)

    if input_layer_f.shape.prod().eval() == 0:
        # Only transform the input data if a NN was specified
        return input_data

    hidden_dim = input_layer_f.shape[1]  # == f_nn_output_layer.shape[0]

    # Input layer activations
    activation = ActivationFunction.from_pytensor_number(activation_function_number_f.eval())
    a0 = activation(pm.math.dot(input_layer_f.transpose(), add_bias(input_data)))

    # Reconstruct hidden layers as a 3 dimensional tensor, where the first dimension represents the number of layers.
    num_hidden_layers = ptt.cast(hidden_layers_f.shape[0] / (hidden_dim + 1), "int32")
    hidden_layers_f = hidden_layers_f.reshape((num_hidden_layers, hidden_dim + 1, hidden_dim))

    # Feed-Forward through the hidden layers
    res, updates = pt.scan(forward,
                           outputs_info=a0,
                           sequences=[hidden_layers_f],
                           non_sequences=[activation_function_number_f])

    h = res[-1]

    # Final result. We don't apply any activation to the final layer not to squeeze the values.
    out = pm.math.dot(output_layer_f.transpose(), add_bias(h))

    return out


def feed_forward_random_f(input_data: np.ndarray,
                          input_layer_f: np.ndarray,
                          hidden_layers_f: np.ndarray,
                          output_layer_f: np.ndarray,
                          activation: Callable):
    if len(input_layer_f) == 0:
        return input_data

    hidden_dim = input_layer_f.shape[1]  # == f_nn_output_layer.shape[0]

    # Input layer activations
    a0 = activation(np.dot(input_layer_f.transpose(), add_bias(input_data)))

    # Reconstruct hidden layers as a 3 dimensional tensor, where the first dimension represents the number of layers.
    num_hidden_layers = int(hidden_layers_f.shape[0] / (hidden_dim + 1))
    hidden_layers_f = hidden_layers_f.reshape((num_hidden_layers, hidden_dim + 1, hidden_dim))

    # Feed-Forward through the hidden layers
    h = a0
    for W in hidden_layers_f:
        h = activation(np.dot(W.transpose(), add_bias(h)))

    # Output layer activation.
    out = activation(np.dot(output_layer_f.transpose(), add_bias(h)))

    return out