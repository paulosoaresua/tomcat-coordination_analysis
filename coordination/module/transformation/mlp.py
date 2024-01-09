from __future__ import annotations

from copy import deepcopy
from typing import List, Optional

import numpy as np
import pymc as pm
import pytensor.tensor as ptt

from coordination.module.constants import (DEFAULT_MLP_ACTIVATION,
                                           DEFAULT_MLP_HIDDEN_DIMENSION_SIZE,
                                           DEFAULT_MLP_MEAN_WEIGHTS,
                                           DEFAULT_MLP_NUM_HIDDEN_LAYERS,
                                           DEFAULT_MLP_SD_WEIGHTS)
from coordination.module.module import ModuleParameters, ModuleSamples
from coordination.module.parametrization2 import (NormalParameterPrior,
                                                  Parameter)
from coordination.module.transformation.transformation import Transformation

ACTIVATIONS = {
    "linear": lambda x: x,
    "relu": lambda x: np.maximum(x, 0)
    if isinstance(x, np.ndarray)
    else pm.math.maximum(x, 0),
    "tanh": lambda x: np.tanh(x) if isinstance(x, np.ndarray) else pm.math.tanh(x),
}


class MLP(Transformation):
    """
    This class represents a Multi-Layer Perceptron transformation.
    """

    def __init__(
            self,
            uuid: str,
            pymc_model: pm.Model,
            output_dimension_size: int,
            mean_w0: float = DEFAULT_MLP_MEAN_WEIGHTS,
            sd_w0: float = DEFAULT_MLP_SD_WEIGHTS,
            num_hidden_layers: int = DEFAULT_MLP_NUM_HIDDEN_LAYERS,
            hidden_dimension_size: int = DEFAULT_MLP_HIDDEN_DIMENSION_SIZE,
            activation: str = DEFAULT_MLP_ACTIVATION,
            input_samples: Optional[ModuleSamples] = None,
            input_random_variable: Optional[pm.Distribution] = None,
            output_random_variable: Optional[pm.Distribution] = None,
            weight_random_variables: Optional[List[pm.Distribution]] = None,
            axis: int = 0,
            weights: Optional[List[np.ndarray]] = None,
    ):
        """
        Creates an MLP.

        We treat the weights as parameters that can be fixed for sampling and cleared for
        inference instead of a non-parameter random variable as we assume a transformation
        is not expected to be sampled during a call to draw_sampled but given.

        The values set to the weights must have dimensions as described below:
        - weights[0]: input_dim x hidden_dimension_size
        - weights[1:-1]: num_hidden_layers x hidden_dimension_size x hidden_dimension_size
        - weights[-1]: hidden_dimension_size x output_dimension_size

        input_dim is defined by the shape of input_samples during a call to draw_samples and
        input_random_variable during a call to create_random_variables at the informed axis.

        @param uuid: string uniquely identifying the transformation in the model.
        @param pymc_model: a PyMC model instance where modules are to be created at.
        @param output_dimension_size: the number of dimensions in the transformed object.
        @param mean_w0: mean of the normal prior distribution of the weights.
        @param sd_w0: standard deviation of the norma prior distribution of the weights.
        @param num_hidden_layers: number of hidden layers in the neural network.
        @param hidden_dimension_size: size of the dimension of each hidden layer, i.e., the number
            of activation units per hidden layer.
        @param activation: name of the activation function. One of: linear, relu, or tanh.
        @param input_samples: samples transformed in a call to draw_samples. This variable must be
            set before such a call.
        @param input_random_variable: random variable to be transformed in a call to
            create_random_variables. This variable must be set before such a call.
        @param output_random_variable: transformed random variable. If set, not transformation is
            performed in a call to create_random_variables.
        @param weight_random_variables: a list of random variables representing the weights in
            each layer of the neural network to be used in a call to create_random_variables. If
            not set, they will be created in such a call.
        @param axis: axis to apply the transformation.
        @param weights: values for the weights of the MLP. It needs to be given for sampling but
            not for inference if it needs to be inferred. If not provided now, it can be set later
            via the module parameters variable.
        """
        super().__init__(
            uuid=uuid,
            pymc_model=pymc_model,
            parameters=MLPParameters(uuid, num_hidden_layers, mean_w0, sd_w0),
            input_samples=input_samples,
            input_random_variable=input_random_variable,
            output_random_variable=output_random_variable,
            observed_values=None,
            axis=axis,
        )
        if weights is not None:
            for i, w in enumerate(self.parameters.weights):
                w.value = weights[i]

        if num_hidden_layers < 0:
            raise ValueError(
                f"The number of layers ({num_hidden_layers}) must be a non-negative "
                f"number."
            )

        if activation not in ACTIVATIONS:
            raise ValueError(
                f"The activations ({activation}) must be one of "
                f"{list(ACTIVATIONS.keys())}."
            )

        self.output_dimension_size = output_dimension_size
        self.num_hidden_layers = num_hidden_layers
        self.hidden_dimension_size = hidden_dimension_size
        self.activation = activation
        self.weight_random_variables = weight_random_variables

    @property
    def num_layers(self) -> int:
        """
        Gets the number of layers in the model.

        @return: number of layers in the model.
        """
        # TODO: I believe this should be self.num_hidden_layers + 1 always
        return 1 if self.num_hidden_layers == 0 else self.num_hidden_layers + 2

    def draw_samples(self, seed: Optional[int], num_series: int) -> ModuleSamples:
        """
        Transforms input samples with a neural network.

        @param seed: random seed for reproducibility.
        @param num_series: how many series of samples to generate.
        @return: transformed samples for each series.
        """
        super().draw_samples(seed, num_series)

        self._validate_module_for_sampling()

        transformed_samples_values = []
        for i, sampled_series in enumerate(self.input_samples.values):
            a = sampled_series
            for layer in range(self.num_layers):
                weights = self.parameters.weights[layer].value
                z = np.tensordot(
                    a, weights, axes=[(self.axis,), (0,)]
                ).swapaxes(-2, -1)

                if layer < self.num_layers - 1:
                    a = ACTIVATIONS[self.activation](z)
                else:
                    # Activation is not applied to the output layer
                    a = z

            transformed_samples_values.append(a)

        transformed_samples = deepcopy(self.input_samples)
        transformed_samples.values = (
            np.array(transformed_samples_values)
            if isinstance(self.input_samples.values, np.ndarray)
            else transformed_samples_values
        )
        return transformed_samples

    def _validate_module_for_sampling(self):
        """
        Performs a series of validations on the weight values and dimensions before proceeding
        with the transformation over samples.

        @raise ValueError: if weights are undefined or have incompatible dimensions.
        """

        # Weights in the input layer
        if self.parameters.weights[0].value is None:
            raise ValueError(
                f"The value of {self.parameters.weights[0].uuid} is undefined."
            )

        next_dim = (
            self.output_dimension_size
            if self.num_hidden_layers == 0
            else self.hidden_dimension_size
        )
        input_dims = (self.input_samples.values[0].shape[self.axis], next_dim)
        if self.parameters.weights[0].value.shape != input_dims:
            raise ValueError(
                f"Dimensions of {self.parameters.weights[0].uuid} "
                f"{self.parameters.weights[0].value.shape} must be {input_dims}."
            )

        if self.num_hidden_layers > 0:
            # Hidden layer
            for h in range(1, self.num_hidden_layers + 1):
                if self.parameters.weights[h].value is None:
                    raise ValueError(
                        f"The value of {self.parameters.weights[h].uuid} is "
                        f"undefined."
                    )

                hidden_dims = (self.hidden_dimension_size, self.hidden_dimension_size)
                if self.parameters.weights[h].value.shape != hidden_dims:
                    raise ValueError(
                        f"Dimensions of {self.parameters.weights[h].uuid} "
                        f"{self.parameters.weights[h].value.shape} must be {hidden_dims}."
                    )

            # Output layer
            if self.parameters.weights[-1].value is None:
                raise ValueError(
                    f"The value of {self.parameters.weights[-1].uuid} is " f"undefined."
                )

            out_dims = (self.hidden_dimension_size, self.output_dimension_size)
            if self.parameters.weights[-1].value.shape != out_dims:
                raise ValueError(
                    f"Dimensions of {self.parameters.weights[-1].uuid} "
                    f"{self.parameters.weights[-1].value.shape} must be "
                    f"{out_dims}."
                )

    def create_random_variables(self):
        """
        Creates neural network weights as random variables in a PyMC model and transforms the input
        random variable.
        """
        super().create_random_variables()

        if self.weight_random_variables is None:
            self.weight_random_variables = [None] * self.num_layers

        self._validate_module_for_random_variable_creation()

        with self.pymc_model:
            if self.weight_random_variables[0] is None:
                next_dim = (
                    self.output_dimension_size
                    if self.num_hidden_layers == 0
                    else self.hidden_dimension_size
                )
                self.weight_random_variables[0] = pm.Normal(
                    name=self.parameters.weights[0].uuid,
                    mu=self.parameters.weights[0].prior.mean,
                    sigma=self.parameters.weights[0].prior.sd,
                    size=(self.input_random_variable.shape[self.axis], next_dim),
                    observed=self.parameters.weights[0].value,
                )

            z = ptt.tensordot(
                self.weight_random_variables[0],
                self.input_random_variable,
                axes=[[0], [self.axis]],
            )

            if self.num_hidden_layers > 0:
                a = ACTIVATIONS[self.activation](z)
                for h in range(1, self.num_hidden_layers + 1):
                    if self.weight_random_variables[h] is None:
                        self.weight_random_variables[h] = pm.Normal(
                            name=f"{self.parameters.weights[h].uuid}_{h}",
                            mu=self.parameters.weights[h].prior.mean,
                            sigma=self.parameters.weights[h].prior.sd,
                            size=(
                                self.hidden_dimension_size,
                                self.hidden_dimension_size,
                            ),
                            observed=self.parameters.weights[h].value,
                        )

                    z = ptt.tensordot(
                        self.weight_random_variables[h], a, axes=[[0], [self.axis]]
                    )
                    a = ACTIVATIONS[self.activation](z)

                if self.weight_random_variables[-1] is None:
                    self.weight_random_variables[-1] = pm.Normal(
                        name=self.parameters.weights[-1].uuid,
                        mu=self.parameters.weights[-1].prior.mean,
                        sigma=self.parameters.weights[-1].prior.sd,
                        size=(self.hidden_dimension_size, self.output_dimension_size),
                        observed=self.parameters.weights[-1].value,
                    )

                z = ptt.tensordot(
                    self.weight_random_variables[-1], a, axes=[[0], [self.axis]]
                )

            self.output_random_variable = z

    def _validate_module_for_random_variable_creation(self):
        """
        Performs a series of validations on the weight random variables if they were previously
        defined to check for dimension compatibility.

        @raise ValueError: if weight variables have incompatible dimensions.
        """

        if len(self.weight_random_variables) != self.num_layers:
            raise ValueError(
                f"The number of weights in weight_random_variables "
                f"({len(self.weight_random_variables)}) doesn't match the "
                f"number of layers in the model ({self.num_layers})."
            )

        if self.weight_random_variables[0] is not None:
            next_dim = (
                self.output_dimension_size
                if self.num_hidden_layers == 0
                else self.hidden_dimension_size
            )
            input_dims = [self.input_random_variable.shape[self.axis], next_dim]
            if ptt.neq(self.weight_random_variables[0].shape, input_dims).any().eval():
                raise ValueError(
                    f"Dimensions of weight_random_variables[0] "
                    f"({self.weight_random_variables[0].shape.eval()}) must be "
                    f"{[dim if isinstance(dim, int) else dim.eval() for dim in input_dims]}."
                )

        if self.num_hidden_layers > 0:
            hidden_dims = [self.hidden_dimension_size, self.hidden_dimension_size]
            for h in range(1, self.num_hidden_layers + 1):
                if self.weight_random_variables[h] is not None:
                    if (
                            ptt.neq(self.weight_random_variables[h].shape, hidden_dims)
                                    .any()
                                    .eval()
                    ):
                        raise ValueError(
                            f"Dimensions of weight_random_variables[{h}] "
                            f"({self.weight_random_variables[h].shape.eval()}) must be "
                            f"{hidden_dims}."
                        )

            if self.weight_random_variables[-1] is not None:
                out_dims = [self.hidden_dimension_size, self.output_dimension_size]
                if (
                        ptt.neq(self.weight_random_variables[-1].shape, out_dims)
                                .any()
                                .eval()
                ):
                    raise ValueError(
                        f"Dimensions of weight_random_variables[-1] "
                        f"({self.weight_random_variables[-1].shape.eval()}) must be "
                        f"{out_dims}."
                    )


###################################################################################################
# AUXILIARY CLASSES
###################################################################################################


class MLPParameters(ModuleParameters):
    """
    This class stores values and priors of the weights of an MLP.
    """

    def __init__(
            self, module_uuid: str, num_hidden_layers: int, mean_w0: float, sd_w0: float
    ):
        """
        Creates an object to store MLP parameter info.

        @param module_uuid: unique ID of the latent component module.
        @param num_hidden_layers: number of hidden layers in the model.
        @param mean_mean_w0: mean of the prior of the weights.
        @param sd_mean_w0: standard deviation of the prior of weights.
        """
        super().__init__()

        self.weights = []
        num_layers = 1 if num_hidden_layers == 0 else num_hidden_layers + 2
        for layer in range(num_layers):
            if layer == 0:
                suffix = "in"
            elif layer == num_layers - 1:
                suffix = "out"
            else:
                suffix = f"h{layer}"

            self.weights.append(
                Parameter(
                    uuid=f"{module_uuid}_weights_{suffix}",
                    prior=NormalParameterPrior(mean_w0, sd_w0),
                )
            )
