import unittest

import numpy as np
import pymc as pm
import pytensor.tensor as ptt

from coordination.module.module import ModuleSamples
from coordination.module.transformation.mlp import MLP


class TestMLP(unittest.TestCase):
    def test_sample_transformation_no_hidden_layer(self):
        weights = [
            np.array(
                [
                    [1, 2, 3],
                    [4, 5, 6],
                ]
            )
        ]

        # dimension x time
        input_data = np.array(
            [
                [1, 3, 5, 7],
                [2, 4, 6, 8],
            ]
        )

        expected_value = np.array(
            [
                [9, 19, 29, 39],
                [12, 26, 40, 54],
                [15, 33, 51, 69],
            ]
        )

        pymc_model = pm.Model()
        mlp = MLP(
            uuid="mlp",
            pymc_model=pymc_model,
            output_dimension_size=3,
            input_samples=ModuleSamples([input_data]),
        )

        mlp.parameters.weights[0].value = weights[0]
        samples = mlp.draw_samples(seed=0, num_series=1)

        self.assertTrue(np.allclose(expected_value, samples.values[0]))

    def test_sample_transformation(self):
        weights = [
            np.array(
                [
                    [1, 2, 3],
                    [4, 5, 6],
                ]
            ),
            np.array(
                [
                    [1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9],
                ]
            ),
            np.array(
                [
                    [1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9],
                ]
            ),
            np.array(
                [
                    [1],
                    [2],
                    [3],
                ]
            ),
        ]

        # dimension x time
        input_data = np.array(
            [
                [1, 3, 5, 7],
                [2, 4, 6, 8],
            ]
        )

        expected_value = np.array([20304, 44280, 68256, 92232])

        pymc_model = pm.Model()
        mlp = MLP(
            uuid="mlp",
            pymc_model=pymc_model,
            output_dimension_size=1,
            num_hidden_layers=2,
            hidden_dimension_size=3,
            input_samples=ModuleSamples([input_data]),
        )

        mlp.parameters.weights[0].value = weights[0]
        mlp.parameters.weights[1].value = weights[1]
        mlp.parameters.weights[2].value = weights[2]
        mlp.parameters.weights[3].value = weights[3]
        samples = mlp.draw_samples(seed=0, num_series=1)

        self.assertTrue(np.allclose(expected_value, samples.values[0]))

    def test_rv_transformation_no_hidden_layer(self):
        weights = [
            ptt.as_tensor(
                np.array(
                    [
                        [1, 2, 3],
                        [4, 5, 6],
                    ]
                )
            )
        ]

        # dimension x time
        input_data = ptt.as_tensor(
            np.array(
                [
                    [1, 3, 5, 7],
                    [2, 4, 6, 8],
                ]
            )
        )

        expected_value = np.array(
            [
                [9, 19, 29, 39],
                [12, 26, 40, 54],
                [15, 33, 51, 69],
            ]
        )

        pymc_model = pm.Model()
        mlp = MLP(
            uuid="mlp",
            pymc_model=pymc_model,
            output_dimension_size=3,
            weight_random_variables=weights,
            input_random_variable=input_data,
        )
        mlp.create_random_variables()

        out_value = mlp.output_random_variable.eval()

        self.assertTrue(np.allclose(expected_value, out_value))

    def test_rv_transformation(self):
        weights = [
            ptt.as_tensor(
                np.array(
                    [
                        [1, 2, 3],
                        [4, 5, 6],
                    ]
                )
            ),
            ptt.as_tensor(
                np.array(
                    [
                        [1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9],
                    ]
                )
            ),
            ptt.as_tensor(
                np.array(
                    [
                        [1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9],
                    ]
                )
            ),
            ptt.as_tensor(
                np.array(
                    [
                        [1],
                        [2],
                        [3],
                    ]
                )
            ),
        ]

        # dimension x time
        input_data = ptt.as_tensor(
            np.array(
                [
                    [1, 3, 5, 7],
                    [2, 4, 6, 8],
                ]
            )
        )

        expected_value = np.array([20304, 44280, 68256, 92232])

        pymc_model = pm.Model()
        mlp = MLP(
            uuid="mlp",
            pymc_model=pymc_model,
            output_dimension_size=1,
            num_hidden_layers=2,
            hidden_dimension_size=3,
            weight_random_variables=weights,
            input_random_variable=input_data,
        )
        mlp.create_random_variables()

        out_value = mlp.output_random_variable.eval()

        print(out_value)

        self.assertTrue(np.allclose(expected_value, out_value))
