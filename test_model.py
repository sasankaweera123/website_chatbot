"""
This module contains unit tests for the NeuralNet model.

The tests verify the forward pass functionality of the NeuralNet model.

Usage:
    - Run the tests using a test runner or the pytest command.
"""

import torch
from model import NeuralNet


def test_NeuralNet_forward():
    """
    Test case for the forward method of the NeuralNet model.

    The test verifies the shape of the output tensor when passing input data through the neural network.

    Raises:
        AssertionError: If the test assertion fails.
    """
    input_size = 10
    hidden_size = 8
    output_size = 5
    batch_size = 16

    model = NeuralNet(input_size, hidden_size, output_size)
    input_data = torch.randn(batch_size, input_size)
    output = model(input_data)

    assert output.shape == torch.Size([batch_size, output_size])
