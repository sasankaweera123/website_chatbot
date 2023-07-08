"""
This module defines a neural network model using the torch.nn module from PyTorch.

The model is implemented as a class named NeuralNet, which inherits from the nn.Module base class.
The NeuralNet class represents a feedforward neural network with linear layers and ReLU activation.

Usage:
    Instantiate an instance of the NeuralNet class with the desired
    input size, hidden size, and number of classes.
    Call the forward() method on the instance to perform the forward pass of the neural network.

Example:
    model = NeuralNet(input_size=100, hidden_size=50, num_classes=10)
    output = model.forward(input_tensor)
"""

from torch import nn


class NeuralNet(nn.Module):
    """
        A feedforward neural network model implemented using the torch.nn module from PyTorch.

        The model consists of linear layers and ReLU activation functions.
    """
    def __init__(self, input_size, hidden_size, num_classes):
        """
        Initialize the NeuralNet class.

        Args:
            input_size (int): Size of the input layer.
            hidden_size (int): Size of the hidden layer.
            num_classes (int): Number of classes in the output layer.
        """
        super().__init__()
        self.line_1 = nn.Linear(input_size, hidden_size)
        self.line_2 = nn.Linear(hidden_size, hidden_size)
        self.line_3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, input_tensor):
        """
        Perform the forward pass of the neural network.

        Args:
            input_tensor (torch.Tensor): Input tensor to the neural network.

        Returns:
            torch.Tensor: Output tensor from the neural network.
        """
        out = self.line_1(input_tensor)
        out = self.relu(out)
        out = self.line_2(out)
        out = self.relu(out)
        out = self.line_3(out)
        # no activation and no softmax at the end
        return out

    def predict(self, input_tensor):
        """
        Perform the prediction using the neural network.

        Args:
            input_tensor (torch.Tensor): Input tensor to the neural network.

        Returns:
            torch.Tensor: Predicted output tensor from the neural network.
        """
        output = self.forward(input_tensor)
        # Apply softmax or other appropriate post-processing
        return output
