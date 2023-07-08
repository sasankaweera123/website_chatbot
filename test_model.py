import torch
from model import NeuralNet


def test_NeuralNet_forward():
    input_size = 10
    hidden_size = 8
    output_size = 5
    batch_size = 16

    model = NeuralNet(input_size, hidden_size, output_size)
    input_data = torch.randn(batch_size, input_size)
    output = model(input_data)

    assert output.shape == torch.Size([batch_size, output_size])
