import torch
from train import train_model


def test_train_model():
    train_model()
    FILE = "data.pth"
    data = torch.load(FILE)

    assert 'model_state' in data
    assert 'input_size' in data
    assert 'output_size' in data
    assert 'hidden_size' in data
    assert 'all_words' in data
    assert 'tags' in data

    model_state = data['model_state']
    input_size = data['input_size']
    output_size = data['output_size']
    hidden_size = data['hidden_size']
    all_words = data['all_words']
    tags = data['tags']

    assert isinstance(model_state, dict)
    assert isinstance(input_size, int)
    assert isinstance(output_size, int)
    assert isinstance(hidden_size, int)
    assert isinstance(all_words, list)
    assert isinstance(tags, list)

    assert len(model_state) > 0
    assert len(all_words) > 0
    assert len(tags) > 0
