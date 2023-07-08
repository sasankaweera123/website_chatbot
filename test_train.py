import pytest
import torch
from train import tokenize, stem, bag_of_words, ChatDataset, NeuralNet


def test_tokenize():
    sentence = "Hello, how are you?"
    expected_tokens = ["Hello", ",", "how", "are", "you", "?"]
    assert tokenize(sentence) == expected_tokens


def test_stem():
    word = "running"
    expected_stemmed_word = "run"
    assert stem(word) == expected_stemmed_word


def test_bag_of_words():
    tokenized_sentence = ["Hello", "world", "!"]
    all_words = ["hello", "world", "goodbye"]
    expected_bag = torch.tensor([1, 1, 0], dtype=torch.float32)
    output_bag = bag_of_words(tokenized_sentence, all_words)
    assert torch.all(torch.eq(torch.from_numpy(output_bag), expected_bag))


def test_ChatDataset():
    dataset = ChatDataset()
    assert len(dataset) > 0


def test_NeuralNet():
    input_size = 10
    hidden_size = 8
    output_size = 5
    model = NeuralNet(input_size, hidden_size, output_size)
    assert isinstance(model, torch.nn.Module)
