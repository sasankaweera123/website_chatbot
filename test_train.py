"""
This module contains unit tests for the `preprocess_data` and `create_dataset`
functions in the `train` module.

The tests verify that the functions produce the expected results based on the
provided `intents` fixture.
"""

import pytest
import torch

from train import preprocess_data, create_dataset


@pytest.fixture
def intents():
    """
    Fixture to provide the intents data structure.

    :return:
        The intents data structure.
    """
    return {
        "intents": [
            {
                "tag": "greeting",
                "patterns": ["Hello", "Hi", "Hey"],
                "responses": ["Hi there!", "Hello!", "Hey! How can I help you?"]
            },
            {
                "tag": "goodbye",
                "patterns": ["Bye", "Goodbye", "See you later"],
                "responses": ["Goodbye!", "See you later!", "Take care!"]
            }
        ]
    }


def test_preprocess_data(intents):
    """
    Test the preprocess_data function.
    """
    all_words, tags, data_points = preprocess_data(intents)
    expected_all_words = ['bye', 'goodby', 'hello', 'hey', 'hi', 'later', 'see', 'you']
    assert len(all_words) == len(expected_all_words)
    assert all(word in all_words for word in expected_all_words)
    assert len(tags) == 2
    assert len(data_points) == 6


def test_create_dataset(intents):
    """
    Test the create_dataset function.
    """
    all_words, tags, data_points = preprocess_data(intents)
    dataset = create_dataset(all_words, tags, data_points)
    assert len(dataset) == 6
    assert len(dataset[0][0]) == len(all_words)
    assert dataset[0][1].dtype == torch.long
