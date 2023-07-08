"""
This module contains unit tests for the nltk_utils module.

The tests verify the functionality of the tokenize, stem, and bag_of_words functions.

Usage:
    - Run the tests using a test runner or the pytest command.
"""

import numpy as np
from nltk_utils import tokenize, stem, bag_of_words


def test_tokenize():
    """
    Test case for the tokenize function.

    The test checks if the tokenize function correctly tokenizes a sentence into individual words.

    Raises:
        AssertionError: If the test assertion fails.
    """
    sentence = "Hello, how are you?"
    expected_tokens = ["Hello", ",", "how", "are", "you", "?"]
    assert tokenize(sentence) == expected_tokens


def test_stem():
    """
    Test case for the stem function.

    The test checks if the stem function correctly stems a word using the Porter stemming algorithm.

    Raises:
        AssertionError: If the test assertion fails.
    """
    word = "running"
    expected_stemmed_word = "run"
    assert stem(word) == expected_stemmed_word


def test_bag_of_words():
    """
    Test case for the bag_of_words function.

    The test checks if the bag_of_words function correctly creates a bag of words representation for a tokenized sentence.

    Raises:
        AssertionError: If the test assertion fails.
    """
    tokenized_sentence = ["Hello", "world", "!"]
    all_words = ["hello", "world", "goodbye"]
    expected_bag = np.array([1, 1, 0], dtype=np.float32)
    assert np.array_equal(bag_of_words(tokenized_sentence, all_words), expected_bag)
