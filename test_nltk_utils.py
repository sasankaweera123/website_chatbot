import numpy as np
from nltk_utils import tokenize, stem, bag_of_words


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
    expected_bag = np.array([1, 1, 0], dtype=np.float32)
    assert np.array_equal(bag_of_words(tokenized_sentence, all_words), expected_bag)
