"""
This module provides utility functions for natural language processing using the nltk library.

The module includes functions for tokenizing sentences, stemming words, and creating a bag of words representation.

Usage:
    - Import the module or specific functions as needed.
    - Call the functions with the appropriate arguments to perform the desired natural language processing tasks.
"""

import nltk
from nltk.stem.porter import PorterStemmer
import numpy as np

# nltk.download('punkt')

stemmer = PorterStemmer()


def tokenize(sentence):
    """
    Tokenizes a sentence into individual words.

    Args:
        sentence (str): The input sentence to be tokenized.

    Returns:
        list: A list of tokenized words.
    """
    return nltk.word_tokenize(sentence)


def stem(word):
    """
    Stems a word using the Porter stemming algorithm.

    Args:
        word (str): The input word to be stemmed.

    Returns:
        str: The stemmed word.
    """
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, all_words):
    """
    Creates a bag of words representation for a tokenized sentence.

    Args:
        tokenized_sentence (list): A list of tokenized words in the sentence.
        all_words (list): A list of all unique words in the corpus.

    Returns:
        np.ndarray: A numpy array representing the bag of words.
    """
    tokenized_sentence = [stem(word) for word in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, word in enumerate(all_words):
        if word in tokenized_sentence:
            bag[idx] = 1.0

    return bag
