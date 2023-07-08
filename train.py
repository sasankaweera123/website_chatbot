"""
This module contains code for training a chatbot model using a neural network.

The script reads training data from a JSON file, preprocesses the data,
creates a PyTorch dataset and data loader,
defines the neural network model, and trains the model using cross-entropy
loss and the Adam optimizer.

Usage:
    - Run the script to train the chatbot model and save the trained model to a file.
"""

import json
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from nltk_utils import tokenize, stem, bag_of_words
from model import NeuralNet


class ChatDataset(Dataset):
    """
    Custom PyTorch dataset class for the chatbot training data.

    Args:
        x_data (np.ndarray): The input data.
        y_data (np.ndarray): The target labels.
    """

    def __init__(self, x_data, y_data):
        self.n_samples = len(x_data)
        self.x_data = x_data
        self.y_data = y_data

    def __getitem__(self, idx):
        return self.x_data[idx], torch.tensor(self.y_data[idx], dtype=torch.long)

    def __len__(self):
        return self.n_samples


def preprocess_data(intents):
    """
    Preprocesses the training data.

    Args:
        intents (dict): The JSON data containing training intents.

    Returns:
        tuple: A tuple containing the preprocessed all_words, tags, and data_points.
    """
    all_words = []
    tags = []
    data_points = []

    for intent in intents['intents']:
        tag = intent['tag']
        tags.append(tag)
        for pattern in intent['patterns']:
            words = tokenize(pattern)
            all_words.extend(words)
            data_points.append((words, tag))

    ignore_words = ['?', '.', '!']
    all_words = [stem(w) for w in all_words if w not in ignore_words]
    all_words = sorted(set(all_words))
    tags = sorted(set(tags))

    return all_words, tags, data_points


def create_dataset(all_words, tags, data_points):
    """
    Creates a PyTorch dataset from the preprocessed data.

    Args:
        all_words (list): The preprocessed list of all words.
        tags (list): The preprocessed list of tags.
        data_points (list): The preprocessed list of data points.

    Returns:
        ChatDataset: The PyTorch dataset.
    """
    x_train = []
    y_train = []

    for (pattern_sentence, tag) in data_points:
        bag = bag_of_words(pattern_sentence, all_words)
        x_train.append(bag)
        label = tags.index(tag)
        y_train.append(label)

    x_train = np.array(x_train)
    y_train = np.array(y_train, dtype=np.int64)

    dataset = ChatDataset(x_train, y_train)
    return dataset


def train_model():
    """
    Trains the chatbot model and saves the trained model to a file.
    """
    with open('intents.json') as file:
        intents = json.load(file)

    all_words, tags, data_points = preprocess_data(intents)
    dataset = create_dataset(all_words, tags, data_points)

    train_loader = DataLoader(dataset=dataset, batch_size=8, shuffle=True, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NeuralNet(len(all_words), 8, len(tags)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1000):
        loss = None  # Initialize loss variable with None

        for (words, labels) in train_loader:
            words = words.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(words)
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/1000], Loss: {loss.item():.4f}')

    data = {
        "model_state": model.state_dict(),
        "input_size": len(all_words),
        "output_size": len(tags),
        "hidden_size": 8,
        "all_words": all_words,
        "tags": tags
    }

    file_name = "data.pth"
    torch.save(data, file_name)

    print(f'Training complete. Model saved to {file_name}')


if __name__ == '__main__':
    train_model()
