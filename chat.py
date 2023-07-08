"""
This module implements a chatbot using PyTorch for natural language processing.

The chatbot utilizes a pre-trained neural network model to generate responses based on user input.

Usage:
    - Run this module to start the chatbot conversation.
"""
import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

BOT_NAME = "Sasa"


def get_response(user_sentence):
    """
      Generate a response from the chatbot based on user input.

      Args:
          user_sentence (str): User input sentence.

      Returns:
          str: Generated response from the chatbot.
    """
    user_sentence = tokenize(user_sentence)
    bag_of_words_vector = bag_of_words(user_sentence, all_words)
    bag_of_words_vector = bag_of_words_vector.reshape(1, bag_of_words_vector.shape[0])
    bag_of_words_vector = torch.from_numpy(bag_of_words_vector)

    output = model(bag_of_words_vector)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])

    return "I do not understand..."


if __name__ == '__main__':
    print("Let's chat! (type 'quit' to exit)")
    while True:
        sentence = input('You: ')
        if sentence == "quit":
            break

        print(get_response(sentence))
