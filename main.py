import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy
import torch
import random
import json
from torch.utils.data import TensorDataset, DataLoader
import model

stemmer = LancasterStemmer()

with open('json file/intents.json') as file:
    data = json.load(file)

words = []
labels = []
docs_x = []
docs_y = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])

    if intent['tag'] not in labels:
        labels.append(intent['tag'])

words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))

labels = sorted(labels)

training = []
output = []

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stem(w.lower()) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)


training = numpy.array(training)
output = numpy.array(output)

train_set = TensorDataset(torch.Tensor(training), torch.Tensor(output))
train_loader = DataLoader(train_set, batch_size=4)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
learning_rate = 0.001

net = model.NeuralNet(len(training[0]), len(output[0])).to(device)
model.train(net, train_loader, 1000, device, learning_rate)
