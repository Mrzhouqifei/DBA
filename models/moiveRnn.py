import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from settings import *

class Model(torch.nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocabLimit):
        super(Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(vocabLimit + 1, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.linearOut = nn.Linear(hidden_dim, 2)

    def forward(self, inputs):
        hidden = self.init_hidden()
        embeddings = self.embeddings(inputs).view(len(inputs), 1, -1)
        var_embeddings = Variable(embeddings, requires_grad=True)
        lstm_out, lstm_h = self.lstm(var_embeddings, hidden)
        x = lstm_out[-1]
        x = self.linearOut(x)
        # x = F.log_softmax(x, dim=1)
        return x, var_embeddings

    def init_hidden(self):
        return (Variable(torch.zeros(1, 1, self.hidden_dim)).to(device),
                Variable(torch.zeros(1, 1, self.hidden_dim)).to(device))
#
# class Model(torch.nn.Module):
#     def __init__(self, embedding_dim, hidden_dim):
#         super(Model, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.lstm = nn.LSTM(embedding_dim, hidden_dim)
#         self.linearOut = nn.Linear(hidden_dim, 2)
#
#     def forward(self, inputs):
#         hidden = self.init_hidden()
#         lstm_out, lstm_h = self.lstm(inputs, hidden)
#         x = lstm_out[-1]
#         x = self.linearOut(x)
#         x = F.log_softmax(x, dim=1)
#         return x
#
#     def init_hidden(self):
#         return (Variable(torch.zeros(1, 1, self.hidden_dim)).to(device),
#                 Variable(torch.zeros(1, 1, self.hidden_dim)).to(device))

# class Embedding(torch.nn.Module):
#     def __init__(self,vocabLimit, embedding_dim):
#         super(Embedding, self).__init__()
#         self.embeddings = nn.Embedding(vocabLimit + 1, embedding_dim)
#
#     def forward(self, inputs):
#         embeddings = self.embeddings(inputs).view(len(inputs), 1, -1)
#         return embeddings
