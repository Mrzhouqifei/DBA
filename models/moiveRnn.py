import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from settings import *

class Model(torch.nn.Module):
    """
    we need to load init embed weights, because var_embeddings can not be trained!
    """
    def __init__(self, embedding_dim, hidden_dim, vocabLimit):
        super(Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(vocabLimit + 1, embedding_dim)
        # self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        # self.linearOut = nn.Linear(hidden_dim, 2)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)
        self.linearOut = nn.Linear(hidden_dim*2, 2)

    def forward(self, inputs, after_embedding=False, train=False):
        hidden = self.init_hidden()
        if not after_embedding:
            embeddings = self.embeddings(inputs).view(len(inputs), 1, -1)
            if train:
                var_embeddings = embeddings
            else:
                var_embeddings = Variable(embeddings, requires_grad=True)
        else:
            var_embeddings = inputs
        lstm_out, (hn, cn) = self.lstm(var_embeddings, hidden)
        x = hn.view(1, -1)
        x = self.linearOut(x)
        return x, var_embeddings

    def init_hidden(self):
        return (Variable(torch.zeros(2, 1, self.hidden_dim)).cuda(),
                Variable(torch.zeros(2, 1, self.hidden_dim)).cuda())
        # return (Variable(torch.zeros(1, 1, self.hidden_dim)).cuda(,
        #         Variable(torch.zeros(1, 1, self.hidden_dim)).cuda()