import re
import os
import unicodedata
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import pickle
from utils.wordProcess import *
from models.moiveRnn import Model#, Embedding
from settings import *

vocabLimit = 50000
max_sequence_len = 500
obj1 = wordIndex()
embedding_dim = 50
hidden_dim = 100
#labeledTrainData   movieTrain
f = open('data/labeledTrainData.tsv').readlines()
print('reading the lines')
for idx, lines in enumerate(f):
    if not idx == 0:
        data = lines.split('\t')[2]
        data = normalizeString(data).strip()
        obj1.add_text(data)
print('read all the lines')

limitDict(vocabLimit, obj1)

model = Model(embedding_dim, hidden_dim, vocabLimit).to(device)
best_acc = 0

if LOAD_CKPT:
    checkpoint = torch.load(MOIVE_CKPT)
    model.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    print('best_acc: %.2f' % best_acc)

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

print('starting training')
""""
1~20000 train
20001~25000 test
"""

for i in range(20):
    sum_loss = 0.0
    right = 0
    for idx, lines in enumerate(f):
        if idx > 0:
            data = lines.split('\t')[2]
            data = normalizeString(data).strip()
            input_data = [obj1.word_to_idx[word] for word in data.split(' ')]
            if len(input_data) > max_sequence_len:
                input_data = input_data[0:max_sequence_len]
            input_data = Variable(torch.LongTensor(input_data)).to(device)

            target = int(lines.split('\t')[1])
            target_data = Variable(torch.LongTensor([target])).to(device)

            if idx <= 20000:
                model.train()
                y_pred, embeddings = model(input_data)
                model.zero_grad()
                loss = loss_function(y_pred, target_data)
                sum_loss += loss.data.item()

                loss.backward()
                # embeddings.grad.data.fill_(0)
                optimizer.step()
            elif idx > 20000:
                model.eval()
                y_pred, embeddings = model(input_data)
                _, predicted = y_pred.max(1)
                right += predicted.eq(target).sum().item()
    acc = right / 5000
    if acc > best_acc:
        best_acc = acc
        print('save checkpoint!')
        state = {
            'net': model.state_dict(),
            'acc': acc,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, MOIVE_CKPT)

    print('train_loss %d epochs is %g' % ((i + 1), (sum_loss / 20000)))
    print('test acc:', acc)
    print('-'*30)

with open('output/dict.pkl', 'wb') as f:
    pickle.dump(obj1.word_to_idx, f)