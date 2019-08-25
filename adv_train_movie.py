import pickle
import re
import os
import unicodedata
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import pickle
from utils.wordProcess import *
from models.moiveRnn import Model
from settings import *
from adversary.fgsm import Attack_MOVIE
from utils.roc_plot import roc_auc

with open('output/dict.pkl','rb') as f :
    word_dict = pickle.load(f)
word_length = len(word_dict)
# print(word_length)
vocabLimit = 50000
max_sequence_len = 500
embedding_dim = 50
hidden_dim = 100

model = Model(embedding_dim, hidden_dim,vocabLimit).to(device)
criterion_none = nn.CrossEntropyLoss(reduction='none')
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

checkpoint = torch.load(MOIVE_CKPT)
model.load_state_dict(checkpoint['net'])
best_acc = 0

f = open('data/labeledTrainData.tsv').readlines()

bim_attack = Attack_MOVIE(model, F.cross_entropy)
def FGSM(x, y_true, eps=0.01):
    x = Variable(x.to(device), requires_grad=True)
    y_true = Variable(y_true.to(device), requires_grad=False)

    x_adv = bim_attack.fgsm(x, y_true, False, eps)
    return x_adv

def jsma(x_in, y_in, model, nb_classes, max_iter=10, fix_iter=False):
    logits, embeddings = model(x_in)
    _, predicted = logits.max(1)
    change_words = 0
    change_list = []
    changed = False
    while not changed and change_words < max_iter:
        change_words += 1
        # create the Jacobian
        grads = None
        for class_ind in range(nb_classes):
            model.zero_grad()
            logits[:, class_ind].sum().backward(retain_graph=True)
            derivatives = embeddings.grad.reshape(len(x_in), -1)
            derivatives = derivatives.sum(dim=1)
            if class_ind == 0:
                grads = derivatives
            else:
                grads = torch.cat((grads, derivatives))
        grads = grads.reshape(nb_classes, -1).cpu().numpy()
        gradsum = np.abs(grads[1-y_in,:]) * (-grads[y_in,:])
        max_index = np.argmax(gradsum)
        while max_index in change_list:
            gradsum[max_index] = -1
            max_index = np.argmax(gradsum)
        change_list.append(max_index)
        min_confidence = torch.nn.functional.softmax(logits, dim=1)[0, y_in]
        best_word = x_in[max_index]
        for i in range(50):
            x_in[max_index] = i
            logits, _ = model(x_in)
            confidence = torch.nn.functional.softmax(logits, dim=1)[0,y_in]
            if confidence < min_confidence:
                min_confidence = confidence
                best_word = i
            if confidence < 0.5:
                break
        x_in[max_index] = best_word
        logits, _ = model(x_in)
        _, predicted = logits.max(1)
        changed = bool(predicted != y_in)
        if fix_iter:
            changed = False

    return changed, x_in, change_words, criterion_none(logits, torch.LongTensor([y_in]).to(device)).detach().cpu().numpy()[0]


for epoch in range(NUM_EPOCHS):
    right = 0
    total = 0
    benignloss_list = []
    advloss_list = []
    for idx, lines in enumerate(f):
        if idx > 0:
            # if idx <= 20000: #20000
            #     continue
            data = lines.split('\t')[2]
            data = normalizeString(data).strip()
            input_data = [word_dict[word] for word in data.split(' ')]
            if len(input_data) > max_sequence_len:
                input_data = input_data[0:max_sequence_len]
            input_data = Variable(torch.LongTensor(input_data)).to(device)
            target = int(lines.split('\t')[1])
            target_data = Variable(torch.LongTensor([target])).to(device)

            if idx <= 20000: #20000
                _, input_data_embedding = model(input_data)
                benign_undercover = FGSM(input_data_embedding, target_data, eps=0.01)

                y_pred, _ = model(input_data)
                adv_pred, _ = model(benign_undercover, after_embedding=True)

                model.zero_grad()
                loss1 = loss_function(y_pred, target_data)
                loss2 = loss_function(adv_pred, target_data)
                loss = loss1 + 0.8 * loss2

                loss.backward()
                optimizer.step()
            elif idx > 20000:
                y_pred, embeddings = model(input_data)
                _, predicted = y_pred.max(1)

                if predicted.eq(target_data).sum().item():
                    right += 1
                    changed, benign_adv, change_words, loss_benign = jsma(input_data, target, model,
                                                                          nb_classes=2, max_iter=20)
                    if changed:
                        _, input_data_embedding = model(input_data)
                        _, benign_adv_embedding = model(benign_adv)
                        benign_undercover = FGSM(input_data_embedding, target_data, eps=0.001)
                        adv_undercover = FGSM(benign_adv_embedding, 1 - target_data, eps=0.001)

                        benign_outputs, _ = model(benign_undercover, after_embedding=True)
                        temp1 = criterion_none(benign_outputs, target_data).detach().cpu().numpy()[0]
                        adv_outputs, _ = model(adv_undercover, after_embedding=True)
                        temp2 = criterion_none(adv_outputs, 1 - target_data).detach().cpu().numpy()[0]

                        benignloss_list.append(temp1)
                        advloss_list.append(temp2)
                total += 1
            if idx % 2000==1:
                print('epoch: %d, idx: %d' % (epoch, idx))
            # if idx == 100:
            #     break

    print('-'*30)
    print('eopch: ', epoch)
    print('acc: ', right / total)
    print('benignloss_mean: ', np.mean(benignloss_list))
    print('advloss_mean: ',np.mean(advloss_list))

    benignloss_list = np.array(benignloss_list)
    advloss_list = np.array(advloss_list)
    losses = np.concatenate((benignloss_list, advloss_list), axis=0)
    labels = np.concatenate((np.ones_like(benignloss_list), np.zeros_like(advloss_list)), axis=0)
    auc_score = roc_auc(labels, losses)
    print('[ROC_AUC] score: %.2f%%' % (100. * auc_score))

    if auc_score > best_acc:
        best_acc = auc_score
        print('save checkpoint!')
        state = {
            'net': model.state_dict(),
            'acc': auc_score,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, MOIVE_CKPT_ADV_TRAINING)

