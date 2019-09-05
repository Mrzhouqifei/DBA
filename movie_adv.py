import os
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
from adversary.jsma import jsma

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

checkpoint = torch.load(MOIVE_CKPT) #MOIVE_CKPT_ADV_TRAINING
model.load_state_dict(checkpoint['net'])
best_acc = 0

f = open('data/labeledTrainData.tsv').readlines()

bim_attack = Attack_MOVIE(model, F.cross_entropy)
def FGSM(x, y_true, eps=0.001):
    x = Variable(x.to(device), requires_grad=True)
    y_true = Variable(y_true.to(device), requires_grad=False)

    x_adv = bim_attack.fgsm(x, y_true, False, eps)
    return x_adv

def train():
    global best_acc
    for epoch in range(NUM_EPOCHS):
        right = 0
        total = 0
        total_changed = 0
        benign_right = 0
        adv_right = 0
        benignloss_list = []
        advloss_list = []
        flag = (epoch % 4 == 3)
        print('-'*20, epoch, flag)
        for idx, lines in enumerate(f):
            if idx > 0:
                data = lines.split('\t')[2]
                data = normalizeString(data).strip()
                input_data = [word_dict[word] for word in data.split(' ')]
                if len(input_data) > max_sequence_len:
                    input_data = input_data[0:max_sequence_len]
                input_data = Variable(torch.LongTensor(input_data)).to(device)
                target = int(lines.split('\t')[1])
                target_data = Variable(torch.LongTensor([target])).to(device)

                if idx <= 20000:
                    _, input_data_embedding = model(input_data)
                    benign_undercover = FGSM(input_data_embedding, target_data)

                    y_pred, _ = model(input_data)
                    adv_pred, _ = model(benign_undercover, after_embedding=True)

                    model.zero_grad()
                    loss1 = loss_function(y_pred, target_data)
                    loss2 = loss_function(adv_pred, target_data)
                    loss = loss1 + 0.8 * loss2

                    loss.backward()
                    optimizer.step()
                elif (idx > 20000) and flag:
                    y_pred, embeddings = model(input_data)
                    _, predicted = y_pred.max(1)

                    if predicted.eq(target_data).sum().item():
                        right += 1
                        changed, benign_adv, change_words, loss_benign = jsma(input_data.clone(), target, model,
                                                                              nb_classes=2, max_iter=20)
                        if changed:
                            total_changed += 1
                            _, input_data_embedding = model(input_data)
                            _, benign_adv_embedding = model(benign_adv)
                            benign_undercover = FGSM(input_data_embedding, target_data)
                            adv_undercover = FGSM(benign_adv_embedding, 1 - target_data)

                            benign_outputs, _ = model(benign_undercover, after_embedding=True)
                            temp1 = criterion_none(benign_outputs, target_data).detach().cpu().numpy()[0]
                            adv_outputs, _ = model(adv_undercover, after_embedding=True)
                            temp2 = criterion_none(adv_outputs, 1 - target_data).detach().cpu().numpy()[0]

                            _, undercover_benign_predicted = benign_outputs.max(1)
                            _, undercover_adv_predicted = adv_outputs.max(1)
                            benign_right += (undercover_benign_predicted == target_data).cpu().numpy()[0]
                            adv_right += (undercover_adv_predicted == (1 - target_data)).cpu().numpy()[0]

                            benignloss_list.append(temp1)
                            advloss_list.append(temp2)
                    total += 1
                if (idx+1) % 2000 == 0:
                    print('epoch: %d, idx: %d' % (epoch, idx))

        if flag:
            print('-'*30)
            print('eopch: ', epoch)
            print('acc: ', right / total)
            print('benignloss_mean: ', np.mean(benignloss_list))
            print('advloss_mean: ',np.mean(advloss_list))
            print('fgsm attack benign: %.2f%% adversary: %.2f%%' % (100. * benign_right / total_changed,
                                                                    100. * adv_right / total_changed))

            benignloss_list = np.array(benignloss_list)
            advloss_list = np.array(advloss_list)
            losses = np.concatenate((benignloss_list, advloss_list), axis=0)
            labels = np.concatenate((np.zeros_like(benignloss_list), np.ones_like(advloss_list)), axis=0)
            auc_score = roc_auc(labels, losses)
            print('split criterion', np.median(losses))
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

def test():
    right = 0
    total = 0
    total_changed = 0
    total_changed_num = 0
    benign_right = 0
    adv_right = 0
    benignloss_list = []
    advloss_list = []
    for idx, lines in enumerate(f):
        if idx > 20000:
            data = lines.split('\t')[2]
            data = normalizeString(data).strip()
            input_data = [word_dict[word] for word in data.split(' ')]
            if len(input_data) > max_sequence_len:
                input_data = input_data[0:max_sequence_len]
            input_data = Variable(torch.LongTensor(input_data)).to(device)
            target = int(lines.split('\t')[1])
            target_data = Variable(torch.LongTensor([target])).to(device)

            y_pred, embeddings = model(input_data)
            _, predicted = y_pred.max(1)

            if predicted.eq(target_data).sum().item():
                right += 1
                changed, benign_adv, change_words, loss_benign = jsma(input_data.clone(), target, model,
                                                                      nb_classes=2, max_iter=20)
                if changed:
                    total_changed += 1
                    total_changed_num += change_words
                    _, input_data_embedding = model(input_data)
                    _, benign_adv_embedding = model(benign_adv)
                    benign_undercover = FGSM(input_data_embedding, target_data)
                    adv_undercover = FGSM(benign_adv_embedding, 1 - target_data)

                    benign_outputs, _ = model(benign_undercover, after_embedding=True)
                    temp1 = criterion_none(benign_outputs, target_data).detach().cpu().numpy()[0]
                    adv_outputs, _ = model(adv_undercover, after_embedding=True)
                    temp2 = criterion_none(adv_outputs, 1 - target_data).detach().cpu().numpy()[0]

                    _, undercover_benign_predicted = benign_outputs.max(1)
                    _, undercover_adv_predicted = adv_outputs.max(1)
                    benign_right += (undercover_benign_predicted == target_data).cpu().numpy()[0]
                    adv_right += (undercover_adv_predicted == (1-target_data)).cpu().numpy()[0]

                    benignloss_list.append(temp1)
                    advloss_list.append(temp2)
            total += 1
            if (idx+1) % 2000 == 0:
                print('idx: %d' % (idx))
            # if idx == 20100:
            #     break

    print('-' * 30)
    print('acc: ', right / total)
    print('mean changed words', total_changed_num / total_changed)
    print('benignloss_mean: ', np.mean(benignloss_list))
    print('advloss_mean: ', np.mean(advloss_list))
    print('fgsm attack benign: %.2f%% adversary: %.2f%%' % (100. * benign_right/total_changed,
                                                            100. * adv_right/total_changed))

    benignloss_list = np.array(benignloss_list)
    advloss_list = np.array(advloss_list)
    losses = np.concatenate((benignloss_list, advloss_list), axis=0)
    labels = np.concatenate((np.zeros_like(benignloss_list), np.ones_like(advloss_list)), axis=0)
    auc_score = roc_auc(labels, losses)
    print('split criterion', np.median(losses))
    print('[ROC_AUC] score: %.2f%%' % (100. * auc_score))

test()
# train()