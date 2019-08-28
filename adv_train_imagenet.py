# coding: utf-8

# In[ ]:


import os
import gc
import re
import shutil
import pynvml

pynvml.nvmlInit()

import time
from torch.utils.data import DataLoader
from torch.autograd import Variable
import pandas as pd
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms
import pretrainedmodels
import torch
import torch.nn as nn
import torch.nn.functional as F
import pretrainedmodels.utils as utils
import numpy as np
from adversary.fgsm import Attack, where
from torch.nn.modules.distance import PairwiseDistance
from utils.roc_plot import roc_auc
import adversary.cw as cw
from adversary.jsma import SaliencyMapMethod

device = 'cuda' if torch.cuda.is_available() else 'cpu'
rootpath = '/home/qifeiz/ImageNetData/competition0723/'

# In[ ]:


# label 预处理之后与比赛方提供的一致
labels_init = pd.read_csv(rootpath + 'dataset0.csv')[['ImageId', 'TrueLabel', 'TargetClass']]
labels_init['TrueLabel'] = labels_init['TrueLabel'] - 1
labels_init['TargetClass'] = labels_init['TargetClass'] - 1

# Load Imagenet Synsets
with open(rootpath + 'imagenet_synsets.txt', 'r') as f:
    synsets = f.readlines()
synsets = [x.strip() for x in synsets]
splits = [line.split(' ') for line in synsets]
key_to_classname = {spl[0]: ' '.join(spl[1:]) for spl in splits}
with open(rootpath + 'imagenet_classes.txt', 'r') as f:
    class_id_to_key = f.readlines()
class_id_to_key = [x.strip() for x in class_id_to_key]

# # 预处理图片
class_names = list(pd.read_table(rootpath + 'imagenet_classes.txt', header=None)[0])
if not os.path.exists(rootpath + 'dataset0_pre/'):
    os.mkdir(rootpath + 'dataset0_pre/')
for class_name in class_names:
    if not os.path.exists(rootpath + 'dataset0_pre/' + class_name):
        os.mkdir(rootpath + 'dataset0_pre/' + class_name)
for i in range(len(labels_init)):
    true_class_name = class_id_to_key[labels_init.TrueLabel[i]]
    shutil.copy(rootpath + 'dataset0/' + labels_init.ImageId[i] + '.png', rootpath + 'dataset0_pre/' + true_class_name)

# In[ ]:


model = pretrainedmodels.__dict__['inceptionv4'](num_classes=1000, pretrained='imagenet')  # .to(device)
tf_img = utils.TransformImage(model, scale=1, random_crop=False, random_hflip=False, random_vflip=False,
                              preserve_aspect_ratio=True)
net = model
# state = {
#     'net': model.state_dict(),
#     'acc': 0,
#     'epoch': 0,
# }
# if not os.path.isdir('checkpoint'):
#     os.mkdir('checkpoint')
# torch.save(state, 'checkpoint/ImageNet.pth')

# checkpoint = torch.load('checkpoint/ImageNet.pth')
# net.load_state_dict(checkpoint['net'])
# best_acc = checkpoint['acc']
# print('best_acc: %.2f%%' % (100. * best_acc))
net.to(device)
print('load success')

# In[ ]:


# train dataloader
train_dataset = torchvision.datasets.ImageFolder(root='/home/qifeiz/ImageNetData/ILSVRC/Data/CLS-LOC/train',
                                                 transform=tf_img)
trainloader = DataLoader(train_dataset, batch_size=28, shuffle=True, num_workers=4)

# In[ ]:


# validate dataloader
dataset = torchvision.datasets.ImageFolder(root=rootpath + 'dataset0_pre', transform=tf_img)
dataset_loader = DataLoader(dataset, batch_size=10, shuffle=False, num_workers=4)

ImageIds = []
for x in dataset.imgs:
    ImageIds.append(x[0][-20:-4])
ImageId = pd.DataFrame(ImageIds)
ImageId.columns = ['ImageId']
labels = ImageId.merge(labels_init, on='ImageId')

# In[ ]:


EPSILON = 16 / 255
l2dist = PairwiseDistance(2)
criterion_none = nn.CrossEntropyLoss(reduction='none')
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
# attacks
bim_attack = Attack(net, F.cross_entropy)
cw_attack = cw.L2Adversary(targeted=False,
                           confidence=0.9,
                           search_steps=10,
                           box=(-1, 1),
                           optimizer_lr=0.001)

# In[ ]:


def FGSM(x, y_true, eps=8 / 255, alpha=1 / 255, iteration=10, bim_a=False):
    x = Variable(x.to(device), requires_grad=False)
    y_true = Variable(y_true.to(device), requires_grad=False)

    if iteration == 1:
        x_adv = bim_attack.fgsm(x, y_true, False, eps, x_val_min=-1, x_val_max=1)
    else:
        if bim_a:
            x_adv = bim_attack.i_fgsm_a(x, y_true, False, eps, alpha, iteration, x_val_min=-1, x_val_max=1)
        else:
            x_adv = bim_attack.i_fgsm(x, y_true, False, eps, alpha, iteration, x_val_min=-1, x_val_max=1)
    return x_adv


# In[ ]:


def train(epoch, method='fgsm'):
    net.train()
    print('-' * 30)
    print('\nEpoch: %d' % epoch)
    correct = 0
    total = 0
    start = time.time()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        net.train()
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        x_adv = FGSM(inputs, targets, eps=EPSILON)
        adv_outputs = net(x_adv)

        loss1 = criterion(outputs, targets)
        loss2 = criterion(adv_outputs, targets)
        loss = loss1 + loss2 * 0.8
        loss.backward()
        optimizer.step()
        end = time.time()
        if (batch_idx + 1) % 1000 == 0:
            del inputs, targets, outputs, predicted, x_adv, adv_outputs, loss1, loss2, loss
            net.cpu()
            gc.collect()
            torch.cuda.empty_cache()
            print('batch:%d, time:%d' % (batch_idx, end - start))
            test(methods=method)
    acc = correct / total
    print('train acc: %.2f%%' % (100. * acc))


# In[ ]:


def test(methods='fgsm'):
    global best_acc
    net.cuda()
    net.eval()
    correct = 0
    total = 0
    total_right = 0
    total_attack_sucess = 0
    benign_fgsm_correct = 0
    adv_fgsm_correct = 0
    attack_correct = 0
    benign_fgsm_loss = None
    adv_fgsm_loss = None
    l2sum = 0

    for batch_idx, (inputs, targets) in enumerate(dataset_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        _, predicted = outputs.max(1)
        if predicted.eq(targets).sum().item() == 0:
            continue
        correct += predicted.eq(targets).sum().item()
        selected = (predicted == targets).cpu().numpy().astype(bool)
        total += inputs.size(0)
        temp_batch = selected.sum()

        inputs = torch.from_numpy(inputs.cpu().numpy()[selected]).to(device)
        targets = torch.from_numpy(targets.cpu().numpy()[selected]).to(device)
        predicted = torch.from_numpy(predicted.cpu().numpy()[selected]).to(device)
        # outputs = torch.from_numpy(outputs.detach().cpu().numpy()[selected]).to(device)
        total_right += inputs.size(0)

        # benign fgsm
        benign_fgsm = FGSM(inputs, predicted, eps=EPSILON)
        benign_fgsm__outputs = net(benign_fgsm)
        _, benign_fgsm_predicted = benign_fgsm__outputs.max(1)
        # temp1 = l2dist.forward(F.softmax(benign_fgsm__outputs, dim=1), F.softmax(outputs, dim=1)).detach().cpu().numpy()
        temp1 = criterion_none(benign_fgsm__outputs, predicted).detach().cpu().numpy()

        # attack begin
        if methods == 'fgsm':
            x_adv = FGSM(inputs, predicted, eps=EPSILON*2, alpha=2 / 255, iteration=1)
        elif methods == 'bim_a':
            x_adv = FGSM(inputs, predicted, eps=EPSILON, alpha=2 / 255, iteration=10, bim_a=True)
        elif methods == 'bim_b':
            x_adv = FGSM(inputs, predicted, eps=EPSILON, alpha=2 / 255, iteration=20)
        # elif methods == 'jsma':
        #     x_adv = jsma_attack.generate(inputs, y=predicted)
        else:
            x_adv = cw_attack(net, inputs, predicted, to_numpy=False)

        l2sum += l2dist.forward(x_adv.reshape(temp_batch, -1),
                                inputs.reshape(temp_batch, -1)).sum().detach().cpu().numpy()
        adv_outputs = net(x_adv)
        _, adv_predicted = adv_outputs.max(1)
        attack_correct += adv_predicted.eq(predicted).sum().item()
        selected = (adv_predicted != targets).cpu().numpy().astype(bool)

        # adv_fgsm
        adv_fgsm = FGSM(x_adv, adv_predicted, eps=EPSILON)  #
        adv_fgsm_outputs = net(adv_fgsm)
        _, adv_fgsm_predicted = adv_fgsm_outputs.max(1)
        # temp2 = l2dist.forward(F.softmax(adv_fgsm_outputs, dim=1), F.softmax(adv_outputs, dim=1)).detach().cpu().numpy()
        temp2 = criterion_none(adv_fgsm_outputs, adv_predicted).detach().cpu().numpy()

        # select the examples which is attacked successfully
        temp1 = temp1[selected].reshape(1, -1)
        temp2 = temp2[selected].reshape(1, -1)
        if batch_idx != 0:
            benign_fgsm_loss = np.column_stack((benign_fgsm_loss, temp1))
            adv_fgsm_loss = np.column_stack((adv_fgsm_loss, temp2))
        else:
            benign_fgsm_loss = temp1
            adv_fgsm_loss = temp2
        print('batch:',batch_idx)

        total_attack_sucess += len(temp1[0])
        benign_fgsm_correct += np.equal(benign_fgsm_predicted.cpu().numpy()[selected],
                                        (predicted.cpu().numpy()[selected])).sum()
        adv_fgsm_correct += np.equal(adv_fgsm_predicted.cpu().numpy()[selected],
                                     (adv_predicted.cpu().numpy()[selected])).sum()

    acc = correct/total
    attack_acc = attack_correct / total_right
    benign_fgsm_acc = benign_fgsm_correct/ total_attack_sucess
    adv_fgsm_acc = adv_fgsm_correct / total_attack_sucess
    print(total, total_right)
    print('valid acc: %.2f%%' % (100. * acc))
    print('attact acc: %.2f%% L2 perturbation: %.2f' % (100. * attack_acc, l2sum / total_right))
    print('fgsm attack benign: %.2f%% adversary: %.2f%%' % (100. * benign_fgsm_acc, 100. * adv_fgsm_acc))
    benign_fgsm_loss = benign_fgsm_loss.reshape(-1)
    adv_fgsm_loss = adv_fgsm_loss.reshape(-1)

    losses = np.concatenate((benign_fgsm_loss, adv_fgsm_loss), axis=0)
    labels = np.concatenate((np.zeros_like(benign_fgsm_loss), np.ones_like(adv_fgsm_loss)), axis=0)
    auc_score = roc_auc(labels, losses)
    print('[ROC_AUC] score: %.2f%%' % (100. * auc_score))

    # Save checkpoint.
    # if auc_score > best_acc:
    #     print('update resnet ckpt!')
    #     state = {
    #         'net': net.state_dict(),
    #         'acc': auc_score,
    #     }
    #     if not os.path.isdir('checkpoint'):
    #         os.mkdir('checkpoint')
    #     torch.save(state, 'checkpoint/ImageNet.pth')
    #     best_acc = auc_score


# In[ ]:


# train(1)

# test('fgsm')
# test('bim_a')
# test('bim_b')
test('cw')