# coding: utf-8

# In[ ]:


import os
import gc
import re
import shutil
import pynvml
import torchvision

pynvml.nvmlInit()
from settings import *

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


import torchvision.models as models

rootpath = '/home/qifeiz/ImageNetData/mini-imagenet/Imagenet-20/'

resnet18 = models.resnet18(pretrained=True)
# resnet18.classifier._modules['6'] = nn.Linear(4096, 20)
fc_features = resnet18.fc.in_features
resnet18.fc = nn.Linear(fc_features, 20)

tf_img = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
net = resnet18

state = {
    'net': net.state_dict(),
    'acc': 0,
    'epoch': 0,
}
if not os.path.isdir('checkpoint'):
    os.mkdir('checkpoint')
torch.save(state, MINI_IMAGENET_CKPT)

checkpoint = torch.load(MINI_IMAGENET_CKPT)
net.load_state_dict(checkpoint['net'])
best_acc = checkpoint['acc']
start_epoch = checkpoint['epoch']
print('best_acc: %.2f%%' %  best_acc)
net.to(device)
print('load success')

# train dataloader
train_dataset = torchvision.datasets.ImageFolder(root=rootpath+'train', transform=tf_img)
trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE_IMAGENET20, shuffle=True, num_workers=4)

# test dataloader
dataset = torchvision.datasets.ImageFolder(root=rootpath+'test', transform=tf_img)
testloader = DataLoader(dataset, batch_size=BATCH_SIZE_IMAGENET20, shuffle=False, num_workers=4)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    acc = 100. * correct / total
    print('train acc: %.2f' % acc)

def test(epoch, update=False):
    global best_acc
    net.eval()
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    # Save checkpoint.
    acc = 100.*correct/total
    print('valid acc: %.2f%%' % acc)
    if acc >= best_acc and update:
        print('update resnet ckpt!')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, MINI_IMAGENET_CKPT)
        best_acc = acc

for epoch in range(start_epoch, start_epoch+NUM_EPOCHS):
    train(epoch)
    torch.cuda.empty_cache()
    test(epoch, update=True)

# test(0)