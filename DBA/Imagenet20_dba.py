# coding: utf-8

import os
import pynvml
pynvml.nvmlInit()
from settings import *
import time
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from adversary.fgsm import Attack
from torch.nn.modules.distance import PairwiseDistance
from util.roc_plot import roc_auc
import adversary.cw as cw
from adversary.jsma import SaliencyMapMethod
import torchvision.models as models


def FGSM(x, y_true, eps=8 / 255, alpha=1 / 255, iteration=10, bim_a=False, train=False):
    x = Variable(x.to(device), requires_grad=False)
    y_true = Variable(y_true.to(device), requires_grad=False)

    if train:
        x_adv = bim_attack.mini_imagenet__train_fgsm(x, y_true, False, eps)
    else:
        if iteration == 1:
            x_adv = bim_attack.fgsm(x, y_true, False, eps)
        else:
            if bim_a:
                x_adv = bim_attack.i_fgsm_a(x, y_true, False, eps, alpha, iteration)
            else:
                x_adv = bim_attack.i_fgsm(x, y_true, False, eps, alpha, iteration)
    return x_adv

def converse(x, mean ,std):
    x[:,0]=x[:,0]*std[0]+mean[0]
    x[:,1]=x[:,1]*std[1]+mean[1]
    x[:,2]=x[:,2].mul(std[2])+mean[2]
    return x

def train(epoch):
    global total_epoch
    total_epoch += 1
    net.train()
    print('-' * 30)
    correct = 0
    adv_correct = 0
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

        x_adv = FGSM(inputs, targets, eps=EPSILON_TRAIN, train=False)
        adv_outputs = net(x_adv)
        _, adv_predicted = adv_outputs.max(1)
        adv_correct += adv_predicted.eq(targets).sum().item()

        loss1 = criterion(outputs, targets)
        loss2 = criterion(adv_outputs, targets)
        loss = loss1 + loss2 * 0.8
        loss.backward()
        optimizer.step()
        end = time.time()
        if (batch_idx + 1) % 100 == 0: #
            print('batch:%d, time:%d' % (batch_idx, end - start))
    acc = correct / total
    adv_acc = adv_correct / total
    print('train acc: %.2f%%' % (100. * acc))
    print('adv acc: %.2f%%' % (100. * adv_acc))

    # print('update resnet ckpt!')
    # state = {
    #     'net': net.state_dict(),
    #     'acc': 0,
    #     'epoch': total_epoch,
    # }
    # if not os.path.isdir('checkpoint'):
    #     os.mkdir('checkpoint')
    # torch.save(state, ADV_MINI_IMAGENET_CKPT)


def test(methods='fgsm', update=False):
    global best_acc, total_epoch
    print('\nEpoch: %d' % total_epoch)
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

    for batch_idx, (inputs, targets) in enumerate(testloader):
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
        outputs = torch.from_numpy(outputs.detach().cpu().numpy()[selected]).to(device)
        total_right += inputs.size(0)

        # benign fgsm
        benign_fgsm = FGSM(inputs, predicted, eps=EPSILON_TRAIN)
        benign_fgsm__outputs = net(benign_fgsm)
        _, benign_fgsm_predicted = benign_fgsm__outputs.max(1)
        temp1 = criterion_none(benign_fgsm__outputs, predicted).detach().cpu().numpy()
        # temp1 = criterion_none(benign_fgsm__outputs, predicted).detach().cpu().numpy() - \
        #         criterion_none(outputs, predicted).detach().cpu().numpy()

        # attack begin
        if methods == 'fgsm':
            x_adv = FGSM(inputs, predicted, eps=EPSILON*2, alpha=1 / 255, iteration=1)
        elif methods == 'bim_a':
            x_adv = FGSM(inputs, predicted, eps=EPSILON, alpha=1 / 255, iteration=10, bim_a=True)
        elif methods == 'bim_b':
            x_adv = FGSM(inputs, predicted, eps=EPSILON/2, alpha=1 / 255, iteration=5)
        elif methods == 'jsma':
            x_adv = jsma_attack.generate(inputs, y=predicted)
        else:
            x_adv = cw_attack(net, inputs, predicted, to_numpy=False)
        l2sum += l2dist.forward(x_adv.reshape(temp_batch, -1),
                                inputs.reshape(temp_batch, -1)).sum().detach().cpu().numpy()
        # l2sum += l2dist.forward(converse(x_adv.clone(), tf_img.mean,tf_img.std).reshape(temp_batch, -1),
        #                         converse(inputs.clone(), tf_img.mean,tf_img.std).reshape(temp_batch, -1)).sum().detach().cpu().numpy()
        adv_outputs = net(x_adv)
        _, adv_predicted = adv_outputs.max(1)
        attack_correct += adv_predicted.eq(predicted).sum().item()
        selected = (adv_predicted != targets).cpu().numpy().astype(bool)
        # selected = np.ones_like(selected).astype(bool)

        # adv_fgsm
        adv_fgsm = FGSM(x_adv, adv_predicted, eps=EPSILON_TRAIN)  #
        adv_fgsm_outputs = net(adv_fgsm)
        _, adv_fgsm_predicted = adv_fgsm_outputs.max(1)
        temp2 = criterion_none(adv_fgsm_outputs, adv_predicted).detach().cpu().numpy()
        # temp2 = criterion_none(adv_fgsm_outputs, adv_predicted).detach().cpu().numpy() - \
        #         criterion_none(adv_outputs, adv_predicted).detach().cpu().numpy()

        # select the examples which is attacked successfully
        temp1 = temp1[selected].reshape(1, -1)
        temp2 = temp2[selected].reshape(1, -1)
        if batch_idx != 0:
            benign_fgsm_loss = np.column_stack((benign_fgsm_loss, temp1))
            adv_fgsm_loss = np.column_stack((adv_fgsm_loss, temp2))
        else:
            benign_fgsm_loss = temp1
            adv_fgsm_loss = temp2
        # print('batch:',batch_idx)

        total_attack_sucess += len(temp1[0])
        benign_fgsm_correct += np.equal(benign_fgsm_predicted.cpu().numpy()[selected],
                                        (predicted.cpu().numpy()[selected])).sum()
        adv_fgsm_correct += np.equal(adv_fgsm_predicted.cpu().numpy()[selected],
                                     (adv_predicted.cpu().numpy()[selected])).sum()
        # print(batch_idx)

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
    print(np.mean(benign_fgsm_loss), np.mean(adv_fgsm_loss))
    print('split criterion', np.median(losses))
    print('[ROC_AUC] score: %.2f%%' % (100. * auc_score))

    # Save checkpoint.
    if (auc_score > best_acc) and update:
        print('update resnet ckpt!')
        state = {
            'net': net.state_dict(),
            'acc': auc_score,
            'epoch': total_epoch,
        }
        if not os.path.isdir('../checkpoint'):
            os.mkdir('../checkpoint')
        torch.save(state, ADV_MINI_IMAGENET_CKPT)
        best_acc = auc_score


if __name__ == '__main__':
    rootpath = '/home/qifeiz/ImageNetData/mini-imagenet/Imagenet-20/'

    resnet18 = models.resnet18(pretrained=True)
    fc_features = resnet18.fc.in_features
    resnet18.fc = nn.Linear(fc_features, 20)

    tf_img = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    net = resnet18
    # state = {
    #     'net': net.state_dict(),
    #     'acc': 0,
    #     'epoch': 0,
    # }
    # if not os.path.isdir('checkpoint'):
    #     os.mkdir('checkpoint')
    # torch.save(state, ADV_MINI_IMAGENET_CKPT)

    checkpoint = torch.load(ADV_MINI_IMAGENET_CKPT)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc'] * 100
    total_epoch = checkpoint['epoch']
    print('best_auc: %.2f%%' % best_acc)
    if best_acc > 99:
        best_acc = 0
        print('set 0')
    net.to(device)
    print('load success')

    # train dataloader
    train_dataset = torchvision.datasets.ImageFolder(root=rootpath + 'train', transform=tf_img)
    trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE_IMAGENET20, shuffle=True, num_workers=4)

    # test dataloader
    dataset = torchvision.datasets.ImageFolder(root=rootpath + 'test', transform=tf_img)
    testloader = DataLoader(dataset, batch_size=BATCH_SIZE_IMAGENET20 // 2, shuffle=False, num_workers=4)

    EPSILON = 8 / 255
    EPSILON_TRAIN = 1 / 255
    l2dist = PairwiseDistance(2)
    criterion_none = nn.CrossEntropyLoss(reduction='none')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    # attacks
    bim_attack = Attack(net, F.cross_entropy)
    cw_attack = cw.L2Adversary(targeted=False,
                               confidence=0.5,
                               search_steps=10,
                               box=(0, 1),
                               optimizer_lr=0.001)
    jsma_params = {'theta': 1, 'gamma': 0.1,
                   'clip_min': 0., 'clip_max': 1.,
                   'nb_classes': 20}
    jsma_attack = SaliencyMapMethod(net, **jsma_params)


    # for i in range(50):
    #     train(i)
    #     if (i+1) % 1 == 0:
    #         test('fgsm', update=True)
    test('cw', update=False)
