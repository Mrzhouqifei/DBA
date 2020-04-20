"""fgsm.py"""
import torch
from torch.autograd import Variable
import numpy as np
from settings import *
import torch.nn.functional as F

class Attack(object):
    def __init__(self, classify_net, criterion):
        self.net = classify_net
        self.criterion = criterion

    def fgsm(self, x, y, targeted=False, eps=8/255, x_val_min=0, x_val_max=1):
        x_adv = Variable(x.data, requires_grad=True)
        h_adv = self.net(x_adv)
        if targeted:
            cost = -self.criterion(h_adv, y)
        else:
            cost = self.criterion(h_adv, y)

        self.net.zero_grad()
        if x_adv.grad is not None:
            x_adv.grad.data.fill_(0)
        cost.backward()

        x_adv = x_adv + eps*x_adv.grad.sign_()
        x_adv = torch.clamp(x_adv, x_val_min, x_val_max)
        return x_adv

    def mini_imagenet__train_fgsm(self, x, y, targeted=False, eps=8/255, x_val_min=0, x_val_max=1):
        x_adv = Variable(x.data, requires_grad=True)
        h_adv = self.net(x_adv)[0]
        if targeted:
            cost = -self.criterion(h_adv, y)
        else:
            cost = self.criterion(h_adv, y)

        self.net.zero_grad()
        if x_adv.grad is not None:
            x_adv.grad.data.fill_(0)
        cost.backward()

        x_adv = x_adv + eps*x_adv.grad.sign_()
        x_adv = torch.clamp(x_adv, x_val_min, x_val_max)
        return x_adv

    """
    BIM_a
    """
    # def i_fgsm_a(self, x, y, targeted=False, eps=8/255, alpha=1/255, iteration=1, x_val_min=0, x_val_max=1):
    #     x_adv = Variable(x.data, requires_grad=True)
    #     for i in range(iteration):
    #         h_adv = self.net(x_adv)
    #         _, predicted = h_adv.max(1)
    #         flag = (predicted != y).detach().cpu().numpy().astype(bool)
    #
    #         if targeted:
    #             cost = -self.criterion(h_adv, y)
    #         else:
    #             cost = self.criterion(h_adv, y)
    #
    #         self.net.zero_grad()
    #         if x_adv.grad is not None:
    #             x_adv.grad.data.fill_(0)
    #         cost.backward()
    #         # examples which have been misclassified won't update
    #         modify = alpha*x_adv.grad.sign_().detach().cpu().numpy()
    #         modify[flag] = 0
    #
    #         x_adv = x_adv + torch.from_numpy(modify).to(device)
    #         x_adv = where(x_adv > x-eps, x_adv, x-eps)
    #         x_adv = where(x_adv < x+eps, x_adv, x+eps)
    #         x_adv = torch.clamp(x_adv, x_val_min, x_val_max)
    #         x_adv = Variable(x_adv.data, requires_grad=True)
    #     return x_adv

    """
    BIM_b
    """
    def i_fgsm(self, x, y, targeted=False, eps=8/255, alpha=1/255, iteration=1, x_val_min=0, x_val_max=1):
        x_adv = Variable(x.data, requires_grad=True)
        for i in range(iteration):
            h_adv = self.net(x_adv)

            if targeted:
                cost = -self.criterion(h_adv, y)
            else:
                cost = self.criterion(h_adv, y)

            self.net.zero_grad()
            if x_adv.grad is not None:
                x_adv.grad.data.fill_(0)
            cost.backward()

            x_adv = x_adv + alpha*x_adv.grad.sign_()
            x_adv = where(x_adv > x-eps, x_adv, x-eps)
            x_adv = where(x_adv < x+eps, x_adv, x+eps)
            x_adv = torch.clamp(x_adv, x_val_min, x_val_max)
            x_adv = Variable(x_adv.data, requires_grad=True)
        return x_adv

    def i_fgsm_a(self, x, y, targeted=False, eps=8 / 255, alpha=1 / 255, iteration=1, x_val_min=0, x_val_max=1,
                       confidence=0.5):
        x_adv = Variable(x.data, requires_grad=True)
        for i in range(iteration):
            h_adv = self.net(x_adv)
            probs, predicted = F.softmax(h_adv, dim=-1).max(1)
            flag = (predicted != y).detach().cpu().numpy().astype(bool) & \
                   (probs >= confidence).detach().cpu().numpy().astype(bool)
            # print((probs > confidence).detach().cpu().numpy(),probs)

            if targeted:
                cost = -self.criterion(h_adv, y)
            else:
                cost = self.criterion(h_adv, y)

            self.net.zero_grad()
            if x_adv.grad is not None:
                x_adv.grad.data.fill_(0)
            cost.backward()
            # examples which have been misclassified won't update
            modify = alpha*x_adv.grad.sign_().detach().cpu().numpy()
            modify[flag] = 0

            x_adv = x_adv + torch.from_numpy(modify).cuda()
            x_adv = where(x_adv > x-eps, x_adv, x-eps)
            x_adv = where(x_adv < x+eps, x_adv, x+eps)
            x_adv = torch.clamp(x_adv, x_val_min, x_val_max)
            x_adv = Variable(x_adv.data, requires_grad=True)
        return x_adv


class Attack_MOVIE(object):
    def __init__(self, classify_net, criterion):
        self.net = classify_net
        self.criterion = criterion

    def fgsm(self, x, y, targeted=False, eps=8/255):
        x_adv = Variable(x.data, requires_grad=True)
        h_adv, _ = self.net(x_adv, after_embedding=True)
        if targeted:
            cost = -self.criterion(h_adv, y)
        else:
            cost = self.criterion(h_adv, y)

        self.net.zero_grad()
        if x_adv.grad is not None:
            x_adv.grad.data.fill_(0)
        cost.backward()

        x_adv = x_adv + eps*x_adv.grad.sign_()
        return x_adv


def where(cond, x, y):
    """
    code from :
        https://discuss.pytorch.org/t/how-can-i-do-the-operation-the-same-as-np-where/1329/8
    """
    cond = cond.float()
    return (cond*x) + ((1-cond)*y)


# def ShannonEntropy(logits, soft_label):
#     log_probs = F.log_softmax(logits, dim=-1)
#     H = - torch.sum(torch.mul(soft_label, log_probs), dim=-1)
#     return H

def ShannonEntropy(logits, soft_label):
    pred_probs = F.softmax(logits, dim=-1)
    H = torch.sum(torch.mul(soft_label, torch.log(soft_label/pred_probs)), dim=-1)
    return H