"""fgsm.py"""
import torch
from torch.autograd import Variable
import numpy as np
from settings import *


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

    """
    BIM_a
    """
    def i_fgsm_a(self, x, y, targeted=False, eps=8/255, alpha=1/255, iteration=1, x_val_min=0, x_val_max=1):
        x_adv = Variable(x.data, requires_grad=True)
        for i in range(iteration):
            h_adv = self.net(x_adv)
            _, predicted = h_adv.max(1)
            flag = (predicted != y).detach().cpu().numpy().astype(bool)

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

            x_adv = x_adv + torch.from_numpy(modify).to(device)
            x_adv = where(x_adv > x-eps, x_adv, x-eps)
            x_adv = where(x_adv < x+eps, x_adv, x+eps)
            x_adv = torch.clamp(x_adv, x_val_min, x_val_max)
            x_adv = Variable(x_adv.data, requires_grad=True)
        return x_adv

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

def where(cond, x, y):
    """
    code from :
        https://discuss.pytorch.org/t/how-can-i-do-the-operation-the-same-as-np-where/1329/8
    """
    cond = cond.float()
    return (cond*x) + ((1-cond)*y)