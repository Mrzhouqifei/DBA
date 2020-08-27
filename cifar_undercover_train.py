import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.autograd import Variable
import torchvision.transforms as transforms
from models.resnet import PreActResNet18
from adversary.fgsm import Attack


def undercover_attack(UndercoverAttack, x, y_true, eps=1/255):
    x = Variable(x.to(device), requires_grad=True)
    y_true = Variable(y_true.to(device), requires_grad=False)
    x_adv = UndercoverAttack.fgsm(x, y_true, False, eps)
    return x_adv


def train(epochs):
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True,
                                              num_workers=4)
    # Model
    print('==> Building model..')
    best_acc = 0.0
    net = PreActResNet18()
    net = net.to(device)
    checkpoint = torch.load(CIFAR_CKPT, map_location=torch.device(device))
    net.load_state_dict(checkpoint['net'])
    start_epoch = int(checkpoint['epoch'])
    best_acc = float(checkpoint['acc'])

    UndercoverAttack = Attack(net, nn.functional.cross_entropy)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    net.train()
    for epoch in range(start_epoch, epochs):
        train_loss = 0
        correct, total = 0, 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            x_adv = undercover_attack(UndercoverAttack, inputs, targets, eps=0.15)
            adv_outputs = net(x_adv)

            loss1 = criterion(outputs, targets)
            loss2 = criterion(adv_outputs, targets)
            loss = loss1 + loss2 * 0.8
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step(epoch)
        acc = 1.0 * correct / total
        print('epoch: %d, train loss: %.2f, train acc: %.4f' % (epoch, train_loss, acc))
        if acc > best_acc:
            best_acc = acc
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, CIFAR_CKPT)


def test():
    # Data
    print('==> Preparing data..')
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False,
                                             num_workers=4)

    # Model
    print('==> Building model..')
    net = PreActResNet18()
    net = net.to(device)
    criterion = nn.CrossEntropyLoss()
    checkpoint = torch.load(CIFAR_CKPT)
    net.load_state_dict(checkpoint['net'])

    net.eval()
    test_loss = 0
    correct, total = 0, 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    acc = 1.0 * correct / total
    print('test loss: %.2f, test acc: %.4f' % (test_loss, acc))


if __name__ == '__main__':
    CIFAR_CKPT = './checkpoint/cifar_undercover.pth'
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    train(150)
    test()
