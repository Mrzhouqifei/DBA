import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
import torchvision.transforms as transforms
import os
from models.mnist_model import MnistModel


def train(epochs):
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True,
                                              num_workers=4)
    # Model
    print('==> Building model..')
    net = MnistModel()
    net = net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)

    net.train()
    best_acc = 0.0
    for epoch in range(epochs):
        train_loss = 0
        correct, total = 0, 0
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
            torch.save(state, MNIST_CKPT)


def test():
    # Data
    print('==> Preparing data..')
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False,
                                             num_workers=4)

    # Model
    print('==> Building model..')
    net = MnistModel()
    net = net.to(device)
    criterion = nn.CrossEntropyLoss()
    checkpoint = torch.load(MNIST_CKPT)
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
    MNIST_CKPT = './checkpoint/mnist.pth'
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

    train(50)
    test()
