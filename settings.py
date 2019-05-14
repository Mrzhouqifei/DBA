import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

NUM_CLASSES = 10
BATCH_SIZE = 128

LOAD_CKPT = True # pre_train
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
SCHEDULER_STEP_SIZE = 20

EPS_MINIST = 0.3
EPS_CIFAR10 = 16/255

NUM_WORKERS = 8
CLASSIFY_CKPT = './checkpoint/resnet-Copy1.pth'
MNIST_CKPT = './checkpoint/conv-Copy1.pth'




