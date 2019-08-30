import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

NUM_CLASSES = 10
BATCH_SIZE = 256
BATCH_SIZE_MNIST = 256
BATCH_SIZE_MNIST_TEST = 1024 #1024
BATCH_SIZE_CIFAR10 = 32 # jsma 32 +cw 64
BATCH_SIZE_MINI_IMAGENET20 = 64

LOAD_CKPT = True # pre_train
NUM_EPOCHS = 100
LEARNING_RATE = 0.001
SCHEDULER_STEP_SIZE = 1000

EPS_MINIST = 0.15
EPS_CIFAR10 = 8/255

NUM_WORKERS = 8

CLASSIFY_CKPT = './checkpoint/resnet_adv.pth'

# conv-Copy1 53
MNIST_CKPT = './checkpoint/conv_adv.pth'

MOIVE_CKPT = 'checkpoint/model-Copy1.pth'
MOIVE_CKPT_ADV_TRAINING = 'checkpoint/model_adv.pth'

IMAGENET_CKPT = 'checkpoint/imageNet.pth'

MINI_IMAGENET_CKPT = 'checkpoint/miniImageNet.pth'
ADV_MINI_IMAGENET_CKPT = 'checkpoint/miniImageNet_adv.pth'

