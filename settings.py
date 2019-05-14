import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

NUM_CLASSES = 10
EMBEDDINGS_SIZE = 32
BATCH_SIZE = 50

LOAD_CKPT = False
NUM_EPOCHS = 40
LEARNING_RATE = 0.001
SCHEDULER_STEP_SIZE = 20

EPS_MINIST = 0.3
EPS_CIFAR10 = 16/255

NUM_WORKERS = 8
CLASSIFY_CKPT = './checkpoint/resnet_100-Copy1.pth'
MNIST_CKPT = './checkpoint/conv_fgsm97.87.pth'




