import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

NUM_CLASSES = 10
BATCH_SIZE = 20

LOAD_CKPT = False # pre_train
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
SCHEDULER_STEP_SIZE = 20

EPS_MINIST = 0.3
EPS_CIFAR10 = 16/255

NUM_WORKERS = 8
# resnet       resnet_98.49%.pth     resnet_bima97.54%.pth
CLASSIFY_CKPT = './checkpoint/resnet_98.49%.pth'
# conv      conv_97.79%     conv_bima99.85%.pth
MNIST_CKPT = './checkpoint/conv_97.79%.pth'

MOIVE_CKPT = 'checkpoint/model-Copy1.pth'

MOIVE_CKPT_ADV_TRAINING = 'checkpoint/model_adv.pth'



