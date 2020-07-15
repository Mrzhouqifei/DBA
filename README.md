# Detection by Attack: Detecting Adversarial Samples by Undercover Attack

## Description
This repository includes the source code of the paper "Detection by Attack: Detecting Adversarial Samples by Undercover Attack". Please cite our paper when you use this program! 😍

## DBA overview
![image.png](https://i.loli.net/2020/04/20/wtAj3ZT2kzN89gG.png)

## Quick glance
A quick glance of the use cases of undercover attack on images and texts:

```
example_images.ipynb
example_texts.ipynb
```

### Image case
![](https://i.loli.net/2019/11/22/R4jFAODNUegpLvs.png)

The Kullback–Leibler divergence ($D_KL$) of benign example:  1.1444092e-05;

$D_KL$ of adversarial example:  2.758562;

the median of $D_KL$ is 1.2505696;

1.1444092e-05 < 1.2505696 --> example1 is judged as a normal example;

2.757648 > 1.2505696 --> example2 is judged as an adversarial example.

### Text case 1
original sentiment: 1, changed words num: 2

original sentence:  touching well directed autobiography of a talented young director producer . a love story with rabin s assassination in the background . worth seeing !

adv sentence: touching well directed autobiography of a talented young director producer . a love story with rabin s assassination in the not . worth from !

original criterion loss: -0.00

adversarial criterion loss: 0.62

### Text case 2
original sentiment: 0, changed words num: 2

original sentence:  comment this movie is impossible . is terrible very improbable bad interpretation e direction . not look !

adv sentence: comment this movie is impossible . is terrible very improbable bad but e direction . it look !

$D_KL$ of original sentence: 0.00

$D_KL$ of adversarial sentence: 0.43

## Datasets

* [MNIST](http://yann.lecun.com/exdb/mnist/): torchvision.datasets.MNIST
* [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html): torchvision.datasets.CIFAR10
* [IMDB](https://www.kaggle.com/iarunava/imdb-movie-reviews-dataset)
* [Quora Question Pairs (QQP)](https://drive.google.com/file/d/0B0PlTAo--BnaQWlsZl9FZ3l1c28/view?usp=sharing)

## Attack methods
Attack methods are implemented in the folder ./adversary.

1. fgsm.py: FGSM, BIM
2. jsma.py: JSMA, JSMA*(ReplaceAttack)
3. cw.py: C&W

## Train
### Train models on normal examples

```
python normal_train/mnist_pre_train.py
```

### Undercover training, DBA

```
python mnist_adv.py
```

## Report issues
Please let us know if you encounter any problems.

The contact email is qifeizhou@pku.edu.cn


