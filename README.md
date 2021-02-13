# Detection by Attack: Detecting Adversarial Samples by Undercover Attack

## Description
This repository includes the source code of the paper "Detection by Attack: Detecting Adversarial Samples by Undercover Attack". Please cite our paper when you use this program! 😍 This paper has been accepted to the conference "European Symposium on Research in Computer Security (ESORICS20)". This paper can be downloaded [here](https://link.springer.com/chapter/10.1007/978-3-030-59013-0_8).

```
@inproceedings{zhou2020detection,
  title={Detection by attack: Detecting adversarial samples by undercover attack},
  author={Zhou, Qifei and Zhang, Rong and Wu, Bo and Li, Weiping and Mo, Tong},
  booktitle={European Symposium on Research in Computer Security},
  pages={146--164},
  year={2020},
  organization={Springer}
}
```

## DBA overview
![image.png](https://i.loli.net/2020/04/20/wtAj3ZT2kzN89gG.png)

The pipeline of our framework consists of two steps:
1. Injecting adversarial samples to train the classification model.
2. Training a simple multi-layer perceptron (MLP) classifier to judge whether the sample is adversarial.

We take MNIST and CIFAR as examples: the mnist_undercover_train.py and cifar_undercover_train.py refer to the step one; the mnist_DBA.ipynb and cifar_DBA.ipynb refer to the step two.

## Report issues
Please let us know if you encounter any problems.

The contact email is qifeizhou@pku.edu.cn
