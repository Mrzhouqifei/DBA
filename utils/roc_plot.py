from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def roc_auc(labels, losses):
    fpt, tpt, thresholds = roc_curve(labels, losses)
    roc_auc = auc(fpt, tpt)
    plt.switch_backend('Agg')
    fig = plt.figure()
    lw = 2
    plt.plot(fpt, tpt, color='red',
             lw=lw, label='ROC curve (auc = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='blue', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('adversarial detect roc curve')
    plt.legend(loc="lower right")
    fig.savefig('./output/roc.png', dpi=fig.dpi)

    return roc_auc

def creterion_func(benign_losses, adv_losses):
    benign_losses = benign_losses[:300]
    adv_losses = adv_losses[:300]
    creterion = pd.DataFrame([benign_losses, adv_losses])
    creterion.to_csv('./output/creterion.csv', index=False)
    fig = plt.figure()
    plt.scatter(np.arange(len(benign_losses)), benign_losses, color='cornflowerblue', s=3, marker='o')
    plt.scatter(np.arange(len(adv_losses)), adv_losses, color='crimson', s=3, marker='*')
    plt.xticks([])
    # plt.yticks([])
    fig.savefig('creterion.png', dpi=400)
    plt.show()
