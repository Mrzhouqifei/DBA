from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

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