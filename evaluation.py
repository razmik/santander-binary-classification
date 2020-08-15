"""
Author: Rashmika Nawaratne
Date: 15-Aug-20 at 11:58 AM
"""
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, confusion_matrix


class Evaluator(object):

    @staticmethod
    def evaluate(y_true, y_predicted):

        print("area under the ROC curve:", roc_auc_score(y_true, y_predicted))
        print(classification_report(y_true, y_predicted))
        print(confusion_matrix(y_true, y_predicted))

        auc = roc_auc_score(y_true, y_predicted)
        fpr, tpr, thresholds = roc_curve(y_true, y_predicted, pos_label=1)

        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1], '--')
        plt.xlim(0, 1.01)
        plt.ylim(0, 1.01)
        plt.title('AUC: {}'.format(round(auc, 2)))
        plt.savefig('roc_auc_curve.jpg', dpi=300)

