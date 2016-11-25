import pandas as pd
import sklearn.metrics
import matplotlib.pyplot as plt
data = pd.read_csv("classification.csv")
true = data.ix[:, 0]
predict = data.ix[:, 1]

TP=0; FP=0; FN=0; TN=0
for i in range(len(data[[0]])):
    if data.ix[i, 0] == data.ix[i, 1] == 1:
        TP += 1
    elif data.ix[i, 0] == data.ix[i, 1] == 0:
        TN += 1
    elif data.ix[i, 0] == 0:
        FP += 1
    else:
        FN += 1
# sklearn.metrics.confusion_matrix(data.ix[:, 0], data.ix[:, 1], labels=[1, 0])  # <== or this method
print("TP =", TP, "FP =", FP, "FN =", FN, "TN =", TN)    # TP = 43 FP = 34 FN = 59 TN = 64

print("-"*80)
print("Accuracy =", sklearn.metrics.accuracy_score(true, predict))     # 0.54
print("Precision =", sklearn.metrics.precision_score(true, predict))   # 0.56
print("Recall =", sklearn.metrics.recall_score(true, predict))         # 0.42
print("F1 =", sklearn.metrics.f1_score(true, predict))                 # 0.48

print("-"*80)
scores = pd.read_csv("scores.csv")
print("Area Under the ROC Curve :")
print("logreg =", sklearn.metrics.roc_auc_score(scores.ix[:, 0], scores.ix[:, 1]))  # 0.71918767507 - the best
print("svm =", sklearn.metrics.roc_auc_score(scores.ix[:, 0], scores.ix[:, 2]))     # 0.708683473389
print("knn =", sklearn.metrics.roc_auc_score(scores.ix[:, 0], scores.ix[:, 3]))     # 0.635154061625
print("tree =", sklearn.metrics.roc_auc_score(scores.ix[:, 0], scores.ix[:, 4]))     # 0.691926770708

scores = pd.read_csv('scores.csv')
ytrue = scores['true']
for algorithm in scores.columns[1:]:
    yscores = scores[algorithm]
    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(ytrue, yscores)
    plt.plot(recall, precision, label=algorithm)
    plt.xlim(0.7, 1)
    plt.ylim(0.5, 0.7)
plt.legend()
plt.show()
