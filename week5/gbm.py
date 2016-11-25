import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt

data = pd.read_csv("gbm-data.csv")
data_np = data.as_matrix()
features = data_np[:, 1:]
target = data_np[:, 0]
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.8, random_state=241)
for rate in [0.2]:  # [1, 0.5, 0.3, 0.2, 0.1]
    clf = GradientBoostingClassifier(
            n_estimators=250, verbose=True, random_state=241, learning_rate=rate)  # "verbose" prints out each iteration
    clf.fit(X_train, y_train)

    log_loss_train = []
    # staged_decision_function - quality of built composition on each iteration
    for pred_train in clf.staged_decision_function(X_train):
        log_loss_train.append(log_loss(y_train, 1/(1 + np.exp(-pred_train))))
    log_loss_test = []
    for pred_test in clf.staged_decision_function(X_test):
        log_loss_test.append(log_loss(y_test, 1/(1 + np.exp(-pred_test))))

    # plots show us overfittng with increase of iterations
    plt.figure()
    plt.plot(log_loss_test, "r", linewidth=2, label="test")
    plt.plot(log_loss_train, "g", linewidth=2, label="train")
    plt.legend()
    # plt.show()

# We count all following parameters for learning_rate = 0.2

log_loss_min = min(log_loss_test)
log_loss_min_iter = log_loss_test.index(log_loss_min) + 1
print("Best quality is", log_loss_min, "on the", str(log_loss_min_iter)+"-th iteration")  # 0.530439819735 37

# for random forest
clf2 = RandomForestClassifier(n_estimators=log_loss_min_iter, random_state=241)
clf2.fit(X_train, y_train)
pred = clf2.predict_proba(X_test)
print("For Random Forest quality with the same amount of iterations is:", log_loss(y_test, pred))  # 0.540911909937
