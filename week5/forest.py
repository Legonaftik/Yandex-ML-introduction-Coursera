import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score

data = pd.read_csv("abalone.csv")
data['Sex'] = data['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))
features = data.ix[:, :-1]
target = data.ix[:, -1]
kf = KFold(len(target), n_folds=5, shuffle=True, random_state=1)
for num in range(1, 51):
    clf = RandomForestRegressor(n_estimators=num, random_state=1)
    clf.fit(features, target)
    score = cross_val_score(clf, features, target, cv=kf, scoring="r2")
    if score.mean() > 0.52:
        print("The least appropriate number of trees is", num,
              "with R2-score =", score.mean())  # 22 - 0.520158353672
        break
