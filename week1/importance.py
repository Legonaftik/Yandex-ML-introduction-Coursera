import pandas
from sklearn.tree import DecisionTreeClassifier
data = pandas.read_csv("titanic.csv", index_col="PassengerId")

choice = data[["Pclass", "Fare", "Age", "Sex", "Survived"]]
choice["Sex"] = choice["Sex"].replace("male", 1).replace("female", 0)
choice = choice.dropna()
target = choice[["Survived"]]
del choice["Survived"]


clf = DecisionTreeClassifier()
clf.fit(choice, target)
importances = clf.feature_importances_
print(importances)  # Fare,Sex
