import pandas
from sklearn.svm import SVC

data = pandas.read_csv("svm-data.csv", header=None)
features = data.ix[:, 1:]
target = data.ix[:, 0]

classifier = SVC(C=100000, kernel="linear", random_state=241)
classifier.fit(features, target)

print("Numbers of support vectors: ", end="")
for index in classifier.support_:
    print(index+1, end=" ")
