import pandas
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale

data = pandas.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data", header=None)
features = data.ix[:, 1:]
klass = data.ix[:, 0]
kf = KFold(n=data.shape[0], n_folds=5, shuffle=True, random_state=42)

results = {}
for k in range(1, 51):
    ks = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(ks, features, klass, cv=kf)
    results[k] = score.mean()
best_k = max(results, key=results.get)
# Returns the number of nearest neighbors with the best quality
print("k =", str(best_k) + ", value =", str(results[best_k]))  # k = 1, value = 0.730476190476

scaled = scale(features)
scaled_results = {}
for k in range(1, 51):
    ks = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(ks, scaled, klass, cv=kf)
    scaled_results[k] = score.mean()

best_scaled = (max(scaled_results, key=scaled_results.get))
# Best neighbors number after scaling
print("k =", str(best_scaled) + ", value =",
      str(scaled_results[best_scaled]))  # k = 29, value = 0.977619047619
