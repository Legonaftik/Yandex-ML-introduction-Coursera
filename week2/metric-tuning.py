import sklearn
import sklearn.datasets
import sklearn.cross_validation
import sklearn.neighbors
from numpy import linspace

data = sklearn.datasets.load_boston()
features = data["data"]
target = data["target"]
scaled = sklearn.preprocessing.scale(features)
kf = sklearn.cross_validation.KFold(n=len(target), n_folds=5, shuffle=True, random_state=42)
results = {}
for i in linspace(1, 10, 200):
    kr = sklearn.neighbors.KNeighborsRegressor(n_neighbors=5,
                                               weights="distance", metric="minkowski", p=i)
    score = sklearn.cross_validation.cross_val_score(
            kr, scaled, target, cv=kf, scoring='mean_squared_error')
    results[i] = score.mean()
best_par = (max(results, key=results.get))
# Best parameter for Minkowski metric
print("p =", str(best_par) + ", value =", str(results[best_par]))  # p = 1.0, value = -16.0502085084
