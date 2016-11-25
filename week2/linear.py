import pandas
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

train_data = pandas.read_csv("perceptron-train.csv", header=None)
train_features = train_data.ix[:, 1:]
train_target = train_data.ix[:, 0]
test_data = pandas.read_csv("perceptron-test.csv", header=None)
test_features = test_data.ix[:, 1:]
test_target = test_data.ix[:, 0]

perceptron = Perceptron(random_state=241)
perceptron.fit(train_features, train_target)
first_prediction = perceptron.predict(test_features)
# Accuracy without normalisation
first_result = accuracy_score(test_target, first_prediction)  # 0.36

scaler = StandardScaler()
train_features_scaled = scaler.fit_transform(train_features)
test_features_scaled = scaler.transform(test_features)  # no "fit_" because we use coefficients from train
perceptron.fit(train_features_scaled, train_target)
final_prediction = perceptron.predict(test_features_scaled)
# Accuracy after normalisation
scaled_result = accuracy_score(test_target, final_prediction)  # 0.925

print("Accuracy improvement after normalisation =", scaled_result - first_result)  # 0.565
