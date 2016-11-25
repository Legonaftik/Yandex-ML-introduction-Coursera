import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

prices = pd.read_csv("close_prices.csv", index_col="date")

pca = PCA(n_components=0.9)
pca.fit(prices)
print(len(pca.explained_variance_ratio_), "components are enough to save 90% dispersion")  # 4

transformed_prices = pca.transform(prices)
first_values = transformed_prices[:, 0]

indexes = pd.read_csv("djia_index.csv", index_col="date")
index_lst = indexes["^DJI"]
print("Correlation between PCA weights and Dow Jones Industrial Average:",
      np.corrcoef(first_values, index_lst)[0, 1])  # 0.909652219305

first_weights = pca.components_[0]
biggest_company_ind = first_weights.argmax()
print("The company with the biggest weight is:", prices.columns[biggest_company_ind])  # V
