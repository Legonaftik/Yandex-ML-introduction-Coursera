import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

data_train = pd.read_csv("salary-train.csv")
data_test = pd.read_csv("salary-test-mini.csv")

# to lower
data_train['FullDescription'] = data_train['FullDescription'].str.lower()
data_test['FullDescription'] = data_test['FullDescription'].str.lower()

# replacing
data_train['FullDescription'] = data_train['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex=True)
data_test['FullDescription'] = data_test['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex=True)

# td-if
td_if = TfidfVectorizer(min_df=6)
X_train_targets = td_if.fit_transform(data_train['FullDescription'])
X_test_targets = td_if.transform(data_test['FullDescription'])

# setting missed values
data_train['LocationNormalized'].fillna('nan', inplace=True)
data_train['ContractTime'].fillna('nan', inplace=True)
data_test['LocationNormalized'].fillna('nan', inplace=True)
data_test['ContractTime'].fillna('nan', inplace=True)

enc = DictVectorizer()
X_train_categ = enc.fit_transform(data_train[['LocationNormalized', 'ContractTime']].to_dict('records'))
X_test_categ = enc.transform(data_test[['LocationNormalized', 'ContractTime']].to_dict('records'))

train_vector = hstack([X_train_categ, X_train_targets])
test_vector = hstack([X_test_categ, X_test_targets])

clf = Ridge(alpha=1.0, fit_intercept=False, solver='lsqr')
clf.fit(train_vector, data_train["SalaryNormalized"])
result = clf.predict(test_vector)
print(result)  # 56054.93845996 37069.21183069
