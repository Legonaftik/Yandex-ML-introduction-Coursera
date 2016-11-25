from sklearn import datasets
import numpy as np
from sklearn.cross_validation import KFold
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer

newsgroups = datasets.fetch_20newsgroups(
                    subset='all',
                    categories=['alt.atheism', 'sci.space']
             )

data = newsgroups.data
target = newsgroups.target

clf = SVC(kernel="linear", random_state=241)
grid = {"C": np.power(10.0, np.arange(-5, 6))}
cv = KFold(len(target), n_folds=5, shuffle=True, random_state=241)
vectorizer = TfidfVectorizer()
vect_data = vectorizer.fit_transform(data)
# gs = GridSearchCV(clf, grid,  scoring="accuracy", cv=cv, n_jobs=-1)
# gs.fit(vect_data, target)
# for a in gs.grid_scores_:
#     print(a.mean_validation_score)  # Best minimal value: C=1.0, score: 0.993281075028 (C=1.0 - default value)
#     print(a.parameters)             # So we do not create a new classifier and do not transform data
clf.fit(vect_data, target)
feature_names = np.asarray(vectorizer.get_feature_names())
best_indices = np.argsort(
        np.absolute(np.asarray(clf.coef_.todense())).reshape(-1))[-10:]  # Indices of words with best absolute weights
best_words = sorted(feature_names[best_indices])
output_str = ""
for word in best_words:
    output_str += word + " "
output_str = output_str[:-1]  # Delete last space... Coursera...
with open("answer.txt", "w") as output_file:
    output_file.write(output_str)
