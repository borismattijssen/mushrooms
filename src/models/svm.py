import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, roc_auc_score, classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import pickle
import os.path
from data import load_pca_data

def optimize_svm(use_cache, cache_file):
    """ Finds the best SVC model parameters by Grid Search CV
    """
    # if we should load from cache and the cache file exists
    if (use_cache and os.path.exists(cache_file) and os.path.isfile(cache_file)):
        clf = pickle.load(open(cache_file, "rb"))
        return (clf.best_estimator_, clf.best_score_)

    # find best k-nn model using grid search and 10-fold cross validation
    parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                         'C': [0.1, 1]},
                        {'kernel': ['linear'], 'C': [0.01, 0.1, 1]},
                        {'kernel': ['poly'], 'C': [0.01, 0.1], 'degree': [2, 3, 4], 'coef0': [0.1, 0.5]}]

    svm = SVC()
    clf = GridSearchCV(svm, parameters, scoring='f1', cv=10, n_jobs=-1)
    (X, y) = load_pca_data()
    clf.fit(X, y)

    # save model to cache location
    pickle.dump(clf, open(cache_file, "wb"))

    return (clf.best_estimator_, clf.best_score_)
