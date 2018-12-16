import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, roc_auc_score, classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import pickle
import os.path
from data import load_pca_data

def optimize_nn(use_cache, cache_file):
    """ Finds the best NN model parameters by Grid Search CV
    """
    # if we should load from cache and the cache file exists
    if (use_cache and os.path.exists(cache_file) and os.path.isfile(cache_file)):
        clf = pickle.load(open(cache_file, "rb"))
        return (clf.best_estimator_, clf.best_score_)

    parameters = [{'hidden_layer_sizes': [(30,), (50,), (50, 10), (100, 10, 10)],
                   'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10], 'max_iter': range(50, 200, 50)}]

    nn = MLPClassifier()
    clf = GridSearchCV(nn, parameters, scoring='f1', cv=10, n_jobs=-1)
    (X, y) = load_pca_data()
    clf.fit(X, y)

    # save model to cache location
    pickle.dump(clf, open(cache_file, "wb"))

    return (clf.best_estimator_, clf.best_score_)
