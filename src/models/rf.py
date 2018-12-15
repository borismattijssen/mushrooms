import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from data import load_pca_data

def optimize(filepath):
    """ Find a random forest model with the best performing parameters.
    """

    # Number of trees in random forest
    n_estimators = [1400, 1600, 1800, 2000]
    # Maximum number of levels in tree
    max_depth = [None]
    # Minimum number of samples required to split a node
    min_samples_split = [2, 3, 4]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2]

    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf}

    # define a RF estimator
    rf = RandomForestClassifier()

    # set up randomized search with cross-validation
    clf = GridSearchCV(estimator = rf,
                       param_grid = random_grid,
                       cv = 10,
                       verbose=20,
                       scoring='f1',
                       n_jobs = -1)

    # fit on the data
    clf.fit(X, y)

    # save model to cache location
    pickle.dump(clf, open(filepath, "wb"))

    return (clf.best_estimator_, clf.best_score_)
