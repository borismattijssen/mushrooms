import pickle

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

from data import load_pca_data

def optimize(filepath):
    """ Find a k-nn model with the best performing `k`.
    """
    # find best k-nn model using grid search and 10-fold cross validation
    parameters = {'n_neighbors': range(5,51,4)}
    knn = KNeighborsClassifier()
    clf = GridSearchCV(knn, parameters, scoring='f1', cv=10, n_jobs=-1)
    (X, y) = load_pca_data()
    clf.fit(X, y)

    # save model to cache location
    pickle.dump(clf, open(filepath, "wb"))

    return (clf.best_estimator_, clf.best_score_)
