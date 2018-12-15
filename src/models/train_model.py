import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import preprocessing, metrics
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, roc_auc_score, classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


def train_svm(x_train, y_train, x_test, y_test):

    log_file = open('log_SVM' + '.txt', 'w')
    scorers = {
        'precision_score': make_scorer(precision_score),
        'recall_score': make_scorer(recall_score),
        'accuracy_score': make_scorer(accuracy_score)
    }
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                         'C': [0.01, 0.1, 1]},
                        {'kernel': ['linear'], 'C': [0.01, 0.1, 1]},
                        {'kernel': ['poly'], 'C': [0.01, 0.1], 'degree': [2, 3, 4], 'coef0': [0, 0.1, 0.5]}]

    for score in scorers:

        print("# Tuning hyper-parameters for %s" % score, file=log_file)
        print()

        clf = GridSearchCV(SVC(), tuned_parameters, cv=10, scoring=scorers(score))
        clf.fit(x_train, y_train)

        print("Best parameters set found on development set:", file=log_file)
        print()
        print(clf.best_params_, file=log_file)
        print()
        print("Grid scores on development set:", file=log_file)
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params), file=log_file)
        print()

        print("Detailed classification report:", file=log_file)
        print()
        y_true, y_pred = y_test, clf.predict(x_test)
        report = classification_report(y_true, y_pred)
        print(report, file=log_file)
        acc = accuracy_score(y_true, y_pred, normalize=True)
        rauc = roc_auc_score(y_true, y_pred, )
        print("Accuracy score: %f, ROC AUC score: %f" % (acc, rauc), file=log_file)
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()


if __name__ == '__main__':

    df = pd.read_csv('../../data/raw/mushrooms_v2.csv')

    # preprocess
    y = df['class']
    print(y)
    lb = preprocessing.LabelBinarizer()
    y = lb.fit_transform(y)
    X = pd.get_dummies(df.drop('class', axis=1))
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y.ravel(), test_size=.2, random_state=42)

    train_svm(X_train, y_train, X_test, y_test)
