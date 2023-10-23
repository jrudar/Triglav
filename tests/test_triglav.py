from triglav import Triglav, ETCProx, NoScale, Scaler, NoResample

from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import LinearSVC

import pandas as pd

import numpy as np

from pathlib import Path

# Checks for symmetry
def check_symmetric(X):

    if np.allclose(X, X.T, rtol = 1e-08, atol = 1e-08):
        pass

    else:
        raise ValueError("The matrix returned by ETCProx() is not symmetric")

dirpath = Path(__file__).parent

expected_output = dirpath / 'data/expected_output.csv'

# Tests the transformer and proximity modules
def test_transformers_prox():

    # Create the dataset
    X, y = make_classification(n_samples = 200,
                                n_features = 20,
                                n_informative = 5,
                                n_redundant = 2,
                                n_repeated = 0,
                                n_classes = 2,
                                shuffle = False,
                                random_state = 0)

    # To prevent negative proportions
    X = np.abs(X)

    # Ensures that the NoScale transformer returns the input
    R = NoScale().fit_transform(X)

    X = pd.DataFrame(X)
    R = pd.DataFrame(R)
    pd.testing.assert_frame_equal(X, R, check_dtype = False)

    # Ensures that Scaler returns the closure of X
    R = pd.DataFrame(Scaler().fit_transform(X.values))

    X_closure = X.values / X.values.sum(axis = 1)[:, None]
    X_closure = pd.DataFrame(X_closure)
    pd.testing.assert_frame_equal(X_closure, R, check_dtype = False)

    # Ensures that NoResample returns the input
    R = pd.DataFrame(NoResample().fit_transform(X))

    pd.testing.assert_frame_equal(X, R, check_dtype = False)

    # Ensure that ETCProx returns a square matrix and symmetric matrixo
    R = ETCProx().transform(X)

    assert R.shape[0] == R.shape[1]
    check_symmetric(R)

    print("Transformer and Proximity Tests Complete.")


# Tests the overall Triglav pipeline
def test_triglav_basic():

    #Create the dataset
    X, y = make_classification(n_samples = 200,
                                n_features = 20,
                                n_informative = 5,
                                n_redundant = 2,
                                n_repeated = 0,
                                n_classes = 2,
                                shuffle = False,
                                random_state = 0)

    #Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size = 0.2, 
                                                        random_state = 0, 
                                                        stratify = y)

    #Set up Triglav
    model = Triglav(n_jobs = 4,
                    verbose = 3,
                    n_iter_fwer=5,
                    estimator = ExtraTreesClassifier(512, bootstrap = True, max_depth = 4),
                    stage_2_estimator = ExtraTreesClassifier(512, bootstrap = True, max_depth = 4),
                    metric = "euclidean",
                    criterion = "maxclust",
                    thresh = 9,
                    run_stage_2="auto",
                    transformer=StandardScaler())

    model = Triglav(n_jobs = 4,
                    verbose = 3,
                    n_iter_fwer=5,
                    estimator = ExtraTreesClassifier(512, bootstrap = True, max_depth = 4),
                    stage_2_estimator = ExtraTreesClassifier(512, bootstrap = True, max_depth = 4),
                    metric = "euclidean",
                    criterion = "maxclust",
                    thresh = 9,
                    run_stage_2="mms",
                    transformer=StandardScaler())

    #Identify predictive features
    model.fit(X_train, y_train)

    features_selected = model.selected_
    features_best = model.selected_best_

    from sklearn.metrics import balanced_accuracy_score
    s1 = ExtraTreesClassifier(512).fit(X_train[:, model.selected_], y_train).predict(X_test[:, model.selected_])
    s2 = ExtraTreesClassifier(512).fit(X_train[:, model.selected_best_], y_train).predict(X_test[:, model.selected_best_])

    s1, s2 = balanced_accuracy_score(y_test, s1), balanced_accuracy_score(y_test, s2)

    assert s1 > 0.75

    assert s2 > 0.75

    print("Triglav Test Complete.")


if __name__ == "__main__":
    test_transformers_prox()
    test_triglav_basic()

    fdfd = 5

