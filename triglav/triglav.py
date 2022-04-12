import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")

import numpy as np
from numpy.random import PCG64

from sklearn.utils import check_random_state, check_X_y
from sklearn.base import TransformerMixin, BaseEstimator, clone
from sklearn.feature_selection import mutual_info_classif, VarianceThreshold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegressionCV, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import make_scorer, balanced_accuracy_score

from statsmodels.stats.multitest import multipletests

import shap as sh
import sage

from joblib import Parallel, delayed

from scipy.stats import betabinom

from math import ceil

from functools import partial

import typing


def sage_scores(X, y, S, T):

    X_comb = get_shadow(X[:, S])

    M = ExtraTreesClassifier(128, max_depth=5).fit(X_comb, y)

    imputer = sage.MarginalImputer(M, X_comb)

    estimator = sage.SignEstimator(imputer)

    sage_values = estimator(X_comb, y, bar=False)

    mu_real = sage_values.values[: X[:, S].shape[1]]

    mu_shadow = sage_values.values[X[:, S].shape[1] :]

    S_new = mu_real > np.percentile(mu_shadow, T)

    S_idx = np.asarray([i for i in range(S.shape[0])])
    S_idx_red = S_idx[S]

    S_final = np.zeros(shape=(S.shape[0]), dtype=bool)
    for i, loc in enumerate(S_idx_red):
        S_final[loc] = S_new[i]

    return S_final


def shap_scores(M, A, B):

    explainer = sh.Explainer(M, masker=sh.maskers.Independent(A))

    s = np.abs(explainer(B).values)

    if s.ndim > 2:
        s = s.mean(axis=2)

    s = s.mean(axis=0)

    return s


def get_shadow(X):

    rnd = np.random.Generator(PCG64())
    X_perm = np.copy(X, "C")
    for col in range(X.shape[1]):
        rnd.shuffle(X_perm[:, col])

    X_comb = np.hstack((X, X_perm))

    return X_comb


def par_train(mod, X, y):

    X_comb = get_shadow(X)

    S = []

    X_1, X_2, y_1, y_2 = train_test_split(X_comb, y, train_size=0.5, stratify=y)

    clf = mod.fit(X_1, y_1)

    try:
        clf = clf.best_estimator_
    except:
        pass

    S.append(shap_scores(clf, X_1, X_2))

    clf = mod.fit(X_2, y_2)

    try:
        clf = clf.best_estimator_
    except:
        pass

    S.append(shap_scores(clf, X_2, X_1))

    return np.mean(S, axis=0)


def model_fun(X, y, mod, threshold):

    S = Parallel(4)(
        delayed(par_train)(
            clone(mod),
            X,
            y,
        )
        for _ in range(10)
    )

    S = np.asarray(S).mean(axis=0)
    S_o = S[: X.shape[1]]
    S_s = S[X.shape[1] :]

    S = S_o > np.percentile(S_s, threshold)

    return S


def mi(X, y, threshold):
    X_comb = get_shadow(X)

    N_Ne = np.random.choice([3, 4, 5, 6, 7], size=1)[0]

    m_i = mutual_info_classif(X_comb, y, discrete_features=True, n_neighbors=N_Ne)

    S = m_i[: X.shape[1]] > np.percentile(m_i[X.shape[1] :], threshold)

    return S


def get_hits(X, y, threshold):

    class_set = np.unique(y)
    n_class = class_set.shape[0]

    if n_class > 2:
        loss = "mlogloss"

    else:
        loss = "logloss"

    selection = set(np.random.choice([i for i in range(5)], 2, replace=False))

    score_fun = make_scorer(balanced_accuracy_score)

    # Create a list of models to execute
    models = {
        0: partial(
            model_fun,
            mod=GridSearchCV(
                SGDClassifier(loss="perceptron", max_iter=3000),
                param_grid={"alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100]},
                cv=5,
                scoring=score_fun,
            ),
            threshold=threshold,
        ),
        1: partial(
            model_fun, mod=ExtraTreesClassifier(384, max_depth=5), threshold=threshold
        ),
        2: partial(
            model_fun,
            mod=GridSearchCV(
                LinearSVC(max_iter=3000),
                param_grid={"C": [0.001, 0.01, 0.1, 1.0, 10.0, 100]},
                cv=5,
                scoring=score_fun,
            ),
            threshold=threshold,
        ),
        3: partial(
            model_fun, mod=LogisticRegressionCV(max_iter=3000), threshold=threshold
        ),
        4: partial(mi, threshold=threshold),
    }

    H = [models[t](X, y) for t in selection]

    if H[0].sum() < H[1].sum():
        return H[0]

    else:
        return H[1]


def fs(X, y, threshold):

    class_set = np.unique(y)
    n_class = class_set.shape[0]

    # Get the index of the original features
    F_ori = np.asarray([i for i in range(X.shape[1])])

    # Remove all columns where features do not vary
    rem = VarianceThreshold().fit(X)
    X_filt = rem.transform(X)
    F_ori = rem.transform([F_ori])[0]

    if X_filt.shape[1] == 0:
        return np.zeros(shape=(X.shape[1],), dtype=bool)

    # Calculate the initial scores
    S = get_hits(X_filt, y, threshold)
    if S.sum() == 0:
        return np.zeros(shape=(X.shape[1],), dtype=bool)

    # Calculate SAGE scores
    S_sage = sage_scores(X_filt, y, S, threshold)
    if S_sage.sum() == 0:
        return np.zeros(shape=(X.shape[1],), dtype=bool)

    S_final = rem.inverse_transform([S])[0]

    return S_final


def binomial_test(X, C, alpha, a, b):

    THRESHOLD = alpha / C

    beta_bin = betabinom(n=X.shape[0], a=a, b=b)

    if X.ndim == 1:

        P_hit = beta_bin.sf(X[:, col].sum() - 1)
        P_rej = beta_bin.cdf(X[:, col].sum() - 1)

        if P_hit <= THRESHOLD:
            P_hit = np.asarray([True])
            P_rej = np.asarray([False])

        else:
            if P_rej <= THRESHOLD:
                P_rej = np.asarray([True])
                P_hit = np.asarray([False])

    elif X.ndim == 2:
        loc = np.where(X.sum(axis=0) > 0, True, False)
        cols = np.asarray([i for i in range(X.shape[1])])

        cols = cols[loc]

        P_hit = []
        P_rej = []
        for col in cols:
            P_hit.append(beta_bin.sf(X[:, col].sum() - 1))
            P_rej.append(beta_bin.cdf(X[:, col].sum() - 1))

        P_hit = np.asarray(P_hit)
        P_rej = np.asarray(P_rej)

        # Correct for comparisons within iteration followed by correction across iterations
        P_hit = multipletests(P_hit, alpha, method="fdr_tsbky")[1]
        P_rej = multipletests(P_rej, alpha, method="fdr_tsbky")[1]

        # Correct for comparisons across iterations
        P_hit = P_hit <= THRESHOLD
        P_rej = P_rej <= THRESHOLD

        # Restore original size
        P_H = np.zeros(shape=(X.shape[1]), dtype=bool)
        P_R = np.ones(shape=(X.shape[1]), dtype=bool)
        for i, col in enumerate(cols):
            P_H[col] = P_hit[i]
            P_R[col] = P_rej[i]

    return P_H, P_R


def sage_testing(X, y, S, T, threshold):

    F_set = S.union(T)

    F_set = list(F_set)
    F_set.sort()
    F_set = np.asarray(F_set)

    V = []

    rnd = np.random.Generator(PCG64())
    X_perm = np.copy(X[:, F_set], "C")
    for col in range(X_perm.shape[1]):
        rnd.shuffle(X_perm[:, col])

    X_comb = np.hstack((X[:, F_set], X_perm))

    X_1, X_2, y_1, y_2 = train_test_split(X_comb, y, train_size=0.5, stratify=y)

    # For X_1
    model = ExtraTreesClassifier(384, max_depth=5).fit(X_1, y_1)
    imputer = sage.MarginalImputer(model, X_2)
    estimator = sage.PermutationEstimator(imputer)
    V_1 = estimator(X_2, y_2, bar=False).values

    # For X_2
    model = ExtraTreesClassifier(384, max_depth=5).fit(X_2, y_2)
    imputer = sage.MarginalImputer(model, X_1)
    estimator = sage.PermutationEstimator(imputer)
    V_2 = estimator(X_1, y_1, bar=False).values

    V.append(np.vstack((V_1, V_2)).mean(axis=0))

    V = np.mean(V, axis=0)

    V = V[: len(F_set)] > np.percentile(V[len(F_set) :], threshold)

    return V


def select_features(
    X,
    y,
    is_binary,
    max_iter,
    alpha,
    threshold,
    threshold_2,
    a,
    b,
    a_2,
    b_2,
    verbose,
    n_jobs,
):

    # Prepare parameters and tracking dictionaries
    IDX = np.asarray([i for i in range(X.shape[1])])
    COMPARISONS = 1

    F_accepted = set()
    F_rejected = set()
    F_tentative = set()

    T_idx = IDX.copy("C")

    # Stage 1: Calculate significance of features in a similar manner as Boruta
    if verbose > 0:
        print("Starting Stage 1...")
        print("Identifying an initial set of features...")

    # Calculate how often features are selected by various algorithms
    H = Parallel(n_jobs)(delayed(fs)(X, y, threshold) for _ in range(10))

    H = np.asarray(H)

    if verbose > 0:
        print("Refining features...")

    for i in range(max_iter):
        if len(T_idx) > 2:

            P_H = np.zeros(shape=(H.shape[1]), dtype=bool)
            P_R = np.zeros(shape=(H.shape[1]), dtype=bool)

            P_h, P_r = binomial_test(H[:, T_idx], COMPARISONS, alpha, a, b)

            for k, loc in enumerate(T_idx):
                P_H[loc] = P_h[k]
                P_R[loc] = P_r[k]

            # Get index of hits
            F_accepted = F_accepted.union(set(IDX[P_H]))

            # Get index of rejections
            F_rejected = F_rejected.union(set(IDX[P_R]))

            # Get index of tentative
            F_tentative = set(IDX) - F_accepted.union(F_rejected)

            # Update T_idx
            T = list(F_tentative)
            T.sort()
            T_idx = T

            if verbose > 0:
                print(
                    "Iteration:",
                    i + 1,
                    "/ Confirmed:",
                    len(F_accepted),
                    "/ Tentative:",
                    len(F_tentative),
                    "/ Rejected:",
                    len(F_rejected),
                    sep=" ",
                )

            # Update the comparisons tracker
            if len(T_idx) > 2:
                COMPARISONS += 1

                # Add a new set of hits
                H_tmp = fs(X[:, T], y, threshold)

                H_new = np.zeros(shape=(H.shape[1]), dtype=bool)
                for i, entry in enumerate(T_idx):
                    H_new[entry] = H_tmp[i]

                H = np.vstack((H, H_new))

    # Stage 2: Use SAGE to find final set of features
    if verbose > 0:
        print("Starting Stage 2...")
        print("Using SAGE with remaining features...")

    H = Parallel(n_jobs)(
        delayed(sage_testing)(X, y, F_accepted, F_tentative, threshold_2)
        for _ in range(15)
    )

    H = np.asarray(H)

    P_h, _ = binomial_test(H, 1, 0.05, a_2, b_2)

    if verbose > 0:
        print("Finalizing...")

    F_set = F_accepted.union(F_tentative)
    F_set = list(F_set)
    F_set.sort()
    F_set = np.asarray(F_set)
    F_accepted = F_set[P_h]

    # Create and populate array using the original transfomed dimensions
    S = np.zeros(shape=(X.shape[1],), dtype=bool)
    for i, loc in enumerate(F_accepted):
        S[IDX[loc]] = True

    if verbose > 0:
        print("Final Confirmed Contains %s Features." % str(len(F_accepted)))

    return S


# Class which combines the above method into a nice interface
class Triglav(TransformerMixin, BaseEstimator):
    """
    Inputs:

    threshold and threshold_2: int, default = 98
        The threshold for comparing shadow and real features in the
        first and second stage.

    a and a_2: float, default = 24 / 20
        The 'a' parameter of the Beta distribution at stage 1 and 2.

    b and b_2: float, default = 32 / 32
        The 'b' parameter of the Beta distribution at stage 1 and 2.

    alpha: float, default = 0.025
        The level at which corrected p-values will be rejected.

    max_iter: int, default = 50
        The maximum number of iterations.

    verbose: int, default = 0
        Specifies if basic reporting is sent to the user.

    n_jobs: int, default = 5
        The number of threads

    Returns:

    An Triglav object.
    """

    def __init__(
        self,
        threshold: int = 95,
        threshold_2: int = 95,
        a: float = 24,
        b: float = 32,
        a_2: float = 20,
        b_2: float = 32,
        alpha: float = 0.025,
        max_iter: int = 50,
        verbose: int = 0,
        n_jobs: int = 5,
    ):

        self.threshold = threshold
        self.threshold_2 = threshold_2
        self.alpha = alpha
        self.a = a
        self.b = b
        self.a_2 = a_2
        self.b_2 = b_2
        self.max_iter = max_iter
        self.verbose = verbose
        self.n_jobs = n_jobs

    def fit(self, X, y):
        """
        Inputs:

        X: NumPy array of shape (m, n) where 'm' is the number of samples and 'n'
        the number of features (taxa, OTUs, ASVs, etc).

        y: NumPy array of shape (m,) where 'm' is the number of samples. Each entry
        of 'y' should be a factor.

        Returns:

        A fitted mECFS object.
        """

        X_in, y_in = self._check_params(X, y)

        self.classes_, y_int_ = np.unique(y_in, return_inverse=True)
        self.n_class_ = self.classes_.shape[0]

        # Check if the features are binary
        self.is_binary_ = np.array_equal(X_in, X.astype(bool))

        # Get the base distribution of EC scores
        self.selected_ = select_features(
            X_in,
            y_int_,
            self.is_binary_,
            self.max_iter,
            self.alpha,
            self.threshold,
            self.threshold_2,
            self.a,
            self.b,
            self.a_2,
            self.b_2,
            self.verbose,
            self.n_jobs,
        )

        return self

    def transform(self, X):
        """
        Input:

        X: NumPy array of shape (m, n) where 'm' is the number of samples and 'n'
        the number of features (taxa, OTUs, ASVs, etc).

        Returns:

        X_transformed: NumPy array of shape (m, p) where 'm' is the number of samples and 'p'
        the number of features (taxa, OTUs, ASVs, etc). 'p' <= m
        """
        return X[:, self.selected_]

    def fit_transform(self, X, y):

        self.fit(X, y)

        return self.transform(X)

    def _check_params(self, X, y):

        # Check if X and y are consistent
        X_in, y_in = check_X_y(X, y)

        # Basic check on parameter bounds
        if (self.threshold <= 0 or self.threshold > 100) or (
            self.threshold_2 <= 0 or self.threshold_2 > 100
        ):
            raise ValueError("The 'threshold' parameter should be between 1 and 100.")

        if (self.a < 0 or self.b < 0) or (self.a_2 < 0 or self.b_2 < 0):
            raise ValueError("The 'a' and 'b' parameters should be greater than 0.")

        if self.alpha <= 0 or self.alpha > 1:
            raise ValueError("The 'alpha' parameter should be between 0 and 1.")

        if self.verbose < 0:
            raise ValueError(
                "The 'verbose' parameter should be greater than or equal to zero."
            )

        return X_in, y_in
