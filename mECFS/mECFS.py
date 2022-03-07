"""
    Adapted From:
    https://www.mathworks.com/matlabcentral/fileexchange/56937-feature-selection-library

    Reference:
    @InProceedings{RoffoECML16, 
     author={G. Roffo and S. Melzi}, 
     booktitle={Proceedings of New Frontiers in Mining Complex Patterns (NFMCP 2016)}, 
     title={Features Selection via Eigenvector Centrality}, 
     year={2016}, 
     keywords={Feature selection;ranking;high dimensionality;data mining}, 
     month={Oct}}
"""
import numpy as np

from sklearn.feature_selection import mutual_info_classif, chi2, VarianceThreshold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils import resample

from scipy.stats import rankdata, gmean

from statsmodels.stats.multitest import multipletests

from xgboost import XGBClassifier

import shap as sh

from joblib import Parallel, delayed

# Estimates the impact of features and their interactions
def nl_features(X, y, n_class, use_xgbt):

    if use_xgbt:
        if n_class == 2:
            clf = XGBClassifier(
                n_estimators=128,
                colsample_bynode=0.50,
                eval_metric="logloss",
                use_label_encoder=False,
                nthread=1,
            )

        elif n_class > 2:
            clf = XGBClassifier(
                n_estimators=128,
                colsample_bynode=0.50,
                objective="multi:softprob",
                eval_metric="mlogloss",
                use_label_encoder=False,
                nthread=1,
            )

        clf.fit(X, y)

        explainer = sh.TreeExplainer(clf)

        scores = explainer.shap_interaction_values(X)

        if n_class > 2:
            scores = np.abs(scores).mean(axis=0).mean(axis=0)

        else:
            scores = np.abs(scores).mean(axis=0)

        scores = scores - np.min(scores)
        scores = scores / np.max(scores)

        return scores

    else:
        clf = ExtraTreesClassifier(384)

        clf.fit(X, y)

        explainer = sh.TreeExplainer(clf, feature_perturbation="tree_path_dependent")

        if X.shape[0] > 150:
            X_re = resample(X, replace=False, n_samples=150, stratify=y)

            scores = explainer.shap_interaction_values(X_re)

        else:
            scores = explainer.shap_interaction_values(X)

        if n_class > 2:
            scores = np.abs(scores).mean(axis=0).mean(axis=0)

        else:
            scores = np.abs(scores[0]).mean(axis=0)

        scores = scores - np.min(scores)
        scores = scores / np.max(scores)

        return scores


def adj_cont(X, y, alpha, use_xgbt):

    class_set = np.unique(y)
    n_class = class_set.shape[0]

    # Calculate Fisher IC
    f_i = np.zeros(shape=(X.shape[1],))

    if n_class > 2:
        col_means = np.mean(X, axis=0)
        for col in range(X.shape[1]):
            f_i_tmp = 0.0
            sum_vars - 0.0

            for i, entry in enumerate(class_set):
                loc = np.where(y == entry, True, False)

                mu_diff_class = np.mean(X[loc, col]) - col_means[col]

                sum_vars += np.var(X[loc], axis=0, ddof=1)

                f_i_tmp += np.power(mu_diff_class, 2)

            f_i_tmp = f_i_tmp / sum_vars
            f_i[col] = f_i_tmp

    else:
        loc_1 = np.where(y == class_set[0], True, False)
        loc_2 = np.where(y == class_set[1], True, False)

        for col in range(X.shape[1]):
            mu_diff_class = (np.mean(X[loc_1, col]) - np.mean(X[loc_2, col])) ** 2

            sum_var = np.var(X[loc_1, col], ddof=1) + np.var(X[loc_2, col], ddof=1)

            f_i[col] = mu_diff_class / sum_var

    # All features are important if all elements are the same
    if np.all(f_i == f_i[0]):
        f_i = np.ones(shape=(X.shape[1],))

    else:
        f_i = f_i - np.min(f_i)
        f_i = f_i / np.max(f_i)

    # Calculate mutual information
    m_i = mutual_info_classif(X, y)
    m_i = m_i - np.min(m_i)
    m_i = m_i / np.max(m_i)

    C = np.vstack((m_i, f_i)).mean(axis=0).reshape(-1, 1)
    K_mfd = np.matmul(C, C.T)

    # Estimate feature interaction scores
    K_t = nl_features(X, y, n_class, use_xgbt)

    # Construct A
    A = alpha * K_mfd + (1 - alpha) * K_t
    A = A - np.min(A)
    A = A / np.max(A)

    return A


def adj_bin(X, y, alpha, use_xgbt):

    class_set = np.unique(y)
    n_class = class_set.shape[0]

    # Calculate Chi-squared statistic
    f_i = chi2(X, y)[0]
    f_i = f_i - np.min(f_i)
    f_i = f_i / np.max(f_i)

    # Calculate mutual information
    m_i = mutual_info_classif(X, y)
    m_i = m_i - np.min(m_i)
    m_i = m_i / np.max(m_i)

    C = np.vstack((m_i, f_i)).mean(axis=0).reshape(-1, 1)
    K_mfd = np.matmul(C, C.T)

    # Estimate feature interaction scores
    K_t = nl_features(X, y, n_class, use_xgbt)

    # Construct A
    A = alpha * K_mfd + (1 - alpha) * K_t
    A = A - np.min(A)
    A = A / np.max(A)

    return A


def ec(A):

    # Calculate eigenvalues and eigenvectors
    l, V = np.linalg.eig(A)

    # Select the eigenvector corresponding to the largest eigenvalue
    l_max = l.argsort()[::-1]

    scores = np.array(V[:, l_max[0]]).flatten()
    norm = np.sign(scores.sum()) * np.linalg.norm(scores)
    scores = scores / norm

    return scores


# Continuous Features
def mec_fs_cn(X, y, use_xgbt, alpha):

    # Calculate the scores, this is the test-statistic
    A_stat = adj_cont(X, y, alpha, use_xgbt)
    return ec(A_stat)


# Binary Features - Works Awesome?
def mec_fs_pa(X, y, use_xgbt, alpha):

    class_set = np.unique(y)
    n_class = class_set.shape[0]

    # Get the index of the original features
    F_ori = np.asarray([i for i in range(X.shape[1])])

    # Remove all columns where features do not vary
    rem = VarianceThreshold().fit(X)
    X_filt = rem.transform(X)
    F_ori = rem.transform([F_ori])[0]

    if X_filt.shape[1] == 0:
        return np.asarray([True for _ in range(X.shape[1])])

    # Calculate the scores, this is the test-statistic
    A_stat = adj_bin(X_filt, y, alpha, use_xgbt)
    EC_stat = ec(A_stat)

    EC_final = np.zeros(shape=(X.shape[1],))
    for i, idx in enumerate(F_ori):
        EC_final[idx] = EC_stat[i]

    return EC_final


def select_features(X, y, alpha, use_xgbt, is_binary, bootstrap, perm):

    X_re, y_re = resample(
        X, y, replace=bootstrap, n_samples=int(X.shape[0] * 0.8), stratify=y
    )

    if perm == True:
        y_re = np.random.permutation(y_re)

    if is_binary:
        ec_score = mec_fs_pa(X_re, y_re, use_xgbt, alpha)

    else:
        ec_score = mec_fs_cn(X_re, y_re, use_xgbt, alpha)

    return ec_score


from numpy.random import PCG64


def calc_stat(a, b, n_feat, n_init):

    rng = np.random.Generator(PCG64())

    # Get a sample experiment
    X_a = resample(a, replace=False, n_samples=n_init)

    # Get geometric mean on ranks
    X_a = X_a * -1
    rank_a = rankdata(X_a, axis=1)
    g_rank_a = gmean(rank_a, axis=0)

    # Create many permuted experiments
    X_b = np.copy(b, "C")
    g_rank_b = np.zeros(shape=(200, n_feat))
    for i in range(200):
        X_b = rng.permuted(X_b, axis=1, out=X_b)

        g_rank_b[i] = gmean(X_b, axis=0)

    # Get expected rank
    g_rank_b = g_rank_b.mean(axis=0)

    # Compare and return
    return np.less_equal(g_rank_b, g_rank_a)


def stat(a, n_feat, n_init):

    # Get real part
    a_real = a.real

    # Prepare an array of ranks for permutation
    X_b = np.zeros(shape=(n_init, n_feat))
    for i in range(n_init):
        X_b[i] = np.asarray([i + 1 for i in range(n_feat)])

    # Compare to null distribution
    p = Parallel(5)(
        delayed(calc_stat)(a_real, X_b, n_feat, n_init) for i in range(10000)
    )

    p = np.asarray(p).mean(axis=0)

    p_adj = multipletests(p, method="fdr_tsbh")[0]

    # Prepare raw ranks
    R_stat = a_real * -1
    R_stat = rankdata(R_stat, axis=1)
    R_stat = gmean(R_stat, axis=0)

    # Order and rank features by increasing R_stat
    R_order = [(i, R) for i, R in enumerate(R_stat)]
    R_order = np.asarray(sorted(R_order, key=lambda x: x[1]))
    R_order[:, 1] = rankdata(R_order[:, 1])
    R_order = R_order.astype(np.int)

    return p_adj, R_order


# Class which combines the above method into a nice interface
class mECFS:
    """
    Inputs:

    n_init: int, default = 6
        The number of resampling steps.

    alpha: float, default = 0.5
        Specifies how much to weigh each adjacency matrix.

    use_xgbt: bool, default = True
        Specifies if XGB Trees will be used to detect interactions between
        features. If False, Extremely Randomized Trees will be used.

    Returns:

    An mECFS object.
    """

    def __init__(self, n_init=30, alpha=0.50, use_xgbt=True, bootstrap=True, n_jobs=6):

        self.n_init = n_init
        self.alpha = alpha
        self.use_xgbt = use_xgbt
        self.bootstrap = bootstrap
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

        self.classes_, y_int_ = np.unique(y, return_inverse=True)
        self.n_class_ = self.classes_.shape[0]

        # Check if the features are binary
        self.is_binary_ = np.array_equal(X, X.astype(bool))

        # Get the base distribution of EC scores
        EC_stat = Parallel(n_jobs=self.n_jobs)(
            delayed(select_features)(
                X,
                y_int_,
                self.alpha,
                self.use_xgbt,
                self.is_binary_,
                self.bootstrap,
                perm=False,
            )
            for _ in range(300)
        )

        EC_stat = np.asarray(EC_stat)

        self.selected_, self.ranks_ = stat(EC_stat, X.shape[1], self.n_init)

        if self.selected_.sum() > 0:
            return self

        else:
            print("No features were found significant. Returning all features.")

            self.selected_ = np.asarray([True for i in range(X.shape[1])])

            return self

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
