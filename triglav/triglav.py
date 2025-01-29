from __future__ import annotations

from collections import defaultdict
from re import L
from typing import Union, Tuple, Mapping, List, Type, Set

import matplotlib.axes
import numpy as np
import sage as sg
import shap as sh
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from scipy.cluster import hierarchy
from scipy.stats import combine_pvalues, wilcoxon, betabinom, mannwhitneyu, binomtest, multinomial
from scipy.spatial.distance import squareform
from sklearn.base import TransformerMixin, BaseEstimator, clone, ClassifierMixin
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.ensemble._forest import BaseForest
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.utils import check_X_y, resample
from sklearn.utils.validation import check_is_fitted
from statsmodels.stats.multitest import multipletests
from imblearn.under_sampling.base import BaseCleaningSampler, BaseUnderSampler
from imblearn.over_sampling.base import BaseOverSampler

import pandas as pd


##################################################################################
# Utility Classes - Transform and Scaling
##################################################################################
class NoScale(TransformerMixin, BaseEstimator):
    """
    This function returns the input unchanged.
    """

    def __init__(self):
        pass

    def fit_transform(self, X, y=None, **fit_params):
        return X


class Scaler(TransformerMixin, BaseEstimator):
    """
    Scales each row so that the sum of each row is equal to one.

    X: Numpy array of shape (m, n) where m is the number of samples
       and n the number of features.

    Returns: A Numpy array of shape (p, n), where p <= m. This array
             contains all samples with non-zero entries in each column.
    """

    def __init__(self):
        pass

    def fit_transform(self, X, y=None, **fit_params):
        self.zero_samps = np.where(np.sum(X, axis=1) == 0, False, True)

        row_sums = np.sum(X, axis = 1)[self.zero_samps]

        return X[self.zero_samps]/np.sum(X[self.zero_samps], axis = 1)[:,None]


##################################################################################
# Utility Classes - Calculation of Dissimilarities for Clustering
##################################################################################
class ETCProx:
    def __init__(self, n_estimators=1024, min_samples_split=0.33, n_sets=5):
        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split
        self.n_sets = n_sets

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using the ETCProx method.

        Parameters
        ----------
        X : np.ndarray
            Data matrix of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Proximity matrix of shape (n_samples, n_samples).
        """
        # Randomize class labels (https://inria.hal.science/hal-01667317/file/unsupervised-extremely-randomized_Dalleau_Couceiro_Smail-Tabbone.pdf)
        y_rnd = [0 for _ in range(X.shape[0] // 2)]
        y_rnd.extend([1 for _ in range(X.shape[0] // 2, X.shape[0])])
        y_rnd = np.asarray(y_rnd)

        y_f = np.hstack(
            [
                np.random.choice(y_rnd, size=(X.shape[0],), replace=False)
                for _ in range(self.n_sets)
            ]
        )

        X_stacked = np.vstack([X for _ in range(self.n_sets)])

        clf = ExtraTreesClassifier(
            self.n_estimators,
            min_samples_split=int(X_stacked.shape[0] * self.min_samples_split),
            max_features=1,
        ).fit(X_stacked, y_f)

        L = clf.apply(X)
        L = OneHotEncoder(sparse_output=False).fit_transform(L)
        S = np.dot(L, L.T)
        S = S / 1024
        S = 1 - S
        return np.sqrt(S)


##################################################################################
# Utility Classes - Resampling
##################################################################################
class NoResample(TransformerMixin, BaseEstimator):
    """
    No resampling transformer.
    """

    def __init__(self):
        pass

    def fit_transform(self, X, y=None, **fit_params):

        return X


##################################################################################
# Functions used by Triglav
##################################################################################
def trinomial_test(W):

    if np.all(np.equal(W, W[0])):
        
        return 0.99

    else:

        # Calculate random variables and probabilities
        N_pos = np.where(W > 0, 1, 0).sum()
        N_neg = np.where(W < 0, 1, 0).sum()
        N_tie = np.where(W == 0, 1, 0).sum()

        N = N_pos + N_neg + N_tie

        P_tie = N_tie / N
    
        # These are equal since we are testing that n+ = n-
        P_pos = (1 - P_tie) / 2
        P_neg = P_pos
    
        probs = [P_pos, P_neg, P_tie]

        # Calculate test statistic, nd (abs. value because testing n+ = n-)
        nd = np.abs(N_pos - N_neg)
        nd_sign = np.sign(N_pos - N_neg)

        if nd_sign > 0: # We only care about the positive case for a hit

            ns = [0]*3
            p_value = 0
            for i in range(nd, N+1):
                for j in range(0, int((N - i)/2)+1):
                    ns[0] = j
                    ns[1] = j + i
                    ns[2] = N - j - (j + i)
                    p_value += multinomial.pmf(ns, N, probs)

            return p_value * 2

        else:
            return 0.99


def beta_binom_test(
    X: np.ndarray,
    C: int = 1,
    alpha: float = 0.05,
    p: float = 0.5,
    p2: float = 0.5,
) -> Tuple[List[bool], List[bool]]:
    """
    Beta-binomial test for features. Successes and failures are modelled
    by separate beta-binomial distributions.

    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (n_samples, n_features).
    C : int, optional
        Number of iterations, default = 1
    alpha : float, optional
        Significance level, by default 0.05
    p : float, optional
        Prior probability of a hit, by default 0.5
    p2 : float, optional
        Prior probability of a rejection, by default 0.5

    Returns
    -------
    P_hit : List[bool]
    P_rej : List[bool]
    """

    if C == 0:
        C = 1

    elif C > 0:
        C = C + 1

    THRESHOLD = alpha / C  # For FWER correction

    n = X.shape[0]  # Number of trials

    # Assume hits are rare
    a_0_h = p * n
    b_0_h = n - a_0_h

    # Assume the probability of a rejection is common
    a_0_r = p2 * n
    b_0_r = n - a_0_r

    P_hit = []
    P_rej = []
    for column in range(X.shape[1]):
        pval_hit = betabinom.sf(X[:, column].sum() - 1, n, a_0_h, b_0_h, loc=0)
        P_hit.append(pval_hit)

        pval_rej = betabinom.cdf(X[:, column].sum(), n, a_0_r, b_0_r, loc=0)
        P_rej.append(pval_rej)

    P_hit = np.asarray(P_hit)
    P_rej = np.asarray(P_rej)

    # Correct for comparing multiple features
    P_hit_fdr = multipletests(P_hit, alpha, method="fdr_by")[0]
    P_rej_fdr = multipletests(P_rej, alpha, method="fdr_by")[0]

    # Correct for comparisons across iterations
    P_hit_b = P_hit <= THRESHOLD
    P_rej_b = P_rej <= THRESHOLD

    # Combine
    P_hit = P_hit_fdr * P_hit_b
    P_rej = P_rej_fdr * P_rej_b

    return P_hit, P_rej


def scale_features(
    X: np.ndarray, transformer: Type[TransformerMixin, BaseEstimator]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Function for scaling features. The transformer must be a Scikit-Learn
    compatible transformer.

    Parameters
    ----------
    X : np.ndarray
        The features to be scaled.
    transformer : Type[TransformerMixin, BaseEstimator]
        The transformer to be used for scaling.

    Returns
    -------
    X_transformed : np.ndarray
        The scaled features.
    zero_samps : np.ndarray
        The samples that were zeroed out during scaling.
    """

    if type(transformer) == NoScale or type(transformer) not in [
        Scaler,
    ]:
        X_transformed = transformer.fit_transform(
            X,
        )
    else:
        X_transformed = transformer.fit_transform(
            X,
        )

    return X_transformed


def get_shadow(
    X: np.ndarray,
    transformer: Type[TransformerMixin, BaseEstimator],
    paried: bool
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates permuted features and appends these features to
    the original dataframe. Features are then scaled.

    Parameters
    ----------
    X : np.ndarray
        The features to be permuted.
    transformer : Type[TransformerMixin, BaseEstimator]
        The transformer to be used for scaling.
    paired: bool
        If features are generated at random from the marginal
        distribution or paired. 

    Returns
    -------
    X_final : np.ndarray
        The permuted and scaled features.
    """

    if not paried:
        # Create a NumPy array the same size of X
        X_perm = np.zeros(shape=X.shape, dtype=X.dtype).transpose()

        # Loop through each column and sample without replacement to create shadow features
        for col in range(X_perm.shape[0]):
            X_perm[col] = resample(X[:, col], replace=False, n_samples=X_perm.shape[1])

        X_final = np.hstack((X, X_perm.transpose()))

        # Scale
        X_final = scale_features(X_final, transformer)

        return X_final

    if paried:
                
        pass


def shap_scores(M: Type[ClassifierMixin], X: np.ndarray) -> np.ndarray:
    """
    Get Shapley Scores
    """

    tree_supported = {
        ExtraTreesClassifier,
        HistGradientBoostingClassifier,
        RandomForestClassifier,
    }

    if type(M) in tree_supported:
        explainer = sh.Explainer(M)

        s = explainer(X, check_additivity=False).values

    else:
        explainer = sh.Explainer(M, X)

        s = explainer(X).values

    s = np.abs(s)

    # If there are more than two classes, get the median for each class
    if s.ndim > 2:
        s_final = []
        
        for i in range(s.shape[-1]):
            s_final.append(np.median(s[:, :, i], axis = 0))

        s_final = np.asarray(s_final)

    # Else for 2 classes

    return s_final


def get_hits(
    X: np.ndarray,
    y: np.ndarray,
    estimator: Type[BaseForest],
    transformer: Type[TransformerMixin, BaseEstimator],
    sampler: Union[
        Type[TransformerMixin, BaseEstimator],
        Union[BaseCleaningSampler, BaseUnderSampler, BaseOverSampler],
    ],
    paired: bool
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get hits and rejections for a single iteration of the algorithm

    Parameters
    ----------
    X : np.ndarray
        The features to be permuted.
    y : np.ndarray
        The labels.
    estimator : Type[BaseForest]
        The estimator to be used.
    transformer : Type[TransformerMixin, BaseEstimator]
        The transformer to be used for scaling.
    sampler : Union[Type[TransformerMixin, BaseEstimator], Union[BaseCleaningSampler, BaseUnderSampler, BaseOverSampler]]
        A imblearn compatable resampler.
    paired: bool
        If features are generated at random from the marginal
        distribution or paired. 

    Returns
    -------
    S_r : np.ndarray
        The real impact scores.
    S_p : np.ndarray
        The shadow impact scores.
    idxs : np.ndarray
        Resampled indicies.
    """
    hp_opts = {GridSearchCV, RandomizedSearchCV}

    if X.ndim > 1:
        X_tmp = np.copy(X, "C")
        y_re = y

    else:
        X_tmp = X.reshape(-1, 1)
        y_re = y

    if type(sampler) != NoResample:
        X_tmp, y_re = sampler.fit_resample(X_tmp, y_re)
        idxs = set(sampler.sample_indices_)

        idxs = np.asarray([True if i in idxs else False for i in range(X.shape[0])])

    else:
        idxs = np.asarray([True for i in range(X.shape[0])])

    if not paired:
        X_resamp = get_shadow(X_tmp, transformer, paired)

        n_features = X.shape[1]

        clf = estimator.fit(X_resamp, y_re)

        # Get the best estimator if a grid search was used
        if type(clf) in hp_opts:
            clf = clf.best_estimator_

        S_r = shap_scores(clf, X_resamp)

        S_p = S_r[:, n_features:]
        S_r = S_r[:, 0:n_features]

    return S_r, S_p, idxs


def fs(
    X: np.ndarray,
    y: np.ndarray,
    estimator: Type[BaseForest],
    C_ID: List[int],
    C: Mapping[int, np.ndarray],
    transformer: Type[TransformerMixin, BaseEstimator],
    sampler: Union[BaseCleaningSampler, BaseUnderSampler, BaseOverSampler],
    paired: bool
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Randomly determine the impact of one feature from each cluster

    Parameters
    ----------
    X : np.ndarray
        The features to be permuted.
    y : np.ndarray
        The labels.
    estimator : Type[BaseForest]
        The estimator to be used.
    C_ID : List[int]
        The cluster IDs.
    C : Dict[int, np.ndarray]
        The cluster IDs and their associated features.
    transformer : Type[TransformerMixin, BaseEstimator]
        The transformer to be used for scaling.
    sampler : Union[Type[TransformerMixin, BaseEstimator], Union[BaseCleaningSampler, BaseUnderSampler, BaseOverSampler]]
        A imblearn compatable resampler.
    paired: bool
        If features are generated at random from the marginal
        distribution or paired. 

    Returns
    -------
    S_r : np.ndarray
        The real impact scores.
    S_p : np.ndarray
        The shadow impact scores.
    zero_samps : np.ndarray
        The samples that were zeroed out during scaling.
    """

    # Select Random Feature from Each Cluster
    S = np.asarray([np.random.choice(C[k], size=1)[0] for k in C_ID])

    # Get Shapley impact scores
    S_r, S_p, idxs = get_hits(
        X[:, S], y, estimator, transformer, sampler, paired
    )

    return S_r, S_p, idxs


def global_imps(
    H_real: np.ndarray,
    H_shadow: np.ndarray,
    alpha: float = 0.05,
    percentile: float = 5.0,
    percentile_last: float = 60.0,
    n_jobs = 1,
) -> np.ndarray:
    """
    Used to calculate if real and shadow features differ significantly.

    Parameters
    ----------
    H_real : np.ndarray
        The real impact scores.
    H_shadow : np.ndarray
        The shadow impact scores.
    alpha : float
        The significance level.
    alternative : str
        The alternative hypothesis.

    Returns
    -------
    np.ndarray
        The p-values.
    """

    # Identify predictive features using a modification of the trinomial test
    selected_final = []

    # Find difference between real and shadow features
    for f_i in range(H_real.shape[1]):
        S = H_real[:, f_i, :] - H_shadow[:, f_i, :]

        # Calculate ROPE - Per Class
        S_mean = np.abs(S).mean(0)
        R = np.percentile(S_mean, percentile)

        S = np.where(np.abs(S) > R, S, 0)
        p_vals_raw = Parallel(n_jobs)(
            delayed(trinomial_test)(
                S[:, col]
            )
            for col in range(H_real.shape[2])
        )

        # Identify features where we reject H_0 (Equivalence of positives and negatives)
        S_keep_rej, S_keep = multipletests(np.asarray(p_vals_raw), alpha, method="fdr_bh")[0:2]

        S_sum = S_keep_rej.sum()

        # Stricter thresholding of selected features dependent on those identified using the Trinomial Test
        if S_sum > 0:

            S_final = S.mean(0)
            S_T = np.percentile(S_final[S_keep_rej], percentile_last)

            # Return hits
            selected_final.append(S_final[S_keep_rej] >= S_T)

        else:
            selected_final.append(S_keep_rej)

    # Return all features selected in at least one class
    selected_final = np.asarray(selected_final).sum(axis = 0) > 0

    return selected_final


def stage_1(
    X: np.ndarray,
    y: np.ndarray,
    estimator: Type[ClassifierMixin, BaseEstimator],
    alpha: float,
    n_jobs: int,
    C_ID: np.ndarray,
    C: Mapping[int, List[int]],
    transformer: Type[TransformerMixin, BaseEstimator],
    sampler: Union[
        Type[TransformerMixin, BaseEstimator],
        Union[BaseCleaningSampler, BaseUnderSampler, BaseOverSampler],
    ],
    percentile: float,
    percentile_last: float,
    paired: bool = False
) -> np.ndarray:
    """
    Trains each model and calculates Shapley values in parallel. Determines the
    significance of a feature.

    Parameters
    ----------
    X : np.ndarray
        The data.
    y : np.ndarray
        The labels.
    estimator : Type[ClassifierMixin, BaseEstimator]
        The estimator to use.
    sampler : Union[BaseCleaningSampler, BaseUnderSampler, BaseOverSampler]
    alpha : float
        The alpha level to use for the Wilcoxon test.
    n_jobs : int
        The number of jobs to run in parallel.
    C_ID : np.ndarray
        The cluster IDs.
    C : Mapping[int, List[int]]
        The cluster indices.
    transformer : Type[TransformerMixin, BaseEstimator]
        The transformer to use.
    sampler: Union[Type[TransformerMixin, BaseEstimator], Union[BaseCleaningSampler, BaseUnderSampler, BaseOverSampler]]
        A imblearn compatable resampler.
    paired: bool
        If features are generated at random from the marginal
        distribution or paired. 

    Returns
    -------
    np.ndarray
        The p-values.
    """

    # Calculate how often features are selected by various algorithms
    D = Parallel(n_jobs)(
        delayed(fs)(
            X, y, clone(estimator), C_ID, C, transformer, clone(sampler), paired
        )
        for _ in range(75)
    )

    H_real = [x[0] for x in D]
    H_shadow = [x[1] for x in D]

    return global_imps(
        np.asarray(H_real),
        np.asarray(H_shadow),
        alpha,
        percentile,
        percentile_last,
        n_jobs=n_jobs
    )
    

def stage_2(
    X: np.ndarray,
    y: np.ndarray,
    estimator: Type[ClassifierMixin, BaseEstimator],
    min_sz: int,
    top_k: int,
    transformer: Type[TransformerMixin, BaseEstimator],
    C: int,
    F: np.ndarray
) -> list[int, np.ndarray]:
    """
    Trains each model and calculates Shapley values in parallel. Determines the
    significance of a feature.

    Parameters
    ----------
    X : np.ndarray
        The data.
    y : np.ndarray
        The labels.
    estimator : Type[ClassifierMixin, BaseEstimator]
        The estimator to use.
    top_k : int
        The number top features to select from each cluster.
    transformer : Type[TransformerMixin, BaseEstimator]
        The transformer to use.
    C: int
        The cluster ID
    F: np.ndarray of int
        The location of each feature

    Returns
    -------
    int, np.ndarray
        The cluster ID and indices of the top_k selected features.
    """

    if len(F) > min_sz:

        # Transform X
        X_trf = transformer.fit_transform(X[:, F])
        y_trf = LabelEncoder().fit_transform(y)

        # Calculate SAGE scores and get top k features
        model = clone(estimator=estimator).fit(X_trf, y_trf)

        I = sg.MarginalImputer(model, X_trf)

        E = sg.SignEstimator(I)

        Sv = E(X_trf, y_trf).values

        return C, F[np.argpartition(Sv, -top_k)[-top_k:]]

    else:
        return C, F

 
def update_lists(
    A: Set[int],
    T: Set[int],
    R: Set[int],
    C_INDS: np.ndarray,
    PH: List[bool],
    PR: List[bool],
) -> Tuple[Set[int], Set[int], Set[int], np.ndarray]:
    """
    Update sets of retained, rejected, and tentative features

    Parameters
    ----------
    A : Set[int]
        The set of accepted features.
    T : Set[int]
        The set of tentative features.
    R : Set[int]
        The set of rejected features.
    C_INDS : np.ndarray
        The cluster indices.
    PH : List[bool]
        Mask of the clusters that are accepted.
    PR : List[bool]
        Mask of the clusters that are rejected.

    Returns
    -------
    Tuple[Set[int], Set[int], Set[int], np.ndarray]
        The updated sets of accepted, tentative, and rejected features.
    """

    A_new = set(C_INDS[PH])
    A_new = A.union(A_new)

    R_new = set(C_INDS[PR])
    R_new = R.union(R_new)

    T_new = set(C_INDS) - R_new - A_new

    T_idx = list(T_new)

    return A_new, T_new, R_new, np.asarray(T_idx)


def get_clusters(
    X: np.ndarray,
    linkage_method: str,
    T: float,
    criterion: str,
    transformer: Type[TransformerMixin, BaseEstimator],
    metric: Union[str, ETCProx],
) -> Tuple[List[int], Mapping[int, List[int]], np.ndarray]:
    """
    Creates the flat clusters to be used by the rest of the algorithm.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The training input samples.

    linkage_method : str
        The linkage method to use for hierarchical clustering.

    T : float
        The threshold to use for hierarchical clustering.

    criterion : str
        The criterion to use for hierarchical clustering.

    transformer : TransformerMixin
        The transformer to use for scaling features.

    metric : str
        The metric to use for calculating distances.

    Returns
    -------
    cluster_ids : array-like of shape (n_features,)
        The cluster ids for each feature.

    cluster_id_to_feature_ids : dict
        A dictionary mapping cluster ids to feature ids.

    cluster_id_to_feature_names : dict
        A dictionary mapping cluster ids to feature names.
    """

    # Cluster Features
    X_final = scale_features(X, transformer)

    if type(metric) == ETCProx:
        D = squareform(metric.transform(X_final.T).astype(np.float32))

    else:
        D = squareform(pairwise_distances(X_final.T, metric=metric).astype(np.float32))

    if linkage_method == "complete":
        D = hierarchy.complete(D)

    elif linkage_method == "ward":
        D = hierarchy.ward(D)

    elif linkage_method == "single":
        D = hierarchy.single(D)

    elif linkage_method == "average":
        D = hierarchy.average(D)

    elif linkage_method == "centroid":
        D = hierarchy.centroid(D)

    cluster_ids = hierarchy.fcluster(D, T, criterion=criterion)
    cluster_id_to_feature_ids = defaultdict(list)

    for idx, cluster_id in enumerate(cluster_ids):
        cluster_id_to_feature_ids[cluster_id].append(idx)

    selected_clusters_ = list(cluster_id_to_feature_ids)
    return selected_clusters_, cluster_id_to_feature_ids, D


def select_features(
    transformer: Type[TransformerMixin, BaseEstimator],
    sampler: Union[
        Type[TransformerMixin, BaseEstimator],
        Union[BaseCleaningSampler, BaseUnderSampler, BaseOverSampler],
    ],
    estimator: Type[ClassifierMixin, BaseEstimator],
    stage_2_estimator: Type[ClassifierMixin, BaseEstimator],
    X: np.ndarray,
    max_iter: int,
    n_iter_fwer: int,
    y: np.ndarray,
    alpha: float,
    p: float,
    p2: float,
    percentile: float,
    percentile_last: float,
    metric: Union[str, ETCProx],
    linkage: str,
    thresh: float,
    criterion: str,
    verbose: int,
    n_jobs: int,
    run_stage_2: bool,
    min_sz: int,
    top_k: int,
    var_thresh: float
):
    """
    Function to run each iteration of the feature selection process.
    """

    # Remove constant and quasi-constant features
    nZVF: VarianceThreshold = VarianceThreshold(threshold = var_thresh).fit(X)
    X_red = nZVF.transform(X)

    # Get clusters
    selected_clusters_, cluster_id_to_feature_ids, D = get_clusters(
        X_red, linkage, thresh, criterion, clone(transformer), metric
    )

    # Prepare tracking dictionaries
    F_accepted = set()
    F_rejected = set()
    F_tentative = set()

    shap_df = []

    T_idx = np.copy(selected_clusters_, "C")

    # Stage 1: Calculate Initial Significance - Only Remove Unimportant Features
    if verbose > 0:
        print("Stage One: Identifying an initial set of tentative features...")

    H_arr = []
    IDX = {x: i for i, x in enumerate(T_idx)}
    for n_iter in range(max_iter):
        ITERATION = n_iter + 1

        H_new = stage_1(
            X_red,
            y,
            estimator,
            alpha,
            n_jobs,
            T_idx,
            cluster_id_to_feature_ids,
            clone(transformer),
            sampler,
            percentile,
            percentile_last
        )

        if ITERATION > 1:
            try:
                H_arr = np.vstack((H_arr, [H_new]))
            except:
                print(np.asarray(H_arr).shape, H_new.shape)
                print(np.asarray(H_arr))
                print(H_new)

        else:
            H_arr.append(H_new)

        if ITERATION >= n_iter_fwer:
            P_h, P_r = beta_binom_test(H_arr, ITERATION - n_iter_fwer, alpha, p, p2)
            F_accepted, F_tentative, F_rejected, _ = update_lists(
                F_accepted, F_tentative, F_rejected, T_idx, P_h, P_r
            )
            T_idx = np.asarray(list(F_tentative))
            idx = np.asarray([IDX[x] for x in T_idx])
            if len(F_tentative) == 0:
                break
            H_arr = H_arr[:, idx]
            IDX = {x: i for i, x in enumerate(T_idx)}

        if verbose > 0:
            if ITERATION >= n_iter_fwer:
                tentative = len(F_tentative)
            else:
                tentative = len(cluster_id_to_feature_ids)

            print(
                f"Round {ITERATION:d} "
                f"/ Tentative (Accepted): {len(F_accepted)} "
                f"/ Tentative (Not Accepted): {tentative} "
                f"/ Rejected: {len(F_rejected)}"
            )

    S = []
    rev_cluster_id = {}
    if run_stage_2:
        if verbose > 0:
            print("Stage Two: Identifying best features from each cluster...")
            
        top_ks = Parallel(n_jobs)(
            delayed(stage_2)(
                X_red, 
                y, 
                estimator, 
                min_sz, # this was 5, needs to be a param
                top_k, #this was 2, needs to be a param
                clone(transformer),
                C,
                np.asarray(cluster_id_to_feature_ids[C])
            )
            for C in F_accepted
        )

        for C, F in top_ks:
            for entry in F:
                S.append(int(entry))

                rev_cluster_id[int(entry)] = C
                 
    else:        
        for C in F_accepted:
            for entry in cluster_id_to_feature_ids[C]:
                S.append(entry)

                rev_cluster_id[entry] = C

    S.sort()
    S_1 = np.asarray(S)

    # Return to original size
    S1s = np.zeros(shape=(X_red.shape[1],), dtype=int)
    for entry in S_1:
        S1s[entry] = 1
    S_1 = nZVF.inverse_transform([S1s])[0]
    S_1 = np.where(S_1 > 0, True, False)

    if verbose > 0:
        print(f"Final Feature Set Contains {str(S_1.sum())} Features.")

    return (S_1, None, None, D)


##################################################################################
# Triglav Class
##################################################################################
class Triglav(TransformerMixin, BaseEstimator):
    """
    Triglav is a feature selection algorithm that uses a hierarchical
    clustering algorithm to group features into clusters. The
    importance of each cluster is then calculated using a Shapley
    value approach. The most important features from each cluster are
    then selected using a SAGE approach.

    Attributes
    ----------
    transformer: default = NoScale()
        The transformer to be used to scale features.
    sampler: default = NoResample()
        The type of sampler (from Imbalanced-learn) to use.
    estimator: default = ExtraTreesClassifier(512, bootstrap = True)
        The estimator used to calculate Shapley scores.
    stage_2_estimator: default = ExtraTreesClassifier(512)
        The estimator used to calculate SAGE values. Only used if the
        'run_stage_2' is set to True.
    n_iter: int, default = 40
        The number of iterations to run Triglav.
    n_iter_fwer: int, default = 11
        The iteration at which Bonferroni corrections begin.
    p_1: float, default = 0.65
        Used to determine the shape of the Beta-Binomial distribution
        modelling hits.
    p_2: float, default = 0.30
        Used to determine the shape of the Beta-Binomial distribution
        modelling misses.
    percentile: float. default = 5.0
        The percentile value under which the difference in Shapley values
        between real and shadow clusters is equivalent. Higher values
        will lower the false-discovery rate.
    metric: str, default = "correlation"
        The dissimilarity measure used to calculate distances between
        features.
    linkage: str, default = "complete"
        The type of hierarchical clustering method to apply. The available
        methods include: single, complete, ward, average, centroid.
    thresh: float, default = 2.0
        The threshold or max number of clusters.
    criterion: str, default = "distance"
        The method used to form flat clusters. The available methods
        include: inconsistent, distance, maxclust, monocrit,
        maxclust_monocrit.
    alpha: float, default = 0.05
        The level at which corrected p-values will be rejected.
    run_stage_2: bool, default = True
        This stage will determine the best feature from each of the
        selected clusters by calculating SAGE values.
    verbose: int, default = 0
        Specifies if basic reporting is sent to the user.
    n_jobs: int, default = 10
        The number of threads
    """

    def __init__(
        self,
        transformer=NoScale(),
        sampler=NoResample(),
        estimator=ExtraTreesClassifier(512, bootstrap=True),
        stage_2_estimator=ExtraTreesClassifier(512),
        n_iter: int = 40,
        n_iter_fwer: int = 11,
        p_1: float = 0.65,
        p_2: float = 0.30,
        percentile: float = 5.0,
        percentile_last: float = 60.0,
        metric: Union[str, ETCProx] = "correlation",
        linkage: str = "complete",
        thresh: Union[int, float] = 2.0,
        criterion: str = "distance",
        alpha: float = 0.05,
        run_stage_2: bool = True,
        min_sz: int = 5,
        top_k: int = 2,
        var_thresh: float = 0.025,
        verbose: int = 0,
        n_jobs: int = 10,
    ):

        self.transformer = transformer
        self.sampler = sampler
        self.estimator = estimator
        self.stage_2_estimator = stage_2_estimator
        self.n_iter = n_iter
        self.n_iter_fwer = n_iter_fwer
        self.p_1 = p_1
        self.p_2 = p_2
        self.percentile = percentile
        self.percentile_last = percentile_last
        self.metric = metric
        self.linkage = linkage
        self.thresh = thresh
        self.criterion = criterion
        self.alpha = alpha
        self.run_stage_2 = run_stage_2
        self.min_sz = min_sz
        self.top_k = top_k
        self.var_thresh = var_thresh
        self.verbose = verbose
        self.n_jobs = n_jobs

    def fit(self, X: np.ndarray, y: np.ndarray) -> Triglav:
        """
        Inputs:

        X: NumPy array of shape (m, n) where 'm' is the number of samples and 'n'
        the number of features (taxa, OTUs, ASVs, etc).

        y: NumPy array of shape (m,) where 'm' is the number of samples. Each entry
        of 'y' should be a factor.

        Returns:

        A fitted Triglav object.
        """

        X_in, y_in = self._check_params(X, y)
        
        self.classes_ = np.unique(y)

        self.n_class_ = self.classes_.shape[0]

        # Find relevant features
        (
            self.selected_,
            self.selected_best_,
            self.sage_values_,
            self.linkage_matrix_,
        ) = select_features(
            transformer=self.transformer,
            sampler=self.sampler,
            estimator=self.estimator,
            stage_2_estimator=self.stage_2_estimator,
            max_iter=self.n_iter,
            n_iter_fwer=self.n_iter_fwer,
            X=X_in,
            y=y_in,
            alpha=self.alpha,
            p=self.p_1,
            p2=self.p_2,
            percentile=self.percentile,
            percentile_last=self.percentile_last,
            metric=self.metric,
            linkage=self.linkage,
            thresh=self.thresh,
            criterion=self.criterion,
            verbose=self.verbose,
            run_stage_2=self.run_stage_2,
            min_sz=self.min_sz,
            top_k=self.top_k,
            var_thresh=self.var_thresh,
            n_jobs=self.n_jobs,
        )

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        X : np.ndarray
            NumPy array of shape (m, n) where 'm' is the number of samples and 'n'
            the number of features (taxa, OTUs, ASVs, etc).

        Returns
        -------
        np.ndarray
            NumPy array of shape (m, p) where 'm' is the number of samples and 'p'
            the number of features (taxa, OTUs, ASVs, etc). 'p' <= m
        """
        check_is_fitted(self, attributes="selected_")

        return X[:, self.selected_]

    def fit_transform(
        self, X: np.ndarray, y: np.ndarray = None, **fit_params
    ) -> np.ndarray:
        """
        Parameters
        ----------
        X : np.ndarray
            NumPy array of shape (m, n) where 'm' is the number of samples and 'n'
            the number of features (taxa, OTUs, ASVs, etc).
        y : np.ndarray, optional
            NumPy array of shape (m,) where 'm' is the number of samples. Each entry
            of 'y' should be a factor.
        fit_params : dict, optional

        Returns
        -------
        np.ndarray
            NumPy array of shape (m, p) where 'm' is the number of samples and 'p'
            the number of features (taxa, OTUs, ASVs, etc). 'p' <= m
        """
        self.fit(X, y)

        return self.transform(X)

    def visualize_hclust(
        self,
        X: np.ndarray,
        y: np.ndarray,
        ax: matplotlib.axes.Axes = None,
        **dendrogram_kwargs,
    ) -> dict:
        """
        Visualize the hierarchical clustering dendrogram.

        Parameters
        ----------
        X : np.ndarray
            NumPy array of shape (m, n) where 'm' is the number of samples and 'n'
            the number of features (taxa, OTUs, ASVs, etc).
        y : np.ndarray
            NumPy array of shape (m,) where 'm' is the number of samples. Each entry
            of 'y' should be a factor.
        ax : matplotlib.axes.Axes, optional
            The axes on which to plot the dendrogram, by default None. If None, the
            dendrogram will be plotted to a new figure subplot axis.

        Returns
        -------
        dict
            A dictionary of data structures computed to render the dendrogram.
            See https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.dendrogram.html
            for more details.
        """

        X_in, y_in = self._check_params(X, y)

        # Remove zero-variance features
        nZVF = VarianceThreshold().fit(X)
        X_red = nZVF.transform(X)

        # Get clusters
        _, _, D = get_clusters(
            X_red,
            self.linkage,
            self.thresh,
            self.criterion,
            self.transformer,
            self.metric,
        )

        if ax is None:
            fig, ax = plt.subplots(1, 1)

        return hierarchy.dendrogram(D, ax=ax, **dendrogram_kwargs)

    def _check_params(self, X, y):

        crit_set = {
            "inconsistent",
            "distance",
            "maxclust",
            "monocrit",
            "maxclust_monocrit",
        }

        link_set = {"single", "complete", "ward", "average", "centroid"}

        metrics = {
            "cityblock",
            "cosine",
            "euclidean",
            "l1",
            "l2",
            "manhattan",
            "braycurtis",
            "canberra",
            "chebyshev",
            "correlation",
            "dice",
            "hamming",
            "jaccard",
            "mahalanobis",
            "minkowski",
            "rogerstanimoto",
            "russellrao",
            "seuclidean",
            "sokalmichener",
            "sokalsneath",
            "sqeuclidean",
            "yule",
        }

        # Check if X and y are consistent
        X_in, y_in = check_X_y(X, y, estimator="Triglav")

        # Basic check on parameter bounds
        if self.alpha <= 0 or self.alpha > 1:
            raise ValueError("The 'alpha' parameter should be between 0 and 1.")

        if (self.p_1 <= 0 or self.p_1 > 1) or (self.p_2 <= 0 or self.p_2 > 1):
            raise ValueError("The 'p' parameter should be between 0 and 1.")

        if self.verbose < 0:
            raise ValueError(
                "The 'verbose' parameter should be greater than or equal to zero."
            )

        if self.n_iter <= 0:
            raise ValueError("The 'max_iter' parameter should be at least one.")

        if self.n_iter_fwer <= 0:
            raise ValueError("The 'n_iter_fwer' parameter should be at least one.")

        if self.n_jobs <= 0:
            raise ValueError(
                "The 'n_jobs' parameter should be greater than or equal to one."
            )

        if self.thresh <= 0:
            raise ValueError("The 'thresh' parameter should be greater than one.")

        if self.metric not in metrics and type(self.metric) != ETCProx:
            raise ValueError(
                "The 'metric' parameter should be one supported by Scikit-Learn or 'ETCProx'."
            )

        if self.criterion not in crit_set:
            raise ValueError(
                "The 'criterion' parameter should be one supported in 'scipy.hierarchy'."
            )

        if self.linkage not in link_set:
            raise ValueError(
                "The 'linkage' parameter should one supported in 'scipy.hierarchy'."
            )

        if type(self.run_stage_2) is not bool:
            raise ValueError("The 'run_stage_2' parameter should be True or False.")

        return X_in, y_in
