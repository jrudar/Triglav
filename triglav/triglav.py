from __future__ import annotations

from collections import defaultdict
from typing import Union, Tuple, Mapping, List, Type, Set

import matplotlib.axes
import numpy as np
import shap as sh
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from scipy.cluster import hierarchy
from scipy.stats import wilcoxon, betabinom
from scipy.spatial.distance import squareform
from sklearn.base import TransformerMixin, BaseEstimator, clone, ClassifierMixin
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.model_selection import cross_val_score
from sklearn.ensemble._forest import BaseForest
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import pairwise_distances, balanced_accuracy_score, f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import check_X_y, resample
from sklearn.utils.validation import check_is_fitted
from statsmodels.stats.multitest import multipletests
from imblearn.under_sampling.base import BaseCleaningSampler, BaseUnderSampler
from imblearn.over_sampling.base import BaseOverSampler

from niapy.problems import Problem
from niapy.task import Task
from niapy.algorithms.basic import ArtificialBeeColonyAlgorithm


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
        L = OneHotEncoder(sparse=False).fit_transform(L)
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
# Utility Classes - PSO
##################################################################################
def comb_score(y_true, y_pred, **kwargs):

    score_ba = balanced_accuracy_score(y_true, y_pred)
    score_f1 = f1_score(y_true, y_pred, average = "weighted")

    score = (score_ba + score_f1) * 0.5

    return score

class ModelPSO(Problem):
    """
    Performs PSO to select features
    """

    def __init__(self, X_train, y_train, model, alpha = 0.99):
        
        super().__init__(dimension=X_train.shape[1], lower=0, upper=1)

        self.X_train = X_train
        self.y_train = y_train
        self.alpha = alpha
        self.model = model

    def _evaluate(self, x):

        selected = x > 0.5
        num_selected = selected.sum()
        if num_selected == 0:
            return 1.0

        accuracy = cross_val_score(self.model,
                                   self.X_train[:, selected],
                                   self.y_train,
                                   cv = 2).mean()

        score = 1 - accuracy

        num_features = self.X_train.shape[1]

        return self.alpha * score + (1 - self.alpha) * (num_selected / num_features)



##################################################################################
# Functions used by Triglav
##################################################################################
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
        Number of classes, by default 1
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
    P_hit_fdr = multipletests(P_hit, alpha, method="fdr_bh")[0]
    P_rej_fdr = multipletests(P_rej, alpha, method="fdr_bh")[0]

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
        zero_samps = np.ones(shape=(X.shape[0],), dtype=bool)
        X_transformed = transformer.fit_transform(
            X,
        )
    else:
        X_transformed = transformer.fit_transform(
            X,
        )
        zero_samps = transformer.zero_samps

    return X_transformed, zero_samps


def get_shadow(
    X: np.ndarray,
    transformer: Type[TransformerMixin, BaseEstimator],
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

    Returns
    -------
    X_final : np.ndarray
        The permuted and scaled features.
    zero_samps : np.ndarray
        The samples that were zeroed out during scaling.
    """

    # Create a NumPy array the same size of X
    X_perm = np.zeros(shape=X.shape, dtype=X.dtype).transpose()

    # Loop through each column and sample without replacement to create shadow features
    for col in range(X_perm.shape[0]):
        X_perm[col] = resample(X[:, col], replace=False, n_samples=X_perm.shape[1])

    X_final = np.hstack((X, X_perm.transpose()))

    # Scale
    X_final, zero_samps = scale_features(X_final, transformer)

    return X_final, zero_samps


def shap_scores(M: Type[ClassifierMixin], X: np.ndarray, per_class: bool) -> np.ndarray:
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

    if not per_class:
        s = np.abs(s)

        if s.ndim > 2:
            s = s.mean(axis=2)

        s = s.mean(axis=0)

    return s


def get_hits(
    X: np.ndarray,
    y: np.ndarray,
    estimator: Type[BaseForest],
    transformer: Type[TransformerMixin, BaseEstimator],
    per_class: bool,
    sampler: Union[
        Type[TransformerMixin, BaseEstimator],
        Union[BaseCleaningSampler, BaseUnderSampler, BaseOverSampler],
    ],
) -> Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray]]:
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
    per_class : bool
        Whether to return per-class scores.
    sampler : Union[Type[TransformerMixin, BaseEstimator], Union[BaseCleaningSampler, BaseUnderSampler, BaseOverSampler]]
        A imblearn compatable resampler.

    Returns
    -------
    S_r : np.ndarray
        The real impact scores.
    S_p : np.ndarray
        The shadow impact scores.
    (idxs, zero_samps) : np.ndarray, np.ndarray
        A tuple of resampled indicies and samples that were zeroed out during scaling.
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

    X_resamp, zero_samps = get_shadow(X_tmp, transformer)

    n_features = X.shape[1]

    clf = estimator.fit(X_resamp, y_re[zero_samps])

    # Get the best estimator if a grid search was used
    if type(clf) in hp_opts:
        clf = clf.best_estimator_

    S_r = shap_scores(clf, X_resamp, per_class)

    if per_class:

        if S_r.ndim == 2:

            S_p = S_r[:, n_features:]
            S_r = S_r[:, 0:n_features]

        else:

            S_p = S_r[:, n_features:, :]
            S_r = S_r[:, 0:n_features, :]

    else:

        S_p = S_r[n_features:]
        S_r = S_r[0:n_features]

    return S_r, S_p, (idxs, zero_samps)


def fs(
    X: np.ndarray,
    y: np.ndarray,
    estimator: Type[BaseForest],
    C_ID: List[int],
    C: Mapping[int, np.ndarray],
    transformer: Type[TransformerMixin, BaseEstimator],
    per_class: bool,
    sampler: Union[BaseCleaningSampler, BaseUnderSampler, BaseOverSampler],
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
    per_class : bool
        Whether to return per-class scores.
    sampler : Union[Type[TransformerMixin, BaseEstimator], Union[BaseCleaningSampler, BaseUnderSampler, BaseOverSampler]]
        A imblearn compatable resampler.

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
    S_r, S_p, zero_samps = get_hits(
        X[:, S], y, estimator, transformer, per_class, sampler
    )

    return S_r, S_p, zero_samps


def global_imps(
    H_real: np.ndarray,
    H_shadow: np.ndarray,
    alpha: float = 0.05,
    alternative: str = "two-sided",
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

    # Calculate p-values associated with each feature using the Wilcoxon Test
    p_vals_raw = []
    for column in range(H_real.shape[1]):

        if np.all(np.equal(H_real[:, column], H_shadow[:, column])):
            p_vals_raw.append(1.0)

        else:
            T_stat, p_val = wilcoxon(
                H_real[:, column], H_shadow[:, column], alternative=alternative
            )
            p_vals_raw.append(p_val)

    # Correct for multiple comparisons
    return multipletests(p_vals_raw, alpha, method="fdr_bh")[0]


def per_class_imps(
    H_real: List[float],
    H_shadow: List[float],
    alpha: float,
    y: np.ndarray,
    Z_loc: List[np.ndarray, np.ndarray],
) -> np.ndarray:
    """
    Determines if Shapley values differ significantly on a
    per-class (one vs rest) level.

    Parameters
    ----------
    H_real : List[float]
        The real impact scores.
    H_shadow : List[float]
        The shadow impact scores.
    alpha : float
        The alpha level to use for the Wilcoxon test.
    y : np.ndarray
        The labels.
    Z_loc : List[np.ndarray]
        The indices of the samples in each cluster.

    Returns
    -------
    np.ndarray
        The p-values for each class.
    """

    H = []

    classes_ = np.unique(y)

    for i, class_name in enumerate(classes_):
        locs = [np.where(y[zs[0]][zs[1]] == class_name, True, False) for zs in Z_loc]

        # For more than two classes or Extra Trees/Random Forest
        if np.asarray([H_real[0]]).ndim > 3:

            # Get the class being examined
            H_real_i = [row[:, :, i] for row in H_real]
            H_shadow_i = [row[:, :, i] for row in H_shadow]

            # Get the rows being examined
            H_real_i = np.asarray(
                [row[locs[j]].mean(axis=0) for j, row in enumerate(H_real_i)]
            )
            H_shadow_i = np.asarray(
                [row[locs[j]].mean(axis=0) for j, row in enumerate(H_shadow_i)]
            )

            # Calculate p-values associated with each feature using the Wilcoxon Test
            H_class = global_imps(H_real_i, H_shadow_i, alpha, alternative="greater")

        # Binary classes
        else:
            H_real_i = np.asarray(
                [row[locs[j]].mean(axis=0) for j, row in enumerate(H_real)]
            )
            H_shadow_i = np.asarray(
                [row[locs[j]].mean(axis=0) for j, row in enumerate(H_shadow)]
            )

            # Calculate p-values associated with each feature using the Wilcoxon Test
            if class_name == 0:
                H_class = global_imps(
                    H_real_i,
                    H_shadow_i,
                    alpha,
                    alternative="less",
                )

            else:
                H_class = global_imps(
                    H_real_i,
                    H_shadow_i,
                    alpha,
                    alternative="greater",
                )

        H.append(H_class)

    H = np.sum(np.asarray(H), axis=0)

    H_fdr = np.where(H > 0, True, False)

    return H_fdr


def stage_1(
    X: np.ndarray,
    y: np.ndarray,
    estimator: Type[ClassifierMixin, BaseEstimator],
    alpha: float,
    n_jobs: int,
    C_ID: np.ndarray,
    C: Mapping[int, List[int]],
    transformer: Type[TransformerMixin, BaseEstimator],
    per_class_imp: bool,
    sampler: Union[
        Type[TransformerMixin, BaseEstimator],
        Union[BaseCleaningSampler, BaseUnderSampler, BaseOverSampler],
    ],
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
    per_class_imp : bool, optional
        Whether to use per-class importance, by default False
    sampler: Union[Type[TransformerMixin, BaseEstimator], Union[BaseCleaningSampler, BaseUnderSampler, BaseOverSampler]]
        A imblearn compatable resampler.

    Returns
    -------
    np.ndarray
        The p-values.
    """

    # Calculate how often features are selected by various algorithms
    D = Parallel(n_jobs)(
        delayed(fs)(
            X, y, clone(estimator), C_ID, C, transformer, per_class_imp, clone(sampler)
        )
        for _ in range(75)
    )

    H_real = [x[0] for x in D]
    H_shadow = [x[1] for x in D]
    Z_loc = [x[2] for x in D]
    return (
        per_class_imps(H_real, H_shadow, alpha, y, Z_loc)
        if per_class_imp
        else global_imps(
            np.asarray(H_real),
            np.asarray(H_shadow),
            alpha,
            alternative="greater",
        )
    )


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
    X_final, _ = scale_features(X, transformer)

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
    per_class_imp: bool,
    X: np.ndarray,
    max_iter: int,
    n_iter_fwer: int,
    y: np.ndarray,
    alpha: float,
    p: float,
    p2: float,
    metric: Union[str, ETCProx],
    linkage: str,
    thresh: float,
    criterion: str,
    verbose: int,
    n_jobs: int,
    run_stage_2: bool,
):
    """
    Function to run each iteration of the feature selection process.
    """

    # Remove zero-variance features
    nZVF: VarianceThreshold = VarianceThreshold().fit(X)
    X_red = nZVF.transform(X)

    # Get clusters
    selected_clusters_, cluster_id_to_feature_ids, D = get_clusters(
        X_red, linkage, thresh, criterion, clone(transformer), metric
    )

    # Prepare tracking dictionaries
    F_accepted = set()
    F_rejected = set()
    F_tentative = set()

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
            per_class_imp,
            sampler,
        )

        if ITERATION > 1:
            H_arr = np.vstack((H_arr, [H_new]))

        else:
            H_arr.append(H_new)

        if ITERATION >= n_iter_fwer:
            P_h, P_r = beta_binom_test(H_arr, ITERATION - 4, alpha, p, p2)
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
            print(
                f"Round {ITERATION:d} "
                f"/ Tentative (Accepted): {len(F_accepted)} "
                f"/ Tentative (Not Accepted): {len(F_tentative)} "
                f"/ Rejected: {len(F_rejected)}"
            )

    S = []
    rev_cluster_id = {}
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

    # Stage 2: Determine the best feature from each cluster using PSO
    if run_stage_2:
        if verbose > 0:
            print("Stage Two: Identifying best features from each cluster...")

        X_red, zero_samps = scale_features(X_red, transformer)

        S_tmp = nZVF.transform([S_1])[0]

        P = ModelPSO(X_red[:, S_tmp], y[zero_samps], stage_2_estimator)
        T = Task(P, max_iters = 100)
        A = ArtificialBeeColonyAlgorithm(population_size = 50)
        best_features, best_fitness = A.run(T)

        S_2 = []
        for ix, f_val in enumerate(best_features):
            if f_val > 0.5:
                S_2.append(S[ix])

        S_2.sort()
        S_2 = np.asarray(S_2)

        # Return to original size
        S2s = np.zeros(shape=(X_red.shape[1],), dtype=int)
        for entry in S_2:
            S2s[entry] = 1
        S_2 = nZVF.inverse_transform([S2s])[0]
        S_2 = np.where(S_2 > 0, True, False)

    if verbose > 0:
        print(f"Final Feature Set Contains {str(S_1.sum())} Features.")

        if run_stage_2:
            print(f"Final Set of Best Features Contains {str(S_2.sum())} Features.")

    return (S_1, S_2, T, D) if run_stage_2 else (S_1, None, None, D)


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
    per_class_imp: bool, default = False
        Specifies if importance scores are calculated globally or per
        class. Note, per class importance scores are calculated in a
        one vs rest manner.
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
        include: distance, maxclust.
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
        per_class_imp: bool = False,
        n_iter: int = 40,
        n_iter_fwer: int = 11,
        p_1: float = 0.65,
        p_2: float = 0.30,
        metric: Union[str, ETCProx] = "correlation",
        linkage: str = "complete",
        thresh: Union[int, float] = 2.0,
        criterion: str = "distance",
        alpha: float = 0.05,
        run_stage_2: bool = True,
        verbose: int = 0,
        n_jobs: int = 10,
    ):

        self.n_class_ = None
        self.classes_ = None
        self.transformer = transformer
        self.sampler = sampler
        self.estimator = estimator
        self.stage_2_estimator = stage_2_estimator
        self.per_class_imp = per_class_imp
        self.n_iter = n_iter
        self.n_iter_fwer = n_iter_fwer
        self.p_1 = p_1
        self.p_2 = p_2
        self.metric = metric
        self.linkage = linkage
        self.thresh = thresh
        self.criterion = criterion
        self.alpha = alpha
        self.run_stage_2 = run_stage_2
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

        self.classes_, y_int_ = np.unique(y_in, return_inverse=True)
        self.n_class_ = self.classes_.shape[0]

        # Find relevant features
        (
            self.selected_,
            self.selected_best_,
            self.task_opt_,
            self.linkage_matrix_,
        ) = select_features(
            transformer=self.transformer,
            sampler=self.sampler,
            estimator=self.estimator,
            stage_2_estimator=self.stage_2_estimator,
            per_class_imp=self.per_class_imp,
            max_iter=self.n_iter,
            n_iter_fwer=self.n_iter_fwer,
            X=X_in,
            y=y_int_,
            alpha=self.alpha,
            p=self.p_1,
            p2=self.p_2,
            metric=self.metric,
            linkage=self.linkage,
            thresh=self.thresh,
            criterion=self.criterion,
            verbose=self.verbose,
            run_stage_2=self.run_stage_2,
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
            "distance",
            "maxclust",
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
                "The 'criterion' parameter should be 'distance' or 'maxclust'."
            )

        if self.linkage not in link_set:
            raise ValueError(
                "The 'linkage' parameter should one supported in 'scipy.hierarchy'."
            )

        if type(self.run_stage_2) is not bool:
            raise ValueError("The 'run_stage_2' parameter should be True or False.")

        if type(self.per_class_imp) is not bool:
            raise ValueError("The 'per_class_imp' parameter should be True or False.")

        return X_in, y_in
