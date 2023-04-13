from __future__ import annotations

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning, module = "sklearn")

import numpy as np
from numpy.random import PCG64

from sklearn.feature_selection import VarianceThreshold
from sklearn.utils import check_X_y, resample
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import LabelEncoder

from statsmodels.stats.multitest import multipletests

import shap as sh

import sage as sg

from joblib import Parallel, delayed

from scipy.stats import wilcoxon, betabinom
from scipy.cluster import hierarchy

from skbio.stats.composition import multiplicative_replacement, closure, clr

from typing import Union

from collections import defaultdict

from gc import collect

##################################################################################
#Utility Functions
##################################################################################
def beta_binom_test(X, C, alpha, p):

    THRESHOLD = alpha / C #For FWER correction

    n = X.shape[0] #Number of trials

    #Estimate sigma for a given prior, p
    a_0 = p * n
    b_0 = n - a_0

    P_hit = []
    P_rej = []
    for column in range(X.shape[1]):

        pval = betabinom.sf(X[:, column].sum() - 1, n, a_0, b_0, loc=0)

        P_hit.append(pval)

        P_rej.append(1.0 - pval)

    P_hit = np.asarray(P_hit)
    P_rej = np.asarray(P_rej)

    #Correct for comparing multiple features
    P_hit_fdr = multipletests(P_hit, alpha, method = "fdr_bh")[0]
    P_rej_fdr = multipletests(P_rej, alpha, method = "fdr_bh")[0]

    #Correct for comparisons across iterations
    P_hit_b = P_hit <= THRESHOLD
    P_rej_b = P_rej <= THRESHOLD
        
    #Combine
    P_hit = P_hit_fdr * P_hit_b
    P_rej = P_rej_fdr * P_rej_b

    return P_hit, P_rej

def scale_features(X, scale, clr_transform):

    zero_samps = np.where(np.sum(X, axis = 1) == 0, False, True)

    if scale == True and clr_transform == True:

        X_final = clr(multiplicative_replacement(closure(X[zero_samps])))

        return X_final

    elif scale == True and clr_transform == False:
        X_final = closure(X[zero_samps])

        return X_final

    return X[zero_samps]

def get_shadow(X, scale, clr_transform):
    """
    Creates permuted fatures and appends these features to
    the original dataframe. Features are then scaled.
    """

    #Create a NumPy array the same size of X
    X_perm = np.zeros(shape = X.shape, dtype = X.dtype).transpose()

    #Loop through each column and sample without replacement to create shadow features
    for col in range(X_perm.shape[0]):
        X_perm[col] = resample(X[:, col], replace = False, n_samples = X_perm.shape[1])

    X_final = np.hstack((X, X_perm.transpose()))

    #Scale
    X_final = scale_features(X_final, scale, clr_transform)

    return X_final

def shap_scores(M, X):
    """
    Get Shapley Scores
    """

    explainer = sh.Explainer(M)

    s = np.abs(explainer(X, check_additivity=False).values)

    if s.ndim > 2:
        s = s.mean(axis = 2)
        
    s = s.mean(axis = 0)

    return s

def get_hits(X, y, scale, clr_transform):
    """
    Return two NumPy arrays: One of real impact scores and the second of shadow impact scores
    """

    if X.ndim > 1:
        X_tmp = np.copy(X, "C")

    else:
        X_tmp = X.reshape(-1, 1)

    X_resamp = get_shadow(X_tmp, scale, clr_transform)
    
    n_features = X.shape[1]

    clf = ExtraTreesClassifier(512, bootstrap = True).fit(X_resamp, y)
    S_r = shap_scores(clf, X_resamp)
    
    S_p = S_r[n_features:]
    S_r = S_r[0:n_features]

    return S_r, S_p

def fs(X, y, C_ID, C, scale, clr_transform):
    """
    Randomly determine the impact of one feature from each cluster
    """

    class_set = np.unique(y)
    n_class = class_set.shape[0]

    #Select Random Feature from Each Cluster
    S = np.asarray([np.random.choice(C[k], size = 1)[0] for k in C_ID])

    #Get Shapley impact scores
    S_r, S_p = get_hits(X[:, S], y, scale, clr_transform)

    return S_r, S_p

def stage_1(X, y, alpha, p, n_jobs, C_ID, C, scale, clr_transform):
    
    #Calculate how often features are selected by various algorithms
    D = Parallel(n_jobs)(
            delayed(fs)(
                    X,
                    y,
                    C_ID, 
                    C, 
                    scale, 
                    clr_transform)
                for _ in range(75)
                )

    H_real = np.asarray([x[0] for x in D])
    H_shadow = np.asarray([x[1] for x in D])

    #Calculate p-values associated with each feature using the Wilcoxon Test
    p_vals_raw = []
    for column in range(H_real.shape[1]):
        T_stat, p_val = wilcoxon(H_real[:, column], H_shadow[:, column], alternative = "greater")
        p_vals_raw.append(p_val)

    #Correct for multiple comparisons
    H_fdr = multipletests(p_vals_raw, alpha, method = "fdr_bh")[0]

    return H_fdr
   
def update_lists(A, T, R, C_INDS, PH, PR):
    """
    Update sets of retained, rejected, and tentative features
    """

    A_new = set(C_INDS[PH])
    A_new = A.union(A_new)
    
    R_new = set(C_INDS[PR])
    R_new = R.union(R_new)
    
    T_new = set(C_INDS) - R_new - A_new

    T_idx = list(T_new)

    return A_new, T_new, R_new, np.asarray(T_idx)

def get_clusters(X, linkage_method, T, criterion, scale, clr_transform, metric):

    #Cluster Features
    X_final = scale_features(X, scale, clr_transform)
    
    D = pairwise_distances(X_final.T, metric = metric)

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

    selected_clusters_ = []
    for idx, cluster_id in enumerate(cluster_ids):
        cluster_id_to_feature_ids[cluster_id].append(idx)

    for id in cluster_id_to_feature_ids.keys():
        selected_clusters_.append(id)

    return selected_clusters_, cluster_id_to_feature_ids

def select_features(X, max_iter, y, alpha, p, metric, linkage, thresh, criterion, verbose, n_jobs, scale, clr_transform):

    #Remove zero-variance features
    nZVF = VarianceThreshold().fit(X)
    X_red = nZVF.transform(X)

    #Get clusters
    selected_clusters_, cluster_id_to_feature_ids = get_clusters(X_red, linkage, thresh, criterion, scale, clr_transform, metric)

    #Prepare tracking dictionaries
    F_accepted = set()
    F_rejected = set()
    F_tentative = set()

    T_idx = np.copy(selected_clusters_, "C")

    #Stage 1: Calculate Initial Significance - Only Remove Unimportant Features
    if verbose > 0:
        print("Stage One: Identifying an initial set of tentative features...")

    H_arr = []
    IDX = {x: i for i, x in enumerate(T_idx)}
    for n_iter in range(max_iter):
        ITERATION = n_iter + 1

        H_new = stage_1(X, y, alpha, p, n_jobs, T_idx, cluster_id_to_feature_ids, scale, clr_transform)

        if ITERATION > 1:
            H_arr = np.vstack((H_arr, [H_new]))

        else:
            H_arr.append(H_new)

        if ITERATION >= 5:
            P_h, P_r = beta_binom_test(H_arr, ITERATION - 4, alpha, p)
            
            F_accepted, F_tentative, F_rejected, _ = update_lists(F_accepted,
                                                                  F_tentative,
                                                                  F_rejected,
                                                                  T_idx,
                                                                  P_h,
                                                                  P_r)

            T_idx = np.asarray(list(F_tentative))

            idx = np.asarray([IDX[x] for x in T_idx])

            if len(F_tentative) == 0:
                break

            else:
                H_arr = H_arr[:, idx]

                IDX = {x: i for i, x in enumerate(T_idx)}

        if verbose > 0:
            print("Round %d" %(ITERATION),
                  "/ Tentative (Accepted):", len(F_accepted),
                  "/ Tentative (Not Accepted):", len(F_tentative), 
                  "/ Rejected:", len(F_rejected),
                  sep = " ")

    S = []
    rev_cluster_id = {}
    for C in F_accepted:
        for entry in cluster_id_to_feature_ids[C]:
            S.append(entry)

            rev_cluster_id[entry] = C

    S.sort()
    S_1 = np.asarray(S)

    #Stage 2: Determine the best feature from each cluster using Sage
    if verbose > 0:
        print("Stage Two: Identifying best features from each cluster...")

    y_enc = LabelEncoder().fit_transform(y)

    X_red = scale_features(X_red, scale, clr_transform)

    model = ExtraTreesClassifier(3072, n_jobs = 5).fit(X_red[:, S_1], y_enc)

    I = sg.MarginalImputer(model, X[:, S])
    E = sg.SignEstimator(I)
    sage = E(X[:, S], y_enc)
    S_vals = sage.values

    best_in_clus = {}
    for ix, f_val in enumerate(S_vals):
        F_id = S[ix]
        C = rev_cluster_id[F_id]

        if C not in best_in_clus:
            best_in_clus[C] = (F_id, f_val)

        else:
            if f_val > best_in_clus[C][1]:
                best_in_clus[C] = (F_id, f_val)

    S_2 = [v[0] for _, v in best_in_clus.items()]
    S_2.sort()
    S_2 = np.asarray(S_2)
        
    #Return to original size
    S1s = np.zeros(shape = (X_red.shape[1],), dtype = int)
    S2s = np.zeros(shape = (X_red.shape[1],), dtype = int)

    for entry in S_1:
        S1s[entry] = 1
    S_1 = nZVF.inverse_transform([S1s])[0]
    S_1 = np.where(S_1 > 0, True, False)

    for entry in S_2:
        S2s[entry] = 1
    S_2 = nZVF.inverse_transform([S2s])[0]
    S_2 = np.where(S_2 > 0, True, False)

    if verbose > 0:
        print("Final Feature Set Contains %s Features." %str(S_1.sum()))
        print("Final Set of Best Features Contains %s Features." %str(S_2.sum()))

    return S_1, S_2, sage
    
##################################################################################
#Triglav Class
##################################################################################
class Triglav(TransformerMixin, BaseEstimator):
    """
    Inputs:

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

    p: float, default = 0.65
        The 'p' parameter used to determine the shape of the Beta-Binomial 
        distribution.

    alpha: float, default = 0.05
        The level at which corrected p-values will be rejected.

    scale: bool, default = True
        Scales the data so the sum of each row is equal to one.

    clr_transform: bool, default = True
        Applies the centered log ratio to the dataset.

    verbose: int, default = 0
        Specifies if basic reporting is sent to the user.

    n_jobs: int, default = 3
        The number of threads

    Returns:

    An Triglav object.
    """

    def __init__(self,
                 n_iter: int = 40,
                 p: float = 0.65,
                 metric: str = "correlation",
                 linkage: str = "complete",
                 thresh: Union[int, float] = 2.0,
                 criterion: str = "distance",
                 alpha: float = 0.05,
                 scale: bool = True,
                 clr_transform: bool = False,
                 verbose: int = 0, 
                 n_jobs: int = 10):

        self.n_iter = n_iter
        self.p = p
        self.metric = metric
        self.linkage = linkage
        self.thresh = thresh
        self.criterion = criterion
        self.alpha = alpha
        self.scale = scale
        self.clr_transform = clr_transform
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
        self.selected_, self.selected_best_, self.sage_values_ = select_features(
                max_iter = self.n_iter,
                X = X_in,
                y = y_int_,
                alpha = self.alpha,
                p = self.p,
                metric = self.metric,
                linkage = self.linkage,
                thresh = self.thresh,
                criterion = self.criterion,
                verbose = self.verbose,
                scale = self.scale,
                clr_transform = self.clr_transform,
                n_jobs = self.n_jobs)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Input:

        X: NumPy array of shape (m, n) where 'm' is the number of samples and 'n'
        the number of features (taxa, OTUs, ASVs, etc).

        Returns:

        X_transformed: NumPy array of shape (m, p) where 'm' is the number of samples and 'p'
        the number of features (taxa, OTUs, ASVs, etc). 'p' <= m
        """
        check_is_fitted(self, attributes = "selected_")

        return X[:, self.selected_]

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:

        self.fit(X, y)

        return self.transform(X)

    def _check_params(self, X, y):

        crit_set = {"inconsistent", 
                    "distance", 
                    "maxclust", 
                    "monocrit",
                    "maxclust_monocrit"}

        link_set = {"single", 
                    "complete", 
                    "ward", 
                    "average", 
                    "centroid"}

        metrics = {'cityblock', 
                   'cosine', 
                   'euclidean', 
                   'l1', 
                   'l2',
                   'manhattan', 
                   'braycurtis', 
                   'canberra', 
                   'chebyshev',
                   'correlation', 
                   'dice', 
                   'hamming', 
                   'jaccard',
                   'mahalanobis', 
                   'minkowski', 
                   'rogerstanimoto', 
                   'russellrao',
                   'seuclidean', 
                   'sokalmichener', 
                   'sokalsneath', 
                   'sqeuclidean',
                   'yule', }

        #Check if X and y are consistent
        X_in, y_in = check_X_y(X, y, estimator = "Triglav")
        
        #Basic check on parameter bounds
        if self.alpha <= 0 or self.alpha > 1:
            raise ValueError("The 'alpha' parameter should be between 0 and 1.")

        if self.p <= 0 or self.p > 1:
            raise ValueError("The 'p' parameter should be between 0 and 1.")

        if self.verbose < 0:
            raise ValueError("The 'verbose' parameter should be greater than or equal to zero.")

        if self.n_jobs <= 0:
            raise ValueError("The 'n_jobs' parameter should be greater than or equal to one.")

        if self.thresh <= 0:
            raise ValueError("The 'thresh' parameter should be greater than one.")

        if self.metric not in metrics:
            raise ValueError("The 'metric' parameter should be one supported by Scikit-Learn.")

        if self.criterion not in crit_set:
            raise ValueError("The 'criterion' parameter should be one supported in 'scipy.hierarchy'.")

        if self.linkage not in link_set:
            raise ValueError("The 'linkage' parameter should one supported in 'scipy.hierarchy'.")

        if type(self.scale) is not bool or type(self.clr_transform) is not bool:
            raise ValueError("The 'scale' and/or 'clr_transform' parameter(s) should be True or False.")

        return X_in, y_in

