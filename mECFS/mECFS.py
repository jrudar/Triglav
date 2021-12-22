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
from sklearn.metrics import balanced_accuracy_score, pairwise_distances
from sklearn.utils import resample

from xgboost import XGBClassifier

import shap as sh

#Estimates the impact of features and their interactions
def nl_features(X, y, n_class, use_xgbt):

    if use_xgbt:
        if n_class == 2:
            clf = XGBClassifier(n_estimators = 128,
                                colsample_bynode = 0.50,
                                eval_metric = "logloss",
                                use_label_encoder = False,
                                nthread = 1
                                )

        elif n_class > 2:
            clf = XGBClassifier(n_estimators = 128,
                                colsample_bynode = 0.50,
                                objective = "multi:softprob",
                                eval_metric = "mlogloss",
                                use_label_encoder = False,
                                nthread = 1
                                )

        clf.fit(X, y)

        explainer = sh.TreeExplainer(clf)

        scores = explainer.shap_interaction_values(X)

        if n_class > 2:
            scores = np.abs(scores).mean(axis = 0).mean(axis = 0)

        else:
            scores = np.abs(scores).mean(axis = 0)

        scores = scores - np.min(scores)
        scores = scores / np.max(scores)

        return scores

    else:
        clf = ExtraTreesClassifier(384)

        clf.fit(X, y)

        explainer = sh.TreeExplainer(clf, feature_perturbation = "tree_path_dependent")

        if X.shape[0] > 150:
            X_re = resample(X, replace = False, n_samples = 150, stratify = y)

            scores = explainer.shap_interaction_values(X_re)

        else:
            scores = explainer.shap_interaction_values(X)

        if n_class > 2:
            scores = np.abs(scores).mean(axis = 0).mean(axis = 0)

        else:
            scores = np.abs(scores[0]).mean(axis = 0)

        scores = scores - np.min(scores)
        scores = scores / np.max(scores)

        return scores

#Continuous Features
def mec_fs(X, y, k_select = "auto", use_xgbt = True):
 
    alphas = [0.1, 0.2, 0.3, 
              0.4, 0.5, 0.6, 
              0.7, 0.8, 0.9]

    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size = 0.2, 
                                                        stratify = y)

    class_set = np.unique(y)
    n_class = class_set.shape[0]

    #Calculate Fisher IC
    f_i = np.zeros(shape = (X_train.shape[1],))
    col_means = np.mean(X_train, axis = 0)
    col_var = np.var(X_train, axis = 0, ddof = 1)
    for col in range(X_train.shape[1]):
        f_i_tmp = 0.0

        for i, entry in enumerate(class_set):
            loc = np.where(y_train == entry, True, False)
            tmp = np.mean(X_train[loc, col]) - col_means[col]
            f_i_tmp += np.power(tmp, 2)

        f_i_tmp = f_i_tmp / col_var[col]
        f_i[col] = f_i_tmp

    f_i = f_i - np.min(f_i)
    f_i = f_i / np.max(f_i)

    #Calculate mutual information
    m_i = mutual_info_classif(X_train, y_train)
    m_i = m_i - np.min(m_i)
    m_i = m_i / np.max(m_i)

    C = np.vstack((m_i, f_i)).mean(axis = 0).reshape(-1, 1)
    K_mfd = np.matmul(C, C.T)

    #Estimate feature interaction scores
    K_t = nl_features(X_train, 
                      y_train,
                      n_class,
                      use_xgbt)

    #Combone Kernels and Estimate Appropriate Alpha
    cv_scores = []
    results = []
    for alpha in alphas:
        A = alpha * K_mfd + (1 - alpha) * K_t

        A = A - np.min(A)
        A = A / np.max(A)

        #Calculate eigenvalues and eigenvectors
        l, V = np.linalg.eig(A)

        #Select the eigenvector corresponding to the largest eigenvalue
        l_max = l.argsort()[::-1]

        scores = np.array(V[:, l_max[0]]).flatten()
        norm = np.sign(scores.sum()) * np.linalg.norm(scores)
        scores = scores / norm

        if k_select == "auto":
            med = np.median(scores)
            mu = np.mean(scores)

            if med >= mu:
                filtered = np.where(scores >= mu, 
                                    True, 
                                    False)

            else:
                filtered = np.where(scores >= med, 
                                    True, 
                                    False)

        else:
            top_idx = np.argsort(scores)[::-1]

            if top_idx.shape[0] < k_select:
                k_select = -1

            filtered = set(top_idx[0:k_select])
        
            filtered = np.asarray([True if i in filtered 
                                   else False 
                                   for i in range(X.shape[1])])

        clf = ExtraTreesClassifier().fit(X_train[:, filtered], 
                                         y_train)

        cv_scores.append(balanced_accuracy_score(y_test, 
                                                 clf.predict(X_test[:, filtered])))
        results.append(filtered)

    #Pick the index of a random score if more than one best score exists
    max_score = np.max(cv_scores)

    cv_scores = np.where(cv_scores == max_score, True, False)
    
    idx = np.random.choice(np.asarray([i for i in range(cv_scores.shape[0]) 
                                       if cv_scores[i] == True]), 
                           1)[0]

    return results[idx]

#Binary Features - Works Awesome?
def mec_fs_pa(X, y, k_select = "auto", use_xgbt = True):
 
    class_set = np.unique(y)
    n_class = class_set.shape[0]

    alphas = [0.1, 0.2, 0.3, 
              0.4, 0.5, 0.6, 
              0.7, 0.8, 0.9]

    #Get the index of the original features
    F_ori = np.asarray([i for i in range(X.shape[1])])

    #Remove all columns where features do not vary
    rem = VarianceThreshold().fit(X)
    X_filt = rem.transform(X)
    F_ori = rem.transform([F_ori])[0]
    
    if X_filt.shape[1] == 0:
        return np.asarray([True for _ in range(X.shape[1])])

    #Divide data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_filt, y, 
                                                        test_size = 0.2, 
                                                        stratify = y)

    #Calculate Chi-squared statistic
    f_i = chi2(X_train, y_train)[0]
    f_i = f_i - np.min(f_i)
    f_i = f_i / np.max(f_i)

    #Calculate mutual information
    m_i = mutual_info_classif(X_train, y_train)
    m_i = m_i - np.min(m_i)
    m_i = m_i / np.max(m_i)

    C = np.vstack((m_i, f_i)).mean(axis = 0).reshape(-1, 1)
    K_mfd = np.matmul(C, C.T)

    #Estimate feature interaction scores
    K_t = nl_features(X_train, 
                      y_train,
                      n_class,
                      use_xgbt)

    #Combone Kernels and Estimate Appropriate Alpha
    cv_scores = []
    results = []
    for alpha in alphas:
        A = alpha * K_mfd + (1 - alpha) * K_t

        A = A - np.min(A)
        A = A / np.max(A)

        #Calculate eigenvalues and eigenvectors
        l, V = np.linalg.eig(A)

        #Select the eigenvector corresponding to the largest eigenvalue
        l_max = l.argsort()[::-1]

        scores = np.array(V[:, l_max[0]]).flatten()
        norm = np.sign(scores.sum()) * np.linalg.norm(scores)
        scores = scores / norm

        if k_select == "auto":
            med = np.median(scores)
            mu = np.mean(scores)

            if med >= mu:
                retained = np.where(scores >= mu, True, False)

            else:
                retained = np.where(scores >= med, True, False)

            F_ori = set(F_ori[retained])

            filtered = np.asarray([True if i in F_ori else False for i in range(X.shape[1])])

        else:
            top_idx = np.argsort(scores)[::-1]

            if top_idx.shape[0] < k_select:
                k_select = -1

            filtered = set(top_idx[0:k_select])
        
            filtered = np.asarray([True if i in filtered else False for i in range(X.shape[1])])

        clf = ExtraTreesClassifier().fit(X_train[:, filtered], 
                                         y_train)

        cv_scores.append(balanced_accuracy_score(y_test, 
                                                 clf.predict(X_test[:, filtered])))
        results.append(filtered)

    #Pick the index of a random score if more than one best score exists
    max_score = np.max(cv_scores)

    cv_scores = np.where(cv_scores == max_score, True, False)
    
    idx = np.random.choice(np.asarray([i for i in range(cv_scores.shape[0]) 
                                       if cv_scores[i] == True]), 
                           1)[0]

    return results[idx]

#Class which combines the above method into a nice interface
class mECFS():
    """
    Inputs:

    n_init: int, default = 6
        The number of resampling steps.

    k_select: int or str, default = "auto"
        The number of features to be selected.

    bootstrap: bool, default = True
        Specifies if bootstrap resampling will be used.

    n_samples: float, default = 0.8
        A number between 0 and 1.0. This is only used if the 'bootstrap'
        parameter is set to 'False'. Specifies the number of samples to be
        randomly selected.

    use_xgbt: bool, default = True
        Specifies if XGB Trees will be used to detect interactions between
        features. If False, Extremely Randomized Trees will be used. If
        False, the Extra Trees Classifier will be used instead.

    Returns:

    An mECFS object.
    """

    def __init__(self, n_init = 6, k_select = "auto", bootstrap = True, n_samples = 0.8, use_xgbt = True):

        self.n_init = n_init
        self.k_select = k_select
        self.bootstrap = bootstrap
        self.n_samples = n_samples
        self.use_xgbt = use_xgbt

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

        self.classes_, y_int_ = np.unique(y, return_inverse = True)
        self.n_class_ = self.classes_.shape[0]

        if self.bootstrap:
            n_samp = X.shape[0]

        else:
            n_samp = int(X.shape[0] * self.n_samples)

        #Check if the features are binary
        self.is_binary_ = np.array_equal(X, X.astype(bool))

        #Run the algorithm multiple times on multiple instances of the dataset
        stability_arr = []
        for _ in range(self.n_init):
            X_re, y_re = resample(X, y_int_, 
                                  replace = self.bootstrap, 
                                  n_samples = n_samp, 
                                  stratify = y_int_)

            if self.is_binary_:
                stability_arr.append(mec_fs_pa(X_re, 
                                               y_re, 
                                               k_select = self.k_select,
                                               use_xgbt = self.use_xgbt))

            else:
                stability_arr.append(mec_fs(X_re, 
                                            y_re, 
                                            k_select = self.k_select,
                                            use_xgbt = self.use_xgbt))

        #Only select features larger than the mean of all non-zero features
        if len(stability_arr) > 1:
            self.selected_ = np.mean(stability_arr, axis = 0)
            
            non_zero = np.where(self.selected_ > 0, 
                                True, 
                                False)

            mu_selected = np.mean(self.selected_[non_zero])
            
            self.selected_ = np.where(self.selected_ >= mu_selected, 
                                      True, 
                                      False)

        else:
            self.selected_ = stability_arr[0]

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

def test_func(data_set = 0, use_xgb = True, n_init = 1):

    from sklearn.datasets import load_breast_cancer, make_classification
    from sklearn.metrics import classification_report
    from sklearn.preprocessing import StandardScaler

    if data_set == 0:
        #First 10 features are informative, rest are noise
        X, y = make_classification(n_samples = 300, n_features = 300, n_informative = 10, n_classes = 3, shuffle = False)

    elif data_set == 1:
        X, y = load_breast_cancer(return_X_y = True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y)
 
    #Preprocess
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    #Selected features
    selector = mECFS(k_select = 10, n_init = n_init, use_xgbt = use_xgb).fit(X_train, y_train)

    X_train_m = selector.transform(X_train)
    X_test_m = selector.transform(X_test)

    model = ExtraTreesClassifier(128).fit(X_train_m, y_train)

    p = model.predict(X_test_m)

    print("Selected Features:")
    print(classification_report(y_test, p))

    #Random Features
    s = np.random.choice([i for i in range(X.shape[1])], 10)

    X_train_r = X_train[:, s]
    X_test_r = X_test[:, s]

    model = ExtraTreesClassifier(128).fit(X_train_r, y_train)

    p = model.predict(X_test_r)

    print("Random Features:")
    print(classification_report(y_test, p))

    #All features
    model = ExtraTreesClassifier(128).fit(X_train, y_train)

    p = model.predict(X_test)

    print("All Features:")
    print(classification_report(y_test, p))

    return selector

if __name__ == "__main__":

    pass