### Feature Selection Using Iterative Refinement

### Overview
    Triglav (named after the Slavic god of divination) attempts to discover
    all relevant features using an iterative refinement approach. This
    approach is based after the method introduced in Boruta with several
    modifications:
    
    1) The "importance" of real and shadow features are based off of
       Shapley and SAGE scores.
       
    2) An ensemble approach is used to identify "hits" in each round and
       each round consists of two selection stages: Generally, in the 
       initial round features are selected using Shapley scores, while
       SAGE scores are always calculated in the second round. The initial
       round includes all features selected from any two of the following
       models: SGD Classifier, Extra Trees, Linear SVC, Logisitic
       Regression, and Mutual Information. The second round only uses the
       Extra Trees Classifier.
       
    3) A beta-binomial distribution is used to calculate p-values.
    
    4) A two-step correction for p-values is used.
    
    5) After the iterative refinement stage, a dataframe is constructed
       using the remaining real and shadow features. The final set of
       features are constructed using from SAGE scores greather than the
       n-th percentile of the shadow scores.

### Install
Once downloaded, go to the location of the download and type:
    pip install triglav-v0.0.1.tar.gz
    
### Class Parameters
    Inputs:

    threshold and threshold_2: int, default = 95
        The threshold for comparing shadow and real features in the
        first and second stage.

    a and a_2: float, default = 24 / 20
        The 'a' parameter of the Beta-Binomial distribution at stage 1 and 2.

    b and b_2: float, default = 32 / 32
        The 'b' parameter of the Beta-Binomial distribution at stage 1 and 2.

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
            
### Fit Parameters
        Inputs:

        X: NumPy array of shape (m, n) where 'm' is the number of samples and 'n'
        the number of features (taxa, OTUs, ASVs, etc).

        y: NumPy array of shape (m,) where 'm' is the number of samples. Each entry
        of 'y' should be a factor.

        Returns:

        A fitted mECFS object.

### Transform Parameters
        Input:

        X: NumPy array of shape (m, n) where 'm' is the number of samples and 'n'
        the number of features (taxa, OTUs, ASVs, etc).

        Returns:

        X_transformed: NumPy array of shape (m, p) where 'm' is the number of samples and 'p'
        the number of features (taxa, OTUs, ASVs, etc). 'p' <= m
        
### Example Usage
        from triglav import Triglav
        from sklearn.datasets import make_classification
        
        X, y = make_classification()
        
        model = Triglav().fit(X, y)

	X_transformed = model.transform(X)

### Disclaimer
This code is still in development. USE AT YOUR OWN RISK.

### References

	Coming Soon
