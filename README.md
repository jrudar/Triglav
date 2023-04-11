### Triglav - Feature Selection Using Iterative Refinement

### Overview
    Triglav (named after the Slavic god of divination) attempts to discover
    all relevant features using an iterative refinement approach. This
    approach is based after the method introduced in Boruta with several
    modifications:
    
    1) The "importance" of real and shadow features are based off of
       Shapley scores.
       
    2) Features are clustered and nn ensemble approach is used to identify 
       impactful clusters. Shapley scores of real feature are compared to 
       their shadow counterparts using a Wilcoxon signed-rank test. p-values 
       are adjusted to correct for multiple comparisons across each round. 
       Only features below a pre-specified alpha are considered a "hit".
       
    3) A beta-binomial distribution is used to calculate the p-value
       associated with each hit. These are corrected for multiple
       comparisions (FDR and FWER).
        
    5) After the iterative refinement stage SAGE scores are used to select
       the best feature from each cluster.

### Install
Once downloaded, go to the location of the download and type:
    pip install triglav-v0.0.1b.tar.gz
    
### Class Parameters
    Inputs:

    threshold and threshold_2: int, default = 99.5 and 100
        The threshold for comparing shadow and real features in the 
        when using SHAP and SAGE scores.

    metric: str, default = "correlation"
        The dissimilarity measure used to calculate distances between
        features.

    linkage: str, default = "complete"

    thresh: float, default = 2.0

    criterion: str, default = "distance"

    p: float, default = 0.35
        The 'p' parameter used to determine the shape of the Beta-Binomial 
        distribution.

    alpha: float, default = 0.05
        The level at which corrected p-values will be rejected.

    verbose: int, default = 0
        Specifies if basic reporting is sent to the user.

    n_jobs: int, default = 3
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
