### Feature Selection Using Iterative Refinement

### Install
Once downloaded, go to the location of the download and type:
    pip install triglav-1.0.0.dev0.tar.gz
    
### Class Parameters
    Inputs:

    threshold and threshold_2: int, default = 95
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
