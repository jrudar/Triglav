### Modified Eigenvector Centrality Feature Selection
    This impliments a variant of Eigenvector Centrality Feature Selection which
    can be used with continuous and binary features. Feature selection also
    incorporates feature importance information forms supervised learning 
    algorithms (using Extremely Randomized Trees or XGBoost). Scores are converted
    into ranks and features are selected using permutation statistics.
    
### Install
    Once downloaded, go to the location of the download and type:
    
    pip install mECFS-1.0.0.dev.tar.gz
    
### Class Parameters
    Inputs:

    n_init: int, default = 30
        The number of resampling steps.

    alpha: float, default = 0.5
    	The weight parameter for each adjacency matrix.

    bootstrap: bool, default = True
        Specifies if bootstrap resampling will be used.

    use_xgbt: bool, default = True
        Specifies if XGB Trees will be used to detect interactions between
        features. If 'False', Extremely Randomized Trees will be used.
	
    n_jobs: int, default = 6
    	The number of processes spawned.

    Returns:

    A mECFS object.
            
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
        the number of features (taxa, OTUs, ASVs, etc). 'p' <= 'm'
        
### Example Usage

        from mECFS import mECFS
        from sklearn.datasets import make_classification
        
        X, y = make_classification()
        
        model = mECFS().fit(X, y)
	
        X_transformed = model.transform(X)

### Disclaimer
This code is still in development. USE AT YOUR OWN RISK.

### References

    Adapted From:
    https://www.mathworks.com/matlabcentral/fileexchange/56937-feature-selection-library

    @InProceedings{RoffoECML16, 
     author={G. Roffo and S. Melzi}, 
     booktitle={Proceedings of New Frontiers in Mining Complex Patterns (NFMCP 2016)}, 
     title={Features Selection via Eigenvector Centrality}, 
     year={2016}, 
     keywords={Feature selection;ranking;high dimensionality;data mining}, 
     month={Oct}}

