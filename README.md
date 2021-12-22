### Modified Eigenvector Centrality Feature Selection

### Install
Once downloaded, go to the location of the download and type:
    pip install mECFS-1.0.0.dev0.tar.gz
    
### Class Parameters
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
        features. If False, Extremely Randomized Trees will be used.

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
        the number of features (taxa, OTUs, ASVs, etc). 'p' <= m
        
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

