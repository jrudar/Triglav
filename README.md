### Triglav - Feature Selection Using Iterative Refinement

### Overview
    Triglav (named after the Slavic god of divination) attempts to discover
    all relevant features using an iterative refinement approach. This
    approach is based after the method introduced in Boruta with several
    modifications:
    
    1) The "importance" of real and shadow features are based off of
       Shapley scores.
       
    2) Features are clustered and an ensemble approach is used to identify 
       impactful clusters. Shapley scores of real feature are compared to 
       their shadow counterparts using a Wilcoxon signed-rank test. p-values 
       are adjusted to correct for multiple comparisons across each round. 
       Only features below a pre-specified alpha are considered a "hit".
       This should return all relevant features since features are chosen based
       on which clusters are retained. Furthermore, this approach is model
       avoids potential biases associated with feature importance scores from
       Random Forest models.
       
    3) A beta-binomial distribution is used to calculate the p-value
       associated with each hit. These are corrected for multiple
       comparisions (FDR and FWER). This model may be more appropriate since
       the successful selection of a cluster at each round may be less likely
       due since only one feature per cluster is tested.
        
    5) After the iterative refinement stage SAGE scores are used to select
       the best feature from each cluster.

### Install
Once downloaded, go to the location of the download and type:
    pip install triglav-v0.0.1b.tar.gz
    
### Class Parameters

    estimator: default = ExtraTreesClassifier(512, bootstrap = True)
        The estimator used to calculate Shapley scores.

    stage_2_estimator: default = ExtraTreesClassifier(512)
        The estimator used to calculate SAGE values. Only used if the
        'run_stage_2' is set to True.

    n_iter: int, default = 40
        The number of iterations to run Triglav.

    p_1: float, default = 0.65
        Used to determine the shape of the Beta-Binomial distribution
        modelling hits.

    p_2: float, default = 0.30
        Used to determine the shape of the Beta-Binomial distribution
        modelling failures.

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

    scale: bool, default = True
        Scales the data so the sum of each row is equal to one.

    clr_transform: bool, default = True
        Applies the centered log ratio to the dataset.

    run_stage_2: bool, default = True
        This stage will determine the best feature from each of the
        selected clusters by calculating SAGE values.

    verbose: int, default = 0
        Specifies if basic reporting is sent to the user.

    n_jobs: int, default = 10
        The number of threads
            
### Fit Parameters
        Inputs:

        X: NumPy array of shape (m, n) where 'm' is the number of samples and 'n'
        the number of features (taxa, OTUs, ASVs, etc).

        y: NumPy array of shape (m,) where 'm' is the number of samples. Each entry
        of 'y' should be a factor.

        Returns:

        A fitted Triglav object.

### Transform Parameters
        Input:

        X: NumPy array of shape (m, n) where 'm' is the number of samples and 'n'
        the number of features (taxa, OTUs, ASVs, etc).

        Returns:

        X_transformed: NumPy array of shape (m, p) where 'm' is the number of samples and 'p'
        the number of features (taxa, OTUs, ASVs, etc). 'p' <= m
        
### Example Usage - Set Up Triglav and Visualize Dendrogram
	from mECFS import Triglav

	from sklearn.datasets import make_classification
	from sklearn.preprocessing import StandardScaler
	from sklearn.model_selection import train_test_split
	from sklearn.ensemble import ExtraTreesClassifier

	if __name__ == "__main__":

	    #Create the dataset. The useful features are found in the first 7 columns of X (indicies 0-6)
	    X, y = make_classification(n_samples = 200,
				       n_features = 20,
				       n_informative = 5,
				       n_redundant = 2,
				       n_repeated = 0,
				       n_classes = 2,
				       shuffle = False,
				       random_state = 0)

	    #Split into train and test sets
	    X_train, X_test, y_train, y_test = train_test_split(X, y, 
								test_size = 0.2, 
								random_state = 0, 
								stratify = y)

	    #Standardize
	    s_trf = StandardScaler().fit(X_train)
	    X_train = s_trf.transform(X_train)
	    X_test = s_trf.transform(X_test)

	    #Set up Triglav
	    model = Triglav(n_jobs = 4, scale = False, clr_transform=False)

	    #Visualize clustering
	    model.visualize_hclust(X_train, y_train)
	    
	    #Reset the threshold based on inspection of the dendrogram
	    model.thresh = 1.25

	    #Identify predictive features
	    model.fit(X_train, y_train)

	    #Transform the test data
	    X_test_trf = model.transform(X_test)
    
![alt text](https://github.com/jrudar/Triglav/blob/main/Triglav_Dend.jpg?raw=true)
    
### Disclaimer
This code is still in development. USE AT YOUR OWN RISK.

### References

	Coming Soon
