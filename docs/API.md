## Overview of Section

This section provides an overview of the `Triglav` and the different parameters
of the `Triglav` class and its methods.

## Class

    class triglav.Triglav(transformer = NoScale(), sampler = NoResample(), estimator = ExtraTreesClassifier(512, bootstrap = True),
                  stage_2_estimator = ExtraTreesClassifier(512, bootstrap = True), per_class_imp = False,
                  n_iter = 40, n_iter_fwer = 11, p_1 = 0.65, p_2 = 0.30, metric = "euclidean", linkage = "ward",
                  thresh = 2.0, criterion = "distance", run_stage_2 = True, verbose = 0, n_jobs = 10)

### Parameters

    transformer: default = NoScale()
        The transformer to be used to scale features.

    sampler: default = NoResample()
        The type of sampler (from Imbalanced-learn) to use.

    estimator: default = ExtraTreesClassifier(512, bootstrap = True)
        The estimator used to calculate Shapley scores.

    stage_2_estimator: default = ExtraTreesClassifier(512)
        The estimator used to calculate MultiSURF CV scores.
        Only used if the 'run_stage_2' is set to True or 'mms'.

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

    metric: str, default = "euclidean"
        The dissimilarity measure used to calculate distances between
        features.

    linkage: str, default = "ward"
        The type of hierarchical clustering method to apply. The available
        methods include: single, complete, ward, average, centroid.

    thresh: float, default = 2.0
        The threshold or max number of clusters.

    criterion: str, default = "distance"
        The method used to form flat clusters. The available methods
        include: distance or maxclust.

    alpha: float, default = 0.05
        The level at which corrected p-values will be rejected.

    run_stage_2: str or bool, default = "mms"
        This stage will determine the best features from the selected
        Triglav features. If 'str' is "auto", swarm optimization is used.
        If "mms" (default), a modified version of the MultiSURF algorithm
        is used. If True, "mms" is used. If False, stage 2 is not run.

    verbose: int, default = 0
        Specifies if basic reporting is sent to the user.

    n_jobs: int, default = 10
        The number of threads

### Attributes

    n_classes_: ndarray of shape (n_classes,)
        The class labels

    n_class_: int
        The number of classes

    selected_: ndarray of shape (n_features,)
        The mask of the selected features.

    selected_best_: ndarray of shape (n_features,) or None
        The mask of the best features from each cluster. Only returns an ndarray
        if the 'run_stage_2' parameter is enabled.

    self.task_opt_: Task Object
        MealPy task optimizer object.

    linkage_matrix_: ndarray
        The SciPy hierarchical clustering encoded as a linkage matrix.

### Methods

    fit(X, y, **fit_params)
        Fits a `Triglav` feature selection model.

        Parameters:

        X: NumPy array of shape (m, n) where 'm' is the number of samples and 'n'
        the number of features (taxa, OTUs, ASVs, etc).

        y: NumPy array of shape (m,) where 'm' is the number of samples. Each entry
        of 'y' should be a factor.

        Returns:

        A fitted Triglav object.

    transform(X)
        Reduces X to the selected features contained the 'selected_' attribute.

        Input:

        X: NumPy array of shape (m, n) where 'm' is the number of samples and 'n'
        the number of features (taxa, OTUs, ASVs, etc).

        Returns:

        X_transformed: NumPy array of shape (m, p) where 'm' is the number of samples and 'p'
        the number of features (taxa, OTUs, ASVs, etc). 'p' <= m

    fit_transform(X, y, **fit_params)
        Fits a `Triglav` feature selection model then reduces X to the selected features
        contained in the 'selected_' attribute.

        Parameters:

        X: NumPy array of shape (m, n) where 'm' is the number of samples and 'n'
        the number of features (taxa, OTUs, ASVs, etc).

        y: NumPy array of shape (m,) where 'm' is the number of samples. Each entry
        of 'y' should be a factor.

        Returns:

        X_transformed: NumPy array of shape (m, p) where 'm' is the number of samples and 'p'
        the number of features (taxa, OTUs, ASVs, etc). 'p' <= m

    visualize_hclust(X, y, ax, **dendrogram_kwargs)
        Creates a visualization of the hierarchical clustering specified using the 'metric',
        'thresh', 'linkage', and 'criterion' parameters of the `Triglav`

        Parameters:

        X: NumPy array of shape (m, n) where 'm' is the number of samples and 'n'
        the number of features (taxa, OTUs, ASVs, etc).

        y: NumPy array of shape (m,) where 'm' is the number of samples. Each entry
        of 'y' should be a factor.

        ax: matplotlib.axes.Axes (optional). The axes on which to plot the dendrogram. If None,
        the dendrogram will be plotted to a new figure sublot axis.

        Returns:

        dict: A dictionary of data structures computed to render the dendrogram.
        See https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.dendrogram.html
        for more details.

## Class

    class triglav.ETCProx(n_estimators = 1024, min_samples_split = 0.33, n_sets = 5)

### Parameters

    n_estimators: int, default = 1024
        The number of estimators the ExtraTreesClassifier will use.

    min_samples_split: float, default = 0.33
        The minimum number of samples to split an internal node.

    n_sets: int, default = 5
        The number of random datasets to be generated.

### Methods

    fit_transform(X, y, **fit_params)
        Fits a `ETCProx` class and returns the dissimilarity matrix.

        Parameters:

        X: NumPy array of shape (m, n) where 'm' is the number of samples and 'n'
        the number of features (taxa, OTUs, ASVs, etc).

        Returns:

        Dissimilarity Matrix: NumPy array of shape (n, n) where 'n' is the
        the number of features (taxa, OTUs, ASVs, etc)

## Class

    class triglav.NoScale()

    class triglav.Scaler()

    class triglav.NoResample()

### Parameters

    None

### Attributes

    zero_samps: bool, ndarray of shape (n_samples,)
        The mask of the all rows which sum to zeroest features from each cluster.
        This is only returned for the Scaler() and CLRTransformer() classes.

### Methods

    fit_transform(X, y = None, **fit_params)
        Fits a transformer method.

        Parameters:

        X: NumPy array of shape (m, n) where 'm' is the number of samples and 'n'
        the number of features (taxa, OTUs, ASVs, etc).

        Returns:

        X_Transformed: NumPy array of shape (m, n) where 'm' is the number of 
        samples and 'n' the number of features (taxa, OTUs, ASVs, etc).'

        NoScale will return X
        Scaler will return the closure of X (all rows sum to one, X must be non-negative)
        NoResample will return X

