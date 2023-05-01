# Triglav - Feature Selection Using Iterative Refinement

## Overview

Triglav (named after the Slavic god of divination) attempts to discover
all relevant features using an iterative refinement approach. This
approach is based after the method introduced in Boruta with several
modifications:

1) Features are clustered and the impact of each cluster is assessed as
   the average of the Shapley scores of the features associated with
   each cluster.

2) Like Boruta, a set of shadow features is created. However, an ensemble
   of classifiers is used to measure the Shapley scores of each real feature 
   and its shadow counterpart, producing a distribution of scores. A Wilcoxon 
   signed-rank test is used to determine the significance of each cluster
   and p-values are adjusted to correct for multiple comparisons across each 
   round. Clusters with adjusted p-values below 'alpha' are considered a hit.

3) At each iteration at or over 'n_iter_fwer', two beta-binomial distributions 
   are used to determine if a cluster should be retained or not. The first
   distribution models the hit rate while the the second distribution models 
   the rejection rate. For a cluster to be successfully selected the probability 
   of a hit must be significant after correcting for multiple comparisons and
   applying a Bonferroni correction for each iteration greater than or equal
   to the 'n_iter_fwer' parameter. For a cluster to be rejected a similar round
   of reasoning applies. Clusters that are neither selected or rejected remain
   tentative.

4) After the iterative refinement stage SAGE scores could be used to select
   the best feature from each cluster.

While this method may not produce all features important for classification,
it does have some nice properties. First of all, by using an Extremely 
Randomized Trees model as the default, dependencies between features can be 
accounted for. Further, decision tree models are better able to partition 
the sample space. This can result in the selection of both globally optimal
and locally optimal features. Finally, this approach identifies stable clusters of 
features since only those which consistently pass the Wilcoxon signed-rank test 
are selected. This makes this approach more robust to differences in training
data.

## Install

With Conda from BioConda:

```bash
conda install -c bioconda triglav
```

From PyPI:

```bash
pip install triglav
```

From source:

```bash
git clone https://github.com/jrudar/Triglav.git
cd Triglav
pip install .
# or create a virtual environment
python -m venv venv
source venv/bin/activate
pip install .
```

## Class Parameters

    transformer: default = NoScale()
        The transformer to be used to scale features. One can use
        the scikit-learn.preprocessing transformers. In addition,
        CLR and Scaler (converts each row into frequencies) are
        available by importing 'CLRTransformer' and 'Scaler' from the
        'triglav' package.
	
    sampler: default = NoResample()
        The resampling method used for imbalanced classes. Should be
        compatable with 'imblearn' or use an 'imblearn' resampler.

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
        modelling failures.

    metric: str, default = "correlation"
        The dissimilarity measure used to calculate distances between
        features. To use Extremely Randomized Trees proximities one
        has to import 'ETCProx' from the 'triglav' package.

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

    run_stage_2: bool, default = True
        This stage will determine the best feature from each of the
        selected clusters by calculating SAGE values.

    verbose: int, default = 0
        Specifies if basic reporting is sent to the user.

    n_jobs: int, default = 10
        The number of threads

## Fit Parameters

        Inputs:

        X: NumPy array of shape (m, n) where 'm' is the number of samples and 'n'
        the number of features (taxa, OTUs, ASVs, etc).

        y: NumPy array of shape (m,) where 'm' is the number of samples. Each entry
        of 'y' should be a factor.

        Returns:

        A fitted Triglav object.

## Transform Parameters

        Input:

        X: NumPy array of shape (m, n) where 'm' is the number of samples and 'n'
        the number of features (taxa, OTUs, ASVs, etc).

        Returns:

        X_transformed: NumPy array of shape (m, p) where 'm' is the number of samples and 'p'
        the number of features (taxa, OTUs, ASVs, etc). 'p' <= m

## Disclaimer

This code is still in development. USE AT YOUR OWN RISK.

## References

Coming Soon

