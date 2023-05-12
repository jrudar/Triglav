# Triglav - Feature Selection Using Iterative Refinement

[![CI](https://github.com/jrudar/Triglav/actions/workflows/ci.yml/badge.svg)](https://github.com/jrudar/Triglav/actions/workflows/ci.yml)
[![Draft PDF](https://github.com/jrudar/Triglav/actions/workflows/draft-pdf.yml/badge.svg)](https://github.com/jrudar/Triglav/actions/workflows/draft-pdf.yml)

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
   of reasoning applies. Clusters that are not rejected remain tentative.

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

## Interface

An overview of the API can be found [here](docs/API.md).

## Usage and Examples

Examples of how to use `Triglav` can be found [here](notebooks/README.md).

## Contributing

To contribute to the development of `Triglav` please read our [contributing guide](docs/CONTRIBUTING.md)

## References

Coming Soon

