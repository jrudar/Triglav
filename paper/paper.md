---
title: 'Knock Knock. Who’s There? Triglav: Iterative Refinement and Selection of Stable Features Using Shapley Scores'
tags:
  - Python
  - feature selection
  - genomics
  - metabarcoding
  - machine learning
authors:
  - name: Josip Rudar
    orcid: 0000-0003-0484-8028
    corresponding: true
    affiliation: "1,3"
  - name: Peter Kruczkiewicz
    orcid: 0000-0002-0044-9460
    affiliation: 3
  - name: Oliver Lung
    orcid: 0000-0002-0044-9460
    affiliation: 3
  - name: G. Brian Golding
    orcid: 0000-0002-7575-0282
    affiliation: 2
  - name: Mehrdad Hajibabaei
    orcid: 0000-0002-8859-7977
    corresponding: true
    affiliation: 1
affiliations:
 - name: Centre for Biodiversity Genomics at Biodiversity Institute of Ontario and Department of Integrative Biology, University of Guelph, 50 Stone Road East, Guelph, ON, N1G 2W1, Canada
   index: 1
 - name: Department of Biology, McMaster University, 1280 Main St. West, Hamilton, ON, L8S 4K1, Canada
   index: 2
 - name: National Centre for Foreign Animal Disease, Canadian Food Inspection Agency, Winnipeg, Manitoba, Canada
   index: 3
date: 2 May 2023
bibliography: paper.bib
---

# Summary

Modern data has become increasingly complex, with the number of generated features growing larger for many datasets. This increase can make the analysis of this data difficult due to the inclusion of noise and other irrelevant features. To tackle this problem, feature selection methods are often used to reduce the complexity of the data while identifying the most relevant features given the task at hand. With genomic and metagenomic datasets this task has become increasingly important since generating models of the data and an understanding of how these models work directly improves our knowledge of complex systems such as disease process, viral transmission into new hosts, and how ecosystems change over time. While most feature selection approaches tend to remove redundant features, this may not necessarily be what is best in the case of biological data. Often, redundant features could allow for important biological insights since organisms and genes form interaction networks which should be considered together. Therefore, it is necessary to develop tools which can identify all relevant predictive features while also ensuring that the selected features reflect actual differences and not the nuances between different sets of training data.


# Statement of need

`Triglav`, named after the Slavic god of divination, is a Python package which can be used to identify relevant and stable sets of features in high-dimensional datasets. `Triglav`, which was inspired by Boruta, uses an iterative approach to identify a stable and predictive subset of features. Briefly, an ensemble approach is used to identify impactful clusters of features and the consistent identification of impactful clusters over many iterations determines if a cluster of features is retained or discarded. This approach is particularly beneficial since the identification of impactful clusters (and features) is accomplished by using explainable artificial intelligence approaches. This provides end-users of this package with the ability to understand which features are informative based on their impact on the model. Further, this package was tested to ensure that the features selected were stable across different training sets. This is important since a stable set of predictive features may point to a useful interpretation of the data.

# Outline of the Triglav Algorithm

The core assumption behind 'Triglav' is that clusters of features sharing similar values across all samples should be discoverable. This is not an unreasonable assumption in biological datasets. For example, different patterns in the abundance of gut bacteria could exist between healthy controls and
Crohn's Disease patients. To take advantage of this observation, `Triglav` begins by clustering features. The first stage of our approach randomly selects one feature from each cluster. A set of shadow features are then created by randomly sampling without replacement from the distribution of each selected feature. The
shadow data is then combined with the original data and is used to train a classification model. Shapley scores are then calculated. This process is repeated to generate a distribution of Shapley values associated with each cluster of features and their shadow counterparts. A Wilcoxon signed-rank test is then used to determine if the distribution of Shapley scores belonging to each cluster of real features is greater than the corresponding shadow cluster. These steps are repeated multiple times, generating a binary matrix where '1' represents a cluster of features differing significantly from its shadow counterpart. A beta-binomial distribution is then used to determine if a feature is to be selected. A second beta-binomial distribution is also used to determine when a feature is to be rejected. Finally, the best feature from each cluster can be optionally discovered by calculating the SAGE importance score. This step is optional. A visual overview is provided in Figure 1.

# Ongoing Research

# Figures

![Figure 1: A high-level overview of the the first half of the `Triglav` algorithm. The output of this part of the algorithm is a binary matrix specifying if the distribution of Shapley values associated with a cluster of features differs significantly from the distribution associated with the corresponding shadow cluster.](Figure 1.png)
