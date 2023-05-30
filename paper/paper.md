---
title: 'Triglav: Iterative Refinement and Selection of Stable Features Using Shapley Values'
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
    affiliation: "3,4"
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
 - name: Deptartment of Biological Sciences, University of Manitoba, 50 Sifton Road, Winnipeg, Manitoba R3T 2N2 Canada.
   index: 4
date: 2 May 2023
bibliography: paper.bib
---

# Summary

`Triglav` is a Python package implementing a feature selection algorithm applicable for identification 
of relevant and stable sets of features in high-dimensional tabular datasets. Like `Boruta` and `BorutaShap`, it uses an iterative approach to 
identify a stable and predictive subset of features [@JSSv036i11; @BortuaShap; @Stability].
Briefly, an ensemble approach is used to identify impactful clusters of features and the consistent identification 
of impactful clusters over many iterations determines if a cluster of features is retained or discarded. 
This approach is particularly beneficial since the identification of impactful clusters (and features) is accomplished using explainable artificial 
intelligence approaches shown to be useful for feature selection [@Thanh-Hai2021]. 
Further, we demonstrate how `Triglav` can be used to identify a stable set of features from a real-world dataset of 16S rRNA amplicon sequencing data from patients 
suffering from Crohn's Disease and healthy controls [@CD]. 
With this metagenomic data, we show that `Triglav` identifies a set of features more stable than those identified by competing methods (see \autoref{fig:overview1}A) [@Stability]. 
By identifying stable sets of predictive features, `Triglav` may lead to useful interpretations of the underlying data.

# Statement of need

As datasets grow in complexity and in the number of features, analysis becomes increasingly difficult due to noise and the presence of irrelevant features.
To tackle this problem, feature selection methods are often used to reduce the complexity of the data while identifying the most relevant
features given the task at hand [@JSSv036i11; @BortuaShap; @NoiseR]. With genomic and metagenomic datasets, this task has become increasingly important since generating 
models of the data and an understanding of how these models work directly improves our knowledge of complex systems such as disease process, 
viral transmission, and how ecosystems change over time [@LANDMark; @TreeOrdination; @MLVir]. Many feature selection approaches tend to remove redundant features, however, 
this may not be optimal for biological datasets. With complex modern genomic and metagenomic data, it is often important to identify the 
relevant functional changes occurring in microbiomes or networks of genes [@lowabdmicrobiome; @SingleCell]. However, the removal of redundant 
features could obfuscate important biological insights since the function of particular organism or gene may not be included in downstream analyses. 
Therefore, `Triglav` was developed to implement an approach capable of identifying all relevant predictive features using explainable artificial
intelligence to ensure that the selected features are enriched with those that reflect actual differences and not the nuances the between different sets of training data. 
This results in a generalized way to measure differentially abundant species or genes in biological datasets. 
With the `Triglav` approach, these measurements would no longer rely on the performance of a particular statistical model, but rather, the ability of a 
machine learning model to successfully classify a dataset.

![`Triglav` analysis identifies a stable set of features from a real-world dataset of 16S rRNA amplicon sequencing data from patients suffering from Crohn's Disease and healthy controls [@CD].
**A**, a comparison of `Triglav` performance against several common approaches.
**B**, SAGE importance scores from each of the selected features.
Many of the selected features were also detected in @CD.
**C**, a clustermap of the top features from each cluster visualizing differences in the microbiomes of healthy patients (blue) and those suffering from Crohn's Disease (red).
\label{fig:overview1}](Figure 1.pdf)

# Outline of the Triglav Algorithm

The core assumption behind `Triglav` is that clusters of impactful features sharing similar pattern of values across all samples should be discoverable. 
This is not an unreasonable assumption for biological datasets. For example, patterns in the abundance of gut bacterial species exist between healthy controls and Crohn's Disease patients [@CD].
To take advantage of this observation,`Triglav` begins by clustering features using a user-supplied dissimilarity measure (\autoref{fig:overview2}A) [@2020SciPy-NMeth]. The feature clustering
is only performed once and the result is used in subsequent steps.
Following clustering, one feature from each cluster is randomly selected. A set of shadow features is then constructed by sampling without replacement from the marginal distribution of each real feature (\autoref{fig:overview2}B) [@JSSv036i11].
The real and shadow data is combined and used to train an Extremely Randomized Tree classifier in the default implementation of `Triglav`. Shapley values are then calculated (\autoref{fig:overview2}C) [@ETC; @BortuaShap; @JSSv036i11; @shapley1951notes; @SHAP1; @SHAP2]. 
This process is repeated to generate a distribution of Shapley values associated with each cluster of features and their shadow counterparts.
A Wilcoxon signed-rank test is then used to determine if the distribution of Shapley values belonging to each cluster of real features is greater than those from the corresponding shadow cluster (Figure 2C) [@wilcoxon]. 
These steps are then repeated multiple times, generating a binary matrix where '1' represents a cluster of features differing significantly from its shadow counterpart (\autoref{fig:overview2}D).
A beta-binomial distribution is then used to determine if a cluster should be selected. A second beta-binomial distribution,
using a different parameterization, is used to determine if a cluster should be rejected (\autoref{fig:overview2}E).
By using two differently parameterized beta-binomial distributions, `Triglav` has a better ability to control the selection and rejection of clusters. Once a significant hit is detected, the cluster is removed from the pool of tentative clusters
and the process begins again. Finally, if there is interest in removing potentially redundant features, the best feature from each cluster can be discovered by calculating SAGE importance scores [@SAGE]. 

![A high-level overview of the `Triglav` algorithm. 
**A**, Features are clustered.
**B**, A number of Extremely Randomized Tree classifiers are trained on randomly selected subsets of features and their shadow counterparts.
**C**, This process is repeated to generate a distribution of Shapley values. 
**D**, A Wilcoxon signed-rank test is used to determine if a cluster's Shapley values are greater than the shadow counterpart. 
**E**, Beta-binomial distributions are then used to determine if a feature is to be kept, rejected, or remain tentative. 
Kept and rejected features are removed and steps **B-E** are repeated using the remaining tentative features. 
False discovery rate corrections are applied at step **D** and **E**.
\label{fig:overview2}](Figure 2.pdf)

# Ongoing Research

Currently, this method is being used in projects to discover features capable of predicting host-origin of viral samples and strain
specific bacterial markers at the National Centre for Foreign Animal Disease with the Canadian Food Inspection Agency. In addition 
to this work, we hope to integrate `Triglav` into an end-to-end suite of software with our previously developed tools, `LANDMark` and 
`TreeOrdination` [@LANDMark; @TreeOrdination]. Together, this will form the basis of a modern toolset capable of investigating
the organisms and genes associated with pathogenicity and environmental outcomes.

# Acknowledgements

We thank Dr. Terri M. Porter, Dr. Oksana Vernygora, and Hoang Hai Nguyen for their thoughtful review of the manuscript and code.
J.R. is supported by funds from the Food from Thought project as part of Canada First Research Excellence Fund and from CSSP-CFPADP-1278. 
M.H. received funding from the Government of Canada through Genome Canada and Ontario Genomics. G.B.G. is supported by a Natural 
Sciences and Engineering Research Council of Canada (NSERC) grant (RGPIN-2020-05733).

# References