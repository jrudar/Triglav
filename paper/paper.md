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

`Triglav`, named after the Slavic god of divination, is a Python package which can be used to identify relevant and stable sets 
of features in high-dimensional datasets. `Triglav` is a wrapper feature selection algorithm applicable to tabular datasets. 
`Triglav`, which was inspired by Boruta [@JSSv036i11], uses an iterative approach to identify a stable and predictive subset 
of features. Briefly, an ensemble approach is used to identify impactful clusters of features and the consistent identification 
of impactful clusters over many iterations determines if a cluster of features is retained or discarded. This approach is 
particularly beneficial since the identification of impactful clusters (and features) is accomplished using explainable artificial 
intelligence approaches, which have already been shown to be useful for feature selection [@Thanh-Hai2021]. Further, we show a
real-world application of Triglav using an amplicon sequencing dataset containing reads from the 16S rRNA gene of patients
suffering from Crohn's Disease and healthy controls [@CD]. Using this data we provide an end-to-end workflow demonstrating how `Triglav` 
can be used to analyze metagenomic data and show that `Triglav` identifies a set of features that are more stable than those
identified by competing methods \autoref{fig:overview1} [@Stability]. This is important since identifying stable set of predictive features 
is necessary since they may point to useful interpretations of the data.

# Statement of need

Modern data has become increasingly complex, with the number of generated features growing larger for many datasets. 
This increase can make the analysis of this data difficult due to the inclusion of noise and other irrelevant features.
To tackle this problem, feature selection methods are often used to reduce the complexity of the data while identifying 
the most relevant features given the task at hand. With genomic and metagenomic datasets this task has become increasingly 
important since generating models of the data and an understanding of how these models work directly improves our 
knowledge of complex systems such as disease process, viral transmission, and how ecosystems change over time. While most 
feature selection approaches tend to remove redundant features, this may not necessarily be what is best in the case of 
biological data. Modern genomic and metagenomic data is complex and often it is important to identify the relevant functional 
changes which occur in these communities or network of genes [@lowabdmicrobiome; @SingleCell]. Therefore, the removal of 
redundant features could obfuscate important biological insights since the function of a particular organisms or gene would 
not be included in the analysis. The mitigation of this problem was the driving force behind the development of `Triglav`.
Specifically, we believe that an approach capable of identifying all relevant predictive features using explainable artificial 
intelligence is needed as it would help ensure that the selected features reflect actual differences, and not the nuances the
between different sets of training data. Finally, work such as this is needed since it allows for a generalized way to measure
differentially abundant species or genes. By pursuing this line of investigation, differential abundance testing will
no longer rely on the performance of a particular statistical model. Rather, it will be directly tied to the ability of
a machine learning model to successfully classify a dataset.

![`Triglav` analysis identifies a stable set of features from a real-world dataset of 16S rRNA amplicon sequencing data from patients suffering from Crohn's Disease and healthy controls [@CD].
**A**, a comparison of `Triglav` performance against several common approaches.
**B**, SAGE importance scores from each of the selected features. Higher scores are indicative of more important features.
Many of the selected features were also detected in @CD.
**C**, a clustermap of the top features from each cluster visualizing differences in the microbiomes of healthy patients (blue) and those suffering from Crohn's Disease (red).
\label{fig:overview1}](Figure 1.svg)

# Outline of the Triglav Algorithm

The core assumption behind `Triglav` is that clusters of impactful features sharing similar pattern of values across all samples should be discoverable. 
Since this is not an unreasonable assumption for biological datasets, different patterns, for example, in the abundance of gut bacterial species could exist between healthy controls and Crohn's Disease patients [@CD]. 
To take advantage of this observation, `Triglav` begins by clustering features (\autoref{fig:overview2}A) [@2020SciPy-NMeth] followed by the random selectiion of one feature from each cluster. 
A set of shadow features, which are copies of the original set of selected features where the marginal distributions of each feature has been permuted, are then created (\autoref{fig:overview2}B) [@JSSv036i11]. 
The shadow data is then combined with the original data and used to train a classification model [@JSSv036i11] and calculate Shapley values (\autoref{fig:overview2}C) [@shapley1951notes; @SHAP1; @SHAP2]. 
This process is repeated a number of times to generate a distribution of Shapley values associated with each cluster of features and their shadow counterparts. 
For each iteration of the `Triglav` algorithm a Wilcoxon signed-rank test is then used to determine if the distribution of Shapley values associated to each cluster of real features is greater than the corresponding shadow cluster (Figure 2C) [@wilcoxon]. 
A binary matrix where '1' represents a cluster of features differing significantly from its shadow counterpart is then appended to at end of each iteration (\autoref{fig:overview2}D). 
A beta-binomial distribution then uses this matrix to determine if a cluster should be selected while a second beta-binomial distribution, using a different parameterization, is used to determine if a cluster should be rejected (\autoref{fig:overview2}E).
By using two differently parameterized beta-binomial distributions, `Triglav` has a better ability to control the selection and rejection of clusters. Once a cluster a significant hit is detected, the cluster is removed from the pool of tentative clusters
and the process begins again. Finally, the best feature from each cluster can be discovered by calculating SAGE importance scores [@SAGE]. This step is optional but can be done to remove potential redundancies between features. 

![A high-level overview of the `Triglav` algorithm. (A) Features are clustered. A number of machine learning models are trained on randomly selected set of features and their shadow counterparts from each cluster (B).
This process is repeated to generate a distribution of Shapley values. A Wilcoxon signed-rank test is used to determine when a cluster's Shapley values are greater than the shadow counterpart (C and D). Beta-binomial distributions
are then used to determine if a feature is to be kept, rejected, or remain tentative (E). Kept and rejected features are removed and steps B-E are repeated using the remaining tentative features. 
False discovery rate corrections are applied at step C and E.
\label{fig:overview2}](Figure 2.svg)

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

# Figures

![Results of an analysis of a small amplicon sequencing dataset to which `Triglav` was applied. Compared to some commonly used
approaches, there is evidence that the feature sets produced by `Triglav` are stable when different training subsets are used (A). 
In (B) SAGE importance scores from each of the selected features are shown. Many of the selected features were also detected in 
[@CD]. A clustermap (C) of the top features from each cluster can then be plotted. This clustermap clearly visualizes differences 
in the microbiomes of healthy patients (blue) and those suffering from Crohn's Disease (red).\label{fig:overview1}](Figure 1.png)

![A high-level overview of the first half of the `Triglav` algorithm. The output of this part of the algorithm is a binary matrix 
specifying if the distribution of Shapley values associated with a cluster of features differs significantly from the distribution 
associated with the corresponding shadow cluster. False discovery rate corrections are applied at this step.\label{fig:overview2}](Figure 2.png)

![A high-level overview of the second half of the `Triglav` algorithm. The output of this part of the algorithm is a list of 
selected features. Two different beta-binomial distributions are used to determine if a feature is selected or rejected. These 
distributions are used by `Triglav` since they can model over-dispersion in zero-counts due to the random selection of features in 
the first-half of the algorithm. For a feature to be selected, the number of times a significant difference was observed should 
fall within the critical region determined by the survival function of the first distribution or the cumulative distribution 
function of the second. A false-discovery rate and Bonferroni correction are applied at this step.\label{fig:overview3}](Figure 3.png)

# References