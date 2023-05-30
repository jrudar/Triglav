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
of features in high-dimensional datasets. `Triglav`, a wrapper feature selection algorithm inspired by Boruta and BorutaShap [@JSSv036i11; @BortuaShap], can be applied to tabular datasets and 
uses an iterative approach to identify a stable and predictive subset 
of features. In biological data these features can include, but are not limited to, genes, species, and single nucleotide variants. Briefly, an ensemble approach is used to identify impactful clusters of features and the consistent identification 
of impactful clusters over many iterations determines if a cluster of features is retained or discarded. Shapley values, which assess the
marginal contribution of a feature across coalitions of features, are used to identify impactful clusters of features [@shapley1951notes; @SHAP1; @SHAP2] and they have 
been shown to be useful for feature selection in biological data [@Thanh-Hai2021]. Further, we provide an end-to-end example of a
real-world application of Triglav using an amplicon sequencing dataset containing reads from the 16S rRNA gene of patients
suffering from Crohn's Disease and healthy controls [@CD]. Using this workflow we show how `Triglav` 
can be used to analyze metagenomic data and demonstrate that `Triglav` can identify a set of stable and predictive features \autoref{fig:overview1} [@Stability]. 
This is important since identifying stable sets of predictive features 
is necessary to properly form a useful interpretations of the data.

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
\label{fig:overview1}](Figure 1.pdf)

# Outline of the Triglav Algorithm

The core assumption behind `Triglav` is that clusters of impactful features sharing similar pattern of values across all samples should be discoverable. 
Since this is not an unreasonable assumption for biological datasets. For example, patterns in the abundance of gut bacterial species exist between healthy controls and Crohn's Disease patients [@CD].
`Triglav` takes advantage of these patterns by first clustering similar features together (\autoref{fig:overview2}A) [@2020SciPy-NMeth]. This is then followed by the random selection of one feature from each cluster and the generation 
of a set of shadow features, which are permuted copies of the selected features (\autoref{fig:overview2}B) [@JSSv036i11]. 
Many classification models are then trained on different sets of randomly selected features and shadow data to generate a distribution of Shapley values associated with each cluster (\autoref{fig:overview2}C) [@JSSv036i11; @shapley1951notes; @SHAP1; @SHAP2]. 
For each iteration of the `Triglav` algorithm a Wilcoxon signed-rank test is then used to determine if the distribution of Shapley values associated to each cluster of real features is greater than those from the corresponding shadow cluster (Figure 2C) [@wilcoxon]. 
A binary matrix where '1' represents a cluster of features differing significantly from its shadow counterpart is then updated at end of each iteration (\autoref{fig:overview2}D). 
A beta-binomial distribution then uses this matrix to determine if a cluster should be selected while a second beta-binomial distribution, using a different parameterization, is used to determine if a cluster should be rejected (\autoref{fig:overview2}E).
By using two differently parameterized beta-binomial distributions, `Triglav` has a better ability to control the selection and rejection of clusters. Once a significant hit is detected, the cluster is removed from the pool of tentative clusters
and the process begins again. Finally, the best feature from each cluster can be discovered by calculating SAGE importance scores [@SAGE]. This step is optional but can be done to remove potential redundancies between features. 

![A high-level overview of the `Triglav` algorithm. (A) Features are clustered. (B) A number of machine learning models are trained on randomly selected subset of features and their shadow counterparts.
(C) This process is repeated to generate a distribution of Shapley values and a Wilcoxon signed-rank test is used to determine if a cluster's Shapley values are greater than the shadow counterpart (C and D). Beta-binomial distributions
are then used to determine if a feature is to be kept, rejected, or remain tentative (E). Kept and rejected features are removed and steps B-E are repeated using the remaining tentative features. 
False discovery rate corrections are applied at step C and E.
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