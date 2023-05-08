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

`Triglav` is a Python package implementing a feature selection algorithm applicable for identification of relevant and stable sets of features in high-dimensional tabular datasets. 
Like Boruta [@JSSv036i11], it uses an iterative approach to identify a stable and predictive subset of features. 
Briefly, an ensemble approach is used to identify impactful clusters of features and the consistent identification 
of impactful clusters over many iterations determines if a cluster of features is retained or discarded. 
This approach is particularly beneficial since the identification of impactful clusters (and features) is accomplished using explainable artificial 
intelligence approaches shown to be useful for feature selection [@Thanh-Hai2021]. 
Further, we demonstrate how `Triglav` can be used to identify a stable set of features from a real-world dataset of 16S rRNA amplicon sequencing data from patients suffering from Crohn's Disease and healthy controls [@CD]. 
With this metagenomic data, we show that `Triglav` identifies a set of features more stable than those identified by competing methods (see \autoref{fig:overview1}) [@Stability]. 
By identifying stable sets of predictive features, `Triglav` may lead to useful interpretations of the underlying data.

# Statement of need

As datasets grow in complexity and features, analysis becomes increasingly difficult due to noise and irrelevant features in the data.
To tackle this problem, feature selection methods are often used to reduce the complexity of the data while identifying the most relevant features given the task at hand [@CITATION-NEEDED]. 
With genomic and metagenomic datasets, this task has become increasingly important since generating models of the data and an understanding of how these models work directly improves our knowledge of complex systems such as disease process, viral transmission, and how ecosystems change over time [@CITATION-NEEDED]. 
Many feature selection approaches tend to remove redundant features, however, this may not necessarily be optimal for biological datasets. 
With complex modern genomic and metagenomic data, it is often important to identify the relevant functional changes occurring in microbiomes or networks of genes [@lowabdmicrobiome; @SingleCell]. 
However, the removal of redundant features could obfuscate important biological insights since the function of particular organisms or genes may not be included in downstream analyses. 
Therefore, `Triglav` was developed to implement an approach capable of identifying all relevant predictive features using explainable artificial intelligence to ensure that the selected features reflect actual differences and not the nuances the between different sets of training data, while allowing for a generalized way to measure differentially abundant species or genes in biological datasets. 
With the `Triglav` approach, differential abundance testing would no longer rely on the performance of a particular statistical model, but rather, the ability of a machine learning model to successfully classify a dataset.

![`Triglav` analysis identifies a stable set of features from a real-world dataset of 16S rRNA amplicon sequencing data from patients suffering from Crohn's Disease and healthy controls [@CD].
**A**, a comparison of `Triglav` performance against several common approaches.
**B**, SAGE importance scores from each of the selected features.
Many of the selected features were also detected in @CD.
**C**, a clustermap of the top features from each cluster visualizing differences in the microbiomes of healthy patients (blue) and those suffering from Crohn's Disease (red).
\label{fig:overview1}](Figure 1.png)

# Outline of the Triglav Algorithm

The core assumption behind `Triglav` is that clusters of impactful features sharing similar pattern of values across all samples should be discoverable. 
Since this is not an unreasonable assumption for biological datasets, different patterns, for example, in the abundance of gut bacterial species could exist between healthy controls and Crohn's Disease patients [@CD]. 
To take advantage of this observation, `Triglav` begins by clustering features [@2020SciPy-NMeth] and then randomly selecting one feature from each cluster. 
A set of shadow features (briefly explain what shadow features means? citation?) are then created by randomly sampling without replacement from the distribution of each selected feature [@JSSv036i11]. 
The shadow data is then combined with the original data and used to train a classification model [@JSSv036i11] and calculate Shapley values [@shapley1951notes; @SHAP1; @SHAP2]. 
This process is repeated to generate a distribution of Shapley values associated with each cluster of features and their shadow counterparts. 
A Wilcoxon signed-rank test is then used to determine if the distribution of Shapley values belonging to each cluster of real features is greater than the corresponding shadow cluster [@wilcoxon]. 
These steps are repeated multiple times, generating a binary matrix where '1' represents a cluster of features differing significantly from its shadow counterpart. 
An overview is provided in \autoref{fig:overview2}. 
A beta-binomial distribution is then used to determine if a feature should be selected, and a second beta-binomial distribution is used to determine if a feature should be rejected. 
Finally, the best feature from each cluster can be optionally discovered by calculating the SAGE importance score [@SAGE]. 
A visual overview is provided in \autoref{fig:overview3}.

![A high-level overview of the first half of the `Triglav` algorithm for producing a binary matrix specifying if the distribution of Shapley values associated with a cluster of features differs significantly from the distribution associated with the corresponding shadow cluster. 
False discovery rate corrections are applied at this step.
\label{fig:overview2}](Figure 2.png)

![A high-level overview of the second half of the `Triglav` algorithm, performing feature selection.
Two different beta-binomial distributions are used to determine if a feature is selected or rejected since they can model over-dispersion in zero-counts due to the random selection of features in the first-half of the algorithm (see \autoref{fig:overview2}).
For a feature to be selected, the number of times a significant difference was observed should fall within the critical region determined by the survival function of the first distribution or the cumulative distribution function of the second.
False discovery rate and Bonferroni corrections are applied at this step.
\label{fig:overview3}](Figure 3.png)

# Ongoing Research

Currently, this method is being used in projects to discover features capable of predicting virus host-origin and strain
specific bacterial markers at the National Centre for Foreign Animal Disease with the Canadian Food Inspection Agency. 
Additionally, we plan to integrate `Triglav` into an end-to-end software suite with our previously developed tools, `LANDMark` and `TreeOrdination` [@LANDMark; @TreeOrdination]. 
Together, this will form the basis of a modern toolset capable of investigating the organisms and genes associated with pathogenicity and environmental outcomes.

# Acknowledgements

We thank Dr. Terri M. Porter, Dr. Oksana Vernygora, and Hoang Hai Nguyen for their thoughtful review of the manuscript and code.
J.R. is supported by funds from the Food from Thought project as part of Canada First Research Excellence Fund and from CSSP-CFPADP-1278. 
M.H. received funding from the Government of Canada through Genome Canada and Ontario Genomics. G.B.G. is supported by a Natural 
Sciences and Engineering Research Council of Canada (NSERC) grant (RGPIN-2020-05733).

# References
