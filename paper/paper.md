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
identified by competing methods [@Stability]. This is important since identifying stable set of predictive features 
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
between different sets of training data.

# Outline of the Triglav Algorithm

The core assumption behind `Triglav` is that clusters of impactful features sharing similar pattern of values across all samples 
should be discoverable. This is not an unreasonable assumption in biological datasets. For example, different patterns in the 
abundance of gut bacteria could exist between healthy controls and Crohn's Disease patients [@CD]. To take advantage of this 
observation, `Triglav` begins by clustering features [@2020SciPy-NMeth]. Then the algorithm randomly selects one feature from each 
cluster. A set of shadow features are then created by randomly sampling without replacement from the distribution of each selected 
feature [@JSSv036i11]. The shadow data is then combined with the original data and is used to train a classification model [@JSSv036i11]. 
Shapley values are then calculated [@shapley1951notes; @SHAP1; @SHAP2]. This process is repeated to generate a distribution of Shapley 
values associated with each cluster of features and their shadow counterparts. A Wilcoxon signed-rank test is then used to determine if 
the distribution of Shapley values belonging to each cluster of real features is greater than the corresponding shadow cluster [@wilcoxon]. 
These steps are repeated multiple times, generating a binary matrix where '1' represents a cluster of features differing significantly 
from its shadow counterpart. An overview is provided in \autoref{fig:overview}. A beta-binomial distribution is then used to determine if 
a feature is to be selected. A second beta-binomial distribution is also used to determine when a feature is to be rejected. Finally, 
the best feature from each cluster can be optionally discovered by calculating the SAGE importance score [@SAGE]. This step is optional. 
A visual overview is provided in \autoref{fig:overview2}.

# Ongoing Research

Currently, this method is being used in projects to discover features capable of predicting host-origin of viral samples and strain
specific bacterial markers at the National Centre for Foreign Animal Disease with the Canadian Food Inspection Agency. In addition 
to this work, we hope to integrate `Triglav` into an end-to-end suite with our previously developed tools, `LANDMark` and 
`TreeOrdination` [@LANDMark; @TreeOrdination]. Together, this will form the basis of a modern toolset capable of investigating
the organisms and genes associated with pathogenicity and environmental outcomes.

# Acknowledgements

We thank Dr. Terri M. Porter, Dr. Oksana Vernygora, and Hoang Hai Nguyen for their thoughtful review of the manuscript and code.
J.R. is supported by funds from the Food from Thought project as part of Canada First Research Excellence Fund and from CSSP-CFPADP-1278. 
M.H. received funding from the Government of Canada through Genome Canada and Ontario Genomics. G.B.G. is supported by a Natural 
Sciences and Engineering Research Council of Canada (NSERC) grant (RGPIN-2020-05733).

# Figures

![A high-level overview of the first half of the `Triglav` algorithm. The output of this part of the algorithm is a binary matrix 
specifying if the distribution of Shapley values associated with a cluster of features differs significantly from the distribution 
associated with the corresponding shadow cluster. False discovery rate corrections are applied at this step.\label{fig:overview}](Figure 1.png)

![A high-level overview of the first half of the `Triglav` algorithm. The output of this part of the algorithm is a binary matrix 
specifying if the distribution of Shapley values associated with a cluster of features differs significantly from the distribution 
associated with the corresponding shadow cluster. False discovery rate corrections are applied at this step.\label{fig:overview}](Figure 2.png)

![A high-level overview of the second half of the `Triglav` algorithm. The output of this part of the algorithm is a list of 
selected features. Two different beta-binomial distributions are used to determine if a feature is selected or rejected. These 
distributions are used by `Triglav` since they can model over-dispersion in zero-counts due to the random selection of features in 
the first-half of the algorithm. For a feature to be selected, the number of times a significant difference was observed should 
fall within the critical region determined by the survival function of the first distribution or the cumulative distribution 
function of the second. A false-discovery rate and Bonferroni correction are applied at this step.\label{fig:overview2}](Figure 3.png)

# References