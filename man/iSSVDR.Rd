% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/iSSVDR.R
\name{iSSVDR}
\alias{iSSVDR}
\title{Integrative Biclustering of Multi-view Data}
\usage{
iSSVDR(
  X = Xdata,
  myseed = 25L,
  standr = FALSE,
  pointwise = TRUE,
  steps = 100L,
  size = 0.5,
  vthr = 0.9,
  ssthr = c(0.6, 0.65),
  nbicluster = 4L,
  rows_nc = FALSE,
  cols_nc = FALSE,
  col_overlap = FALSE,
  row_overlap = FALSE,
  pceru = 0.1,
  pcerv = 0.1,
  merr = 1e-04,
  iters = 100L,
  assign_unclustered_samples = TRUE
)
}
\arguments{
\item{X}{Input data, expected to be a list of views or datasets that are integrated using the biclustering algorithm.}

\item{myseed}{Seed for random number generation.}

\item{standr}{A boolean flag for standardizing the data. Default: FALSE.}

\item{pointwise}{If TRUE, a fast pointwise control method will be performed for stability selection. Default: TRUE.}

\item{steps}{Number of subsamples used to perform stability selection. Default: 100.}

\item{size}{Size of the subsamples used to perform stability selection. Default: 0.5.}

\item{vthr}{Variance threshold for determining the initial estimate of the number of biclusters based on the cumulative sum of normalized singular values. Default: 0.9.}

\item{ssthr}{Range of the threshold for stability selection. Default: c(0.6, 0.65).}

\item{nbicluster}{A user specified number of biclusters to be detected. Default: 4.}

\item{rows_nc}{If TRUE allows for negative correlation of rows over columns. Default: FALSE.}

\item{cols_nc}{If TRUE allows for negative correlation of columns over rows. Default: FALSE.}

\item{col_overlap}{If TRUE allows for column overlaps among biclusters. Default: FALSE.}

\item{row_overlap}{If TRUE allows for row overlaps among biclusters. Default: FALSE.}

\item{pceru}{Per-comparison error rate to control the number of falsely selected coefficients in u. Default: 0.1.}

\item{pcerv}{Per-comparison error rate to control the number of falsely selected coefficients in v. Default: 0.1.}

\item{merr}{Minimum error threshold. Default: 0.0001.}

\item{iters}{Maximum iterations for detecting each bicluster. Default: 100.}

\item{assign_unclustered_samples}{If TRUE then automatically assigns unclustered samples to clusters based on the calculated probabilities indicating the likelihood of each unclustered sample belonging to a particular cluster. Default: TRUE.}
}
\value{
A list with the following elements:
\item{results}{ Stability selection results of left and right singular vectors;}
\item{N}{ Number of biclusters detected;}
\item{Sample_index}{ The indices of bicluster samples;}
\item{Variable_index}{ The indices of bicluster variables;}
\item{Interaction}{ The interactions run for each bicluster;}
\item{N_unclustered}{ Number of unclustered samples;}
\item{unclustered_index}{ Indices of the samples that are not clustered;}
\item{extract_data}{ The extracted data of each cluster and each view.}
}
\description{
The integrative sparse singular value decomposition (iSSVD) is based on sparse singular value decomposition for single-view data to data from multiple views. Theproposedalgorithmestimatestheprobabilityof samplesandvariables tobelongtoabicluster, finds stablebiclusters, andresults in interpretable row-column associations. Simulations and real data analyses show that iSSVD out performs several other single- and multi-view biclustering methods and is able to detect meaningful biclusters.
}
\details{
Suppose that data are available from \eqn{D} different views, and each view is arranged in an \eqn{n \times p^{(d)}} matrix \eqn{X^{(d)}}, where
the superscript \eqn{d} corresponds to the \eqn{d}-th view. For instance, for the same set of \eqn{n} individuals,
matrix \eqn{X^{(1)}} consists of RNA sequencing data and \eqn{X^{(2)}} consists of proteomics data, for \eqn{D = 2} views.
We wish to cluster the \eqn{p^{(d)}} variables, and the \eqn{n} subjects so that each subject subgroup is
associated with a specific variable subgroup of relevant variables. We consider estimating the left and right singular vectors and inferring the non-zero coefficients, sample clusters, and variable clusters. We subsample variables in each view \eqn{I} times without
replacement, while ensuring that each view contains the same set of samples. Please refer to the main paper \href{https://doi.org/10.1177/09622802221122427}{Zhang, Weijie, et al. "Robust integrative biclustering for multi-view data." Statistical methods in medical research 31.11 (2022): 2201-2216.} for more details. For this R manuscript, \eqn{X^{(d)}}, \eqn{X^{(1)}}, \eqn{X^{(2)}} and  \eqn{p^{(d)}} is not showing as what I want
}
\examples{
library(iSSVD)
# Generate two views with four biclusters, number of samples 50, number of variables 1000.
mydata = generateData(myseed = 20L) # Uses default settings
X1 = mydata[[1]][[1]][[1]]
X2 = mydata[[1]][[1]][[2]]

Xdata1 = list(X1, X2)

iSSVD_results = iSSVDR(X = Xdata1, myseed = 25L, standr = FALSE, vthr = 0.9, nbicluster = 4L)

sample_clusters = lapply(iSSVD_results[[3]][[3]], as.integer)
variable_clusters = lapply(iSSVD_results[[3]][[4]], lapply, as.integer)

# Visualize cluster
plotHeatMapR(X = Xdata1, Rows = sample_clusters, Cols = variable_clusters, D = 2L, nbicluster = 4L)
}
\references{
\href{https://doi.org/10.1177/09622802221122427}{Zhang, Weijie, et al. "Robust integrative biclustering for multi-view data." Statistical methods in medical research 31.11 (2022): 2201-2216.}

\href{https://doi.org/10.1111/j.1467-9868.2010.00740.x}{Meinshausen, Nicolai, and Peter Bühlmann. "Stability selection." Journal of the Royal Statistical Society Series B: Statistical Methodology 72.4 (2010): 417-473.}

\href{https://doi.org/10.1093/bioinformatics/btr322}{Sill, Martin, et al. "Robust biclustering by sparse singular value decomposition incorporating stability selection." Bioinformatics 27.15 (2011): 2089-2097.}
}
