#' Evaluating Biclustering Performance
#'
#' The function evaluates the performance of biclustering results by comparing them against true biclusters using various metrics.
#'
#' @param rows The indices of rows from the original dataset that have been grouped into biclusters by iSSVDR.
#' @param cols A nested list of arrays corresponding to the column indices of the variables included in each bicluster for each view (dataset).
#' @param true_rows The true indices of rows that define the biclusters in the original dataset. It is a list of arrays, each containing the row indices of a known true bicluster.
#' @param true_cols A nested list of arrays containing the true indices of the variables that define the biclusters in each view of the original dataset.
#' @param n The total number of samples (rows) in the original dataset. This is used to initialize arrays and to calculate the diagnostics. Need to append a letter L to the integer. Default: 200.
#' @param p The total number of variables (columns) in the original dataset. Like n, this is used in the initialization of arrays for the diagnostics. Need to append a letter L to the integer. Default: 1000.
#' @param D The number of views or datasets used in the integrative biclustering. Need to append a letter L to the integer. Default: 2.
#'
#' @details The function is designed to evaluate the performance of biclustering results, particularly focusing on the comparison between discovered biclusters and true biclusters. It calculates several diagnostic metrics, including recovery, relevance, F1 score, false positive rate, and false negative rate. Please refer to the main paper \href{https://doi.org/10.1177/09622802221122427}{Zhang, Weijie, et al. "Robust integrative biclustering for multi-view data." Statistical methods in medical research 31.11 (2022): 2201-2216.}  for more details.
#' @return A list with the following elements:
#' \item{Recovery}{ Measures how well the detected clusters match the true clusters. A higher value indicates better recovery of the true cluster structure.}
#' \item{Relevance}{ Similar to recovery, this metric assesses the relevance of the detected clusters to the true clusters. A higher value indicates that the detected clusters are more relevant to the true cluster structure.}
#' \item{F-Score}{ The F-Score combines the information of both recovery and relevance into a single score, with higher values indicating better clustering performance.}
#' \item{False Positives (FPs)}{ The average minimum proportion of false positive elements across all detected clusters. A lower value indicates fewer elements are wrongly included, which is desirable.}
#' \item{False Negatives (FNs)}{ The average minimum proportion of false negative elements across all detected clusters. A lower value indicates fewer elements are wrongly excluded, which is also desirable.}
#' @references
#' \href{https://doi.org/10.1177/09622802221122427}{Zhang, Weijie, et al. "Robust integrative biclustering for multi-view data." Statistical methods in medical research 31.11 (2022): 2201-2216.}
#'
#' \href{https://doi.org/10.1111/j.1467-9868.2010.00740.x}{Meinshausen, Nicolai, and Peter Bühlmann. "Stability selection." Journal of the Royal Statistical Society Series B: Statistical Methodology 72.4 (2010): 417-473.}
#'
#' \href{https://doi.org/10.1093/bioinformatics/btr322}{Sill, Martin, et al. "Robust biclustering by sparse singular value decomposition incorporating stability selection." Bioinformatics 27.15 (2011): 2089-2097.}
#'
#' @examples
#' library(iSSVD)
#' # Generate two views with four biclusters, number of samples 50, number of variables 1000.
#' mydata = generateData(myseed = 20L) # Uses default settings
#' X1 = mydata[[1]][[1]][[1]]
#' X2 = mydata[[1]][[1]][[2]]
#' Xdata1 = list(X1, X2)
#'
#' iSSVD_results = iSSVDR(X = Xdata1, myseed = 25L, standr = FALSE, vthr = 0.9, nbicluster = 4L)
#' sample_clusters = iSSVD_results[[3]][[3]]
#' variable_clusters = iSSVD_results[[3]][[4]]
#'
#' true_sample_clusters = mydata[[2]][[1]]
#' true_variable_clusters = mydata[[3]][[1]]
#'
#' diag = diagnostics(rows = sample_clusters, cols = variable_clusters, true_rows = true_sample_clusters, true_cols = true_variable_clusters,
#' n = 200L, p = 1000L, D = 2L)

#main iSSVD function
diagnostics=function(rows,cols,true_rows,true_cols,n=200L,p=1000L,D=2L){

  library(reticulate)
  #reticulate::use_python(python = Sys.which("python"), required = TRUE)
  reticulate::py_config()
  reticulate::source_python(system.file("python/issvd_functions2.py",
                                        package = "iSSVD"))

  if (!reticulate::py_available()) {
    stop("python not available")
  }

  #set defaults
  if(is.null(n)){
    n=200L
  }
  if(is.null(p)){
    p=1000L
  }
  if(is.null(D)){
    D=2L
  }

  # rows_minus1 <- lapply(rows, function(x) x - 1)
  # true_rows_minus1 <- lapply(true_rows, function(x) x - 1)
  # cols_minus1 <- lapply(cols, function(outerList) {
  #   lapply(outerList, function(innerArray) {
  #     innerArray - 1
  #   })
  # })
  # true_cols_minus1 <- lapply(true_cols, function(outerList) {
  #   lapply(outerList, function(innerArray) {
  #     innerArray - 1
  #   })
  # })

  myRows_minus1 <- lapply(rows, function(x) x - 1)
  myCols_minus1 <- lapply(cols, function(sublist) {
    # For each sublist, use sapply to iterate over the vectors and subtract 1
    lapply(sublist, function(vec) {
      vec - 1  # Subtract 1 from every element in the vector
    })
  })

  # To interger
  true_rows_int <- lapply(true_rows, function(x) as.integer(x))
  true_cols_int <- lapply(true_cols, function(outerList) {
    lapply(outerList, function(array) {
      as.integer(array)
    })
  })

  myRows_minus1_int <- lapply(myRows_minus1, function(x) as.integer(x))
  myCols_minus1_int <- lapply(myCols_minus1, function(outerList) {
    lapply(outerList, function(array) {
      as.integer(array)
    })
  })


  res = issvd_diagnostics2(rows=myRows_minus1_int,cols=myCols_minus1_int,true_rows=true_rows_int,true_cols=true_cols_int,
                           n=n,p=p,D=D)
  #shift sample and column indices by 1 since python indexing starts at 0

  # for (i in 1:length(res[[3]])){
  #   for (j in 1:length(res[[3]][[i]])){
  #     res[[3]][[i]][[j]]=res[[3]][[i]][[j]]+1
  #   }
  # }
  #
  # for (i in 1:length(res[[4]])){
  #   for (j in 1:length(res[[4]][[i]])){
  #     res[[4]][[i]][[j]]=res[[4]][[i]][[j]]+1
  #   }
  # }



  return(list(recovery = res[[1]], relevance = res[[2]],fscore = res[[3]],fps = res[[4]],fns = res[[5]]))
}



