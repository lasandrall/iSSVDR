#' Make summary table for clinical data
#'
#' The function allows the user to make a summary table based on clinical data and sample clusters get from iSSVDR() function.
#'
#' @param data Input clinical data. Can use data("clinical_data") to use example data that contains 120 rows and 4 variables: "Age", "Sex", "crp", "ddimer".
#' @param res Input results directly generated from the iSSVDR() function.
#' @param ... Any variables you want to be shown in the summary table.
#'
#' @details summary_table returns a summary table showing you the summary statistics (N, Mean, SD) and p-value from Kruskal-Wallis rank sum test (for continuous variables)/ Pearson’s Chi-squared test (for categorical variables). This function depends on the gtsummary package.
#' @return A summary table that displays and compares summary statistics across sample clusters, and give p-values.
#' @references
#' \href{https://doi.org/10.1177/09622802221122427}{Zhang, Weijie, et al. "Robust integrative biclustering for multi-view data." Statistical methods in medical research 31.11 (2022): 2201-2216.}
#'
#' \href{https://doi.org/10.1111/j.1467-9868.2010.00740.x}{Meinshausen, Nicolai, and Peter Bühlmann. "Stability selection." Journal of the Royal Statistical Society Series B: Statistical Methodology 72.4 (2010): 417-473.}
#'
#' \href{https://doi.org/10.1093/bioinformatics/btr322}{Sill, Martin, et al. "Robust biclustering by sparse singular value decomposition incorporating stability selection." Bioinformatics 27.15 (2011): 2089-2097.}
#'
#'
#' @examples
#' library(iSSVD)
#' # Call example data
#' data("clinical_data") # Clinical data has no association with clusters, this is just for demonstrations.
#'
#' # Generate three views with four biclusters, number of samples 120, number of variables 1000.
#' mydata = generateData(n = 120L, p = 1000L, D = 3L, rowsize = 30L, colsize = 100L,
#'                       numbers = 1L, sigma = 0.1, nbicluster = 4L, orthonm = FALSE)
#' Xdata1 <- list()
#' for (i in 1:3) {
#'   Xdata1[[i]] <- mydata[[1]][[1]][[i]]
#' }
#'
#' res4 = iSSVDR(X = Xdata1, standr = FALSE, pointwise = TRUE, steps = 100, size = 0.5,
#'               vthr = 0.9, ssthr = c(0.6, 0.65), nbicluster = 4, rows_nc = FALSE, cols_nc = FALSE,
#'               col_overlap = FALSE, row_overlap = FALSE, pceru = 0.8, pcerv = 0.8, merr = 0.0001,
#'               iters = 100, assign_unclustered_samples = FALSE)
#'
#' # Generate Table
#' summary_table(clinical_data, res4, Age, Sex, crp, ddimer)

summary_table <- function(data, res, ...){
  library(gtsummary)
  library(dplyr)

  library(rlang)
  library(purrr)
  data1 <- data
  var_quosures <- enquos(...)

  # data1 <- data1 %>%
  #   mutate(across(all_of(names(select(data1, !!!var_quosures))), as.numeric))

  safe_numeric_conversion <- function(x) {
    num_x <- as.numeric(x)
    if (all(is.na(num_x), !is.na(x))) {
      return(x)
    } else {
      return(num_x)
    }
  }

  # Update the dataframe
  data1 <- data1 %>%
    mutate(across(all_of(names(select(data1, !!!var_quosures))), safe_numeric_conversion))


  bicluster_indices <- res[[3]][[3]]
  data1$Bicluster <- NA
  for (i in seq_along(bicluster_indices)) {
    bicluster_label <- paste("Bicluster", i)
    data1$Bicluster[bicluster_indices[[i]]] <- bicluster_label
  }

  summary <- data1 %>%
    tbl_summary(
      include = c(!!!var_quosures, "Bicluster"),
      by = Bicluster,
      statistic = list(all_continuous() ~ "{mean} ({sd})", all_categorical() ~ "{n} ({p}%)"),
      digits = all_continuous() ~ c(2, 2),
      missing = "no") %>%
    add_n()%>%
    add_p()
  return(summary)

}
