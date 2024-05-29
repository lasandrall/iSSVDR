#' Generate simulated data for a given number of clusters
#'
#' The function generate data matrices with the truth of underlying biclusters that can be used for dianostics.
#'
#' @param myseed An integer to set a seed. Need to append a letter L to the integer, for example 25L. It's set to a default value of 25.
#' @param n An even integer for the number of rows in the generated data. It's set to a default value of 200.
#' @param p An integer for the number of the number of columns in the generated data. It's set to a default value of 1000.
#' @param D The number of views or data sources. In multi-view matrix factorization, you often have data from multiple sources or views. D controls how many such views are generated. It's set to a default value of 2.
#' @param rowsize An integer for the number of rows in each submatrix. It's set to a default value of 50.
#' @param colsize An integer for the number of columns in each submatrix. It's set to a default value of 100.
#' @param numbers An integer for the number of sets of data matrices to be generated. It's set to a default value of 1.
#' @param sigma A number for the standard deviation of the random noise added to the data. It's set to a default value of 0.1.
#' @param nbicluster The number of clusters in the row and column spaces for each synthetic dataset. It's set to a default value of 4.
#' @param orthonm A boolean flag. If TRUE, The left singular vector and right singular vector are orthogonalized. The default is FALSE.
#'
#' @details Please refer to the main paper \href{https://doi.org/10.1177/09622802221122427}{Zhang, Weijie, et al. "Robust integrative biclustering for multi-view data." Statistical methods in medical research 31.11 (2022): 2201-2216.}  for more details.
#' @return The function will return a list of synthetic data matrices. The list has a structure that reflects the different samples and factors in the generated data.
#' @references
#' \href{https://doi.org/10.1177/09622802221122427}{Zhang, Weijie, et al. "Robust integrative biclustering for multi-view data." Statistical methods in medical research 31.11 (2022): 2201-2216.}
#'
#' \href{https://doi.org/10.1111/j.1467-9868.2010.00740.x}{Meinshausen, Nicolai, and Peter Bühlmann. "Stability selection." Journal of the Royal Statistical Society Series B: Statistical Methodology 72.4 (2010): 417-473.}
#'
#' \href{https://doi.org/10.1093/bioinformatics/btr322}{Sill, Martin, et al. "Robust biclustering by sparse singular value decomposition incorporating stability selection." Bioinformatics 27.15 (2011): 2089-2097.}
#'
#'
#' @examples
#' # Generate the data list
#' mydata = generateData(myseed = 25L, n = 200L, p = 1000L, D = 2L, rowsize = 50L, colsize = 100L,
#'                       numbers = 1L, sigma = 0.1, nbicluster = 4L, orthonm = FALSE)
#'
#' # Select two data X1, X2 from the generated data list
#' X1 = mydata[[1]][[1]][[1]]
#' X2 = mydata[[1]][[1]][[2]]
#'
#' # Combine two data into a list
#' Xdata1 = list(X1, X2)


# generateData=function(myseed=25L,n=200L,p=1000L,D=2L,rowsize=50L, colsize=100L,
#                       numbers=1L,sigma=0.1,nbicluster=4L, orthonm=FALSE){
generateData=function(myseed=25L,n=200L,p=1000L,D=2L,rowsize=50L, colsize=100L,
                      numbers=1L,sigma=0.1,nbicluster=4L, orthonm=FALSE){
  library(reticulate)
  #reticulate::use_python(python = Sys.which("python"), required = TRUE)
  reticulate::py_config()
  reticulate::source_python(system.file("python/issvd_functions.py",
                                        package = "iSSVD"))

  if (!reticulate::py_available()) {
    stop("python not available")
  }


  #set defaults
  if(is.null(myseed)){
    myseed=12345L
  }

  if(is.null(n)){
    n=200L
  }

  if(is.null(p)){
    p=1000L
  }

  if(is.null(D)){
    D=2L
  }

  if(is.null(rowsize)){
    rowsize=50L
  }

  if(is.null(colsize)){
    colsize=100L
  }

  if(is.null(numbers)){
    numbers=1L
  }

  if(is.null(sigma)){
    sigma=0.1
  }

  if(is.null(nbicluster)){
    nbicluster=4L
  }

  if(is.null(orthonm)){
    orthonm=FALSE
  }

  mydata=gen_sim_vec(myseed=myseed,n=n,p=p,D=D,rowsize=rowsize,
                     colsize=colsize, numbers=numbers, sigma=sigma, nbicluster=nbicluster, orthonm=orthonm)

  return(mydata)

}
