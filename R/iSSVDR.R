#' Integrative Biclustering of Multi-view Data
#'
#' The integrative sparse singular value decomposition (iSSVD) is based on sparse singular value decomposition for single-view data to data from multiple views. Theproposedalgorithmestimatestheprobabilityof samplesandvariables tobelongtoabicluster, finds stablebiclusters, andresults in interpretable row-column associations. Simulations and real data analyses show that iSSVD out performs several other single- and multi-view biclustering methods and is able to detect meaningful biclusters.
#'
#' @param X Input data, expected to be a list of views or datasets that are integrated using the biclustering algorithm.
#' @param myseed Seed for random number generation.
#' @param standr A boolean flag for standardizing the data. Default: FALSE.
#' @param pointwise If TRUE, a fast pointwise control method will be performed for stability selection. Default: TRUE.
#' @param steps Number of subsamples used to perform stability selection. Default: 100.
#' @param size Size of the subsamples used to perform stability selection. Default: 0.5.
#' @param vthr Variance threshold for determining the initial estimate of the number of biclusters based on the cumulative sum of normalized singular values. Default: 0.9.
#' @param ssthr Range of the threshold for stability selection. Default: c(0.6, 0.65).
#' @param nbicluster A user specified number of biclusters to be detected. Default: 4.
#' @param rows_nc If TRUE allows for negative correlation of rows over columns. Default: FALSE.
#' @param cols_nc If TRUE allows for negative correlation of columns over rows. Default: FALSE.
#' @param col_overlap If TRUE allows for column overlaps among biclusters. Default: FALSE.
#' @param row_overlap If TRUE allows for row overlaps among biclusters. Default: FALSE.
#' @param pceru Per-comparison error rate to control the number of falsely selected coefficients in u. Default: 0.1.
#' @param pcerv Per-comparison error rate to control the number of falsely selected coefficients in v. Default: 0.1.
#' @param merr Minimum error threshold. Default: 0.0001.
#' @param iters Maximum iterations for detecting each bicluster. Default: 100.
#' @param assign_unclustered_samples If TRUE then automatically assigns unclustered samples to clusters based on the calculated probabilities indicating the likelihood of each unclustered sample belonging to a particular cluster. Default: TRUE.
#'
#' @details Suppose that data are available from \eqn{D} different views, and each view is arranged in an \eqn{n \times p^{(d)}} matrix \eqn{X^{(d)}}, where
#' the superscript \eqn{d} corresponds to the \eqn{d}-th view. For instance, for the same set of \eqn{n} individuals,
#' matrix \eqn{X^{(1)}} consists of RNA sequencing data and \eqn{X^{(2)}} consists of proteomics data, for \eqn{D = 2} views.
#' We wish to cluster the \eqn{p^{(d)}} variables, and the \eqn{n} subjects so that each subject subgroup is
#' associated with a specific variable subgroup of relevant variables. We consider estimating the left and right singular vectors and inferring the non-zero coefficients, sample clusters, and variable clusters. We subsample variables in each view \eqn{I} times without
#' replacement, while ensuring that each view contains the same set of samples. Please refer to the main paper \href{https://doi.org/10.1177/09622802221122427}{Zhang, Weijie, et al. "Robust integrative biclustering for multi-view data." Statistical methods in medical research 31.11 (2022): 2201-2216.} for more details. For this R manuscript, \eqn{X^{(d)}}, \eqn{X^{(1)}}, \eqn{X^{(2)}} and  \eqn{p^{(d)}} is not showing as what I want
#' @return A list with the following elements:
#' \item{results}{ Stability selection results of left and right singular vectors;}
#' \item{N}{ Number of biclusters detected;}
#' \item{Sample_index}{ The indices of bicluster samples;}
#' \item{Variable_index}{ The indices of bicluster variables;}
#' \item{Interaction}{ The interactions run for each bicluster;}
#' \item{N_unclustered}{ Number of unclustered samples;}
#' \item{unclustered_index}{ Indices of the samples that are not clustered;}
#' \item{extract_data}{ The extracted data of each cluster and each view.}
#'
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
#' # Generate two views with four biclusters, number of samples 50, number of variables 1000.
#' mydata = generateData(myseed = 20L) # Uses default settings
#' X1 = mydata[[1]][[1]][[1]]
#' X2 = mydata[[1]][[1]][[2]]
#'
#' Xdata1 = list(X1, X2)
#'
#' iSSVD_results = iSSVDR(X = Xdata1, myseed = 25L, standr = FALSE, vthr = 0.9, nbicluster = 4L)
#'
#' sample_clusters = lapply(iSSVD_results[[3]][[3]], as.integer)
#' variable_clusters = lapply(iSSVD_results[[3]][[4]], lapply, as.integer)
#'
#' # Visualize cluster
#' plotHeatMapR(X = Xdata1, Rows = sample_clusters, Cols = variable_clusters, D = 2L, nbicluster = 4L)
#main iSSVD function
iSSVDR=function(X=Xdata, myseed=25L, standr=FALSE, pointwise=TRUE, steps=100L,size=0.5,vthr=0.9, ssthr=c(0.6,0.65),
               nbicluster=4L,rows_nc=FALSE,cols_nc=FALSE,col_overlap=FALSE,row_overlap=FALSE,
               pceru=0.1,pcerv=0.1,merr=0.0001,iters=100L,assign_unclustered_samples=TRUE){

  if(is.data.frame(X[[1]])){

    variable_names <- list()
    for(i in 1:length(X)){
      variable_names[[i]] <- colnames(X[[i]])
      X[[i]] <- as.matrix(X[[i]])
    }

    flag=1
  }else{
    flag=0
  }

  library(reticulate)
  #reticulate::use_python(python = Sys.which("python"), required = TRUE)
  reticulate::py_config()
  reticulate::source_python(system.file("python/issvd_functions2.py",
                                        package = "iSSVD"))

  if (!reticulate::py_available()) {
    stop("python not available")
  }

  #check inputs for training data
  dsizes=lapply(X, function(x) dim(x))
  n=dsizes[[1]][1]
  nsizes=lapply(X, function(x) dim(x)[1])

  myseed=as.integer(myseed)
  steps=as.integer(steps)
  nbicluster=as.integer(nbicluster)
  iters=as.integer(iters)

  if(all(nsizes!=nsizes[[1]])){
    stop('The datasets  have different number of observations')
  }


  #check data
  if (is.list(X)) {
    D = length(X)
    if(D==1){
      stop("There should be at least two datasets")
    }
  } else {
    stop("Input data should be a list")
  }


  #set defaults
  if(is.null(myseed)){
    myseed=25L
  }

  if(is.null(standr)){
    standr=FALSE
  }

  #standardize each view to have Frobeniums norm 1
  if(standr==TRUE){
   X=lapply(X, function(x) x/norm(x,type="F"))
  }

  if(is.null(pointwise)){
    pointwise=TRUE
  }

  if(is.null(steps)){
    steps=100L
  }

  if(is.null(size)){
    size=0.5
  }

  if(is.null(ssthr)){
    ssthr=c(0.6,0.65)
  }

  if(is.null(nbicluster)){
    nbicluster=4L
  }

  if(is.null(rows_nc)){
    rows_nc=FALSE
  }

  if(is.null(cols_nc)){
    cols_nc=FALSE
  }

  if(is.null(col_overlap)){
    col_overlap=FALSE
  }

  if(is.null(row_overlap)){
    row_overlap=FALSE
  }

  if(is.null(pceru)){
    pceru=0.1
  }

  if(is.null(pcerv)){
    pcerv=0.1
  }

  if(is.null(merr)){
    merr=0.0001
  }

  if(is.null(iters)){
    iters=100L
  }

  if(is.null(assign_unclustered_samples)){
    assign_unclustered_samples=TRUE
  }
  if(is.null(vthr)){
    vthr=0.9
  }

  res = issvd(X=X, myseed=myseed, standr=standr, pointwise=pointwise, steps=steps,size=size,vthr=vthr,ssthr=ssthr,
              nbicluster=nbicluster,rows_nc=rows_nc,cols_nc=cols_nc,col_overlap=col_overlap,row_overlap=row_overlap,
              pceru=pceru,pcerv=pcerv,merr=merr,iters=iters,assign_unclustered_samples=assign_unclustered_samples)
  #shift sample and column indices by 1 since python indexing starts at 0

  if ( length(res) >= 3 && is.list(res[[3]]) && length(res[[3]]) >= 1 ){
  for (i in 1:length(res[[3]])){
    for (j in 1:length(res[[3]][[i]])){
      res[[3]][[i]][[j]]=res[[3]][[i]][[j]]+1
    }
  }

  for (i in 1:length(res[[4]])){
    for (j in 1:length(res[[4]][[i]])){
      res[[4]][[i]][[j]]=res[[4]][[i]][[j]]+1
    }
  }
  }

  total_rows <- nrow(X[[1]])
  # Flatten the clusters_list to get all indices in the clusters
  indices_in_clusters <- unlist(res[[3]])
  # Find the indices that are not in any of the clusters
  indices_not_in_clusters <- setdiff(1:total_rows, indices_in_clusters)

  #res$Sample_index=lapply(res$Sample_index, function(x) x+1)
  #res$Variable_index=lapply(res$Variable_index, function(x) x+1)

  #return(res)
  N_unclustered = length(indices_not_in_clusters)
  unclustered_index = indices_not_in_clusters

  extract_data=list()

  if ( length(res) >= 3 && is.list(res[[3]]) && length(res[[3]]) >= 1 ){
  for (i in 1:length(X)){
    extract_data[[i]]=list()
    for (j in 1:length(res[[3]])){
      n_col=length(res[[4]][[j]][[i]])
      extract_data[[i]][[j]]=data.frame(matrix(ncol = n_col, nrow = 0))
      for (k in res[[3]][[j]]){
        extract_data[[i]][[j]]=rbind(extract_data[[i]][[j]],X[[i]][k,res[[4]][[j]][[i]]])
      }
      extract_data[[i]][[j]]<-t(extract_data[[i]][[j]])
      extract_data[[i]][[j]]<-t(extract_data[[i]][[j]])
    }
  }
  }


  ############################################
  # Function to retrieve variable names based on nested lists of indices
  replace_numbers_with_names <- function(data, var_names) {
    # Recursive function to traverse the list structure
    # Adding index parameter to keep track of the current vector's position within its parent list
    recursive_replace <- function(item, index) {
      if (is.list(item)) {
        # Apply function to each element in the list, passing the index along
        return(lapply(seq_along(item), function(i) recursive_replace(item[[i]], i)))
      } else if (is.numeric(item)) {
        # Replace each number with the corresponding variable name
        # Use the passed index to determine which list of names to use
        return(var_names[[index]][item])
      }
      return(item)
    }

    # Start the recursive replacement with a dummy index since the outermost list does not have corresponding variable names
    return(recursive_replace(data, NA))
  }

  if(flag==1){

  # Get the corresponding variable names
  VariableName_index <- replace_numbers_with_names(res[[4]], variable_names)
  # res[[length(res) + 1]] <- variable_names_selected
  before <- res[1:4]  # Elements before the new insertion
  after <- res[5:length(res)]  # Elements from the fifth position onwards

  # Create a named list for the insertion to preserve the name in the structure
  named_insertion <- list(VariableName_index = VariableName_index)

  # Concatenate the lists together with the new element in between
  res <- c(before, named_insertion, after)

  }
  ############################################



  return(list(N_unclustered = N_unclustered,
              unclustered_index = unclustered_index,
              results = res,
              extract_data = extract_data))
}



