#' Plot heatmap using generated and results from iSSVDR
#'
#' The plotHeatMapR function takes a data matrix, organizes it into biclusters, and creates and displays two side-by-side heatmaps to visualize the data's structure and relationships within two dimensions.
#'
#' @param X A list of matrices to be visualized. It's assumed that this list contains multiple views or data sources. The length of this list should be equal to the value of D, representing the number of views. Each matrix in the list corresponds to one view of the data.
#' @param Rows A list of indices that represent the sample clusters from iSSVDR.
#' @param Cols A list of indices that represent the variable clusters from iSSVDR.
#' @param D This parameter represents the number of views or data sources in the synthetic data. It's set to 2 by default, but you can change it to match the actual number of views in your data.
#' @param nbicluster The number of biclusters to visualize. This parameter specifies how many biclusters should be included in the heatmap visualization. It defaults to 4, but you can adjust it to visualize a different number of biclusters.
#'
#' @details Please refer to the main paper \href{https://doi.org/10.1177/09622802221122427}{Zhang, Weijie, et al. "Robust integrative biclustering for multi-view data." Statistical methods in medical research 31.11 (2022): 2201-2216.} for more details.
#' @return The result of this function is the display of two heatmaps side by side, each representing a different dimension of the data.
#' @references
#' \href{https://doi.org/10.1177/09622802221122427}{Zhang, Weijie, et al. "Robust integrative biclustering for multi-view data." Statistical methods in medical research 31.11 (2022): 2201-2216.}
#'
#' \href{https://doi.org/10.1111/j.1467-9868.2010.00740.x}{Meinshausen, Nicolai, and Peter Bühlmann. "Stability selection." Journal of the Royal Statistical Society Series B: Statistical Methodology 72.4 (2010): 417-473.}
#'
#' \href{https://doi.org/10.1093/bioinformatics/btr322}{Sill, Martin, et al. "Robust biclustering by sparse singular value decomposition incorporating stability selection." Bioinformatics 27.15 (2011): 2089-2097.}
#'
#'
#' @examples
#' # Generate data and perform integrative biclustering using iSSVDR()
#' mydata = generateData(myseed = 20L) # Uses default settings
#' X1 = mydata[[1]][[1]][[1]]
#' X2 = mydata[[1]][[1]][[2]]
#' Xdata1 = list(X1, X2)
#'
#' iSSVD_results = iSSVDR(X = Xdata1, myseed = 25L, standr = FALSE, vthr = 0.9, nbicluster = 4L)
#'
#' # Plot HeatMap using the results from iSSVDR
#' myRows = iSSVD_results[[3]][[3]]
#' myCols = iSSVD_results[[3]][[4]]
#' myRows <- lapply(myRows, as.integer)
#' myCols <- lapply(myCols, lapply, as.integer)
#'
#' plotHeatMapR(X = Xdata1, Rows = myRows, Cols = myCols, D = 2L, nbicluster = 4L)

plotHeatMapR <- function(X=Xdata, Rows=myRows, Cols=myCols, D=2L, nbicluster=4L){
  library(ggplot2)
  library(reshape2)
  library(gridExtra) # for arranging multiple plots

  plot_list <- list() # List to store ggplot objects

  # Assuming X is a list of data frames or matrices
  for (d in seq_len(D)) {
    cs <- list()
    col <- integer(0)
    for (i in seq_len(nbicluster)) {
      # Convert X[[d]] to a matrix if it's a data frame, ensuring column names are numeric
      if (is.data.frame(X[[d]])) {
        x_mat <- as.matrix(X[[d]])
        # colnames(x_mat) <- seq_len(ncol(x_mat))
      } else {
        x_mat <- X[[d]]
      }

      r1 <- x_mat[Rows[[i]], , drop = FALSE]
      col <- c(col, Cols[[i]][[d]])
      c1 <- r1[, col, drop = FALSE]
      c2 <- r1[, -col, drop = FALSE]
      c3 <- cbind(c1, c2)
      cs[[i]] <- c3
    }

    # Melt the data frame for the current view
    df_long <- melt(do.call(rbind, cs))

    # Create the ggplot object for the current view
    p <- ggplot(df_long, aes(x = as.numeric(Var2), y = as.numeric(Var1), fill = value)) +
      geom_tile() +
      scale_fill_gradient(low = "black", high = "red") +
      scale_y_reverse() +
      labs(x = "Cols", y = "Rows") +
      theme_minimal() +
      theme(axis.text.x = element_text(angle = 90, hjust = 1),
            axis.text.y = element_text(angle = 0, vjust = 0.5))

    plot_list[[d]] <- p # Add the plot to the list
  }

  # Arrange and display both plots side by side
  do.call(grid.arrange, c(plot_list, ncol=2))
}
