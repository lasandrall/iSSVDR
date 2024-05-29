#' Create a Python virtual environment with necessary packages
#'
#' This function is used to create a Python virtual environment (called "iSSVD") and installs
#' necessary packages such as pandas, numpy, and scikit-learn to the environment. Once created, the
#' environment will automatically be used upon loading the package.
#'
#' @details If there is an error installing the Python packages, try restarting your computer and running R/RStudio as administrator.
#' If the virtual environment is not created, the user's default system Python installation will be used.
#' In this case, the user will need to have the following packages in their main local Python installation:
#' \itemize{
#'   \item pandas
#'   \item matplotlib
#'   \item seaborn
#'   \item scikit-learn
#'   \item numpy
#' }
#' Alternatively, the user can use their own virtual environment with reticulate by activating it with
#' reticulate::use_virtualenv() or a similar function prior to loading RandMVLearn.
#'
#' @examples
#' # Create Python virtual environment "iSSVD"
#' createVirtualenv()

createVirtualenv=function(){

  env_name="iSSVD_env"
  new_env = identical(env_name, "iSSVD_env")

  if(new_env && reticulate::virtualenv_exists(envname=env_name) == TRUE){
    reticulate::virtualenv_remove(env_name)
  }

  package_req <- c( "matplotlib", "seaborn", "scikit-learn", "numpy", "pandas")

  reticulate::virtualenv_create(env_name, packages = package_req)

  { cat(paste0("\033[0;", 32, "m","Virtual environment installed, please restart your R session and reload the package.","\033[0m","\n"))}
}
