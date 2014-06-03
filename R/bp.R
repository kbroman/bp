#' MCMC with bipartite graph 
#'
#' MCMC with bipartite graph
#' @param out Root name for output files
#' @param whole Input file with whole node info
#' @param part Input file with part node info
#' @param edge Input file with edge info
#' @param alpha Parameter alpha (0 < alpha < beta < 1)
#' @param beta Parameter alpha (0 < alpha < beta < 1)
#' @param pi Parameter pi (0 < pi < 1)
#' @param nburn Number of burn-in generations
#' @param ngen Number of sample generations
#' @param sub Subsample rate for burn-in and sample files
#' @param penalty Penalty per illegal node to loglikelihood
#' @param initial Initial state (see Details)
#'
#' @details The \code{initial} argument can take one of three values:
#' \code{"inactive"} - all whole nodes inactive; \code{"random"} - all
#' whole nodes active with probability \code{pi}, no illegal nodes; or
#' \code{"high"} - all nodes with proportion of connected part nodes
#' with response equal to 1 above 0.4 are active, no illegal nodes.
#'
#' @export
#' @useDynLib bp
#' @keywords models
bp <-
function(out="run1",  whole="T2D.whole", part="T2D.part", edge="T2D.edge",
         alpha=0.05, beta=0.2, pi=0.01,
         nburn=10000, ngen=100000, sub=1000, penalty=2,
         initial=c("inactive", "random", "high"))
{
  initial <- match.arg(initial)

  stopifnot(alpha > 0, alpha < 1)
  stopifnot(beta > 0, beta < 1)
  stopifnot(pi > 0, pi < 1)
  stopifnot(nburn >=0)
  stopifnot(ngen >= 0)
  stopifnot(sub >= 0)
  
  z <- .C("R_bp",
          as.character(out),
          as.character(whole),
          as.character(part),
          as.character(edge),
          as.double(alpha),
          as.double(beta),
          as.double(pi),
          as.integer(nburn),
          as.integer(ngen),
          as.integer(sub),
          as.double(penalty),
          as.character(initial))
}
