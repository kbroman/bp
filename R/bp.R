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
          as.character(run1),
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
