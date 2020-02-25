from rwtools.graphtools.solvers import direct_solver, solve_cg, solve_cg_mg, solve_gpu, cholesky_solver

solvers = {"direct": direct_solver,
           "cholesky": cholesky_solver,
           "cg_mg": solve_cg_mg,
           "cg": solve_cg,
           "cuda": solve_gpu}