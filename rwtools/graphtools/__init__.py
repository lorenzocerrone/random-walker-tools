from rwtools.graphtools.solvers import direct_solver, solve_cg, solve_cg_mg, solve_gpu

solvers = {"direct": direct_solver,
           "cg_mg": solve_cg_mg,
           "cg": solve_cg,
           "cuda": solve_gpu}