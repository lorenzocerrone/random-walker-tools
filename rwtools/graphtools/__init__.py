from rwtools.graphtools.solvers import direct_solver, solve_cg, solve_cg_mg,\
    solve_gpu, cholesky_solver, mp_cg, mp_cg_ichol, sp_cg, sp_cg_ichol

solvers = {"direct": direct_solver,
           "cholesky": cholesky_solver,
           "cg_mg": solve_cg_mg,
           "cg": solve_cg,
           "cuda": solve_gpu,
           "mp_cg": mp_cg,
           "mp_cg_ichol": mp_cg_ichol,
           "sp_cg": sp_cg,
           "sp_cg_ichol": sp_cg_ichol
           }
