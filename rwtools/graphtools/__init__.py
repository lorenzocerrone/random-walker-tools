from rwtools.graphtools.solvers import direct_solver, solve_cg, solve_cg_mg,\
    solve_gpu, cholesky_solver, solve_numba_cg, solve_gpu_cg

solvers = {"direct": direct_solver,
           "cholesky": cholesky_solver,
           "cg_mg": solve_cg_mg,
           "cg": solve_cg,
           "cuda": solve_gpu,
           "cuda_cg": solve_gpu_cg,
           "numba_cg": solve_numba_cg,
           }

