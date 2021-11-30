from rwtools.graphtools.basic_solvers import direct_solver, solve_cg, solve_cg_mg, cholesky_solver
from rwtools.graphtools.cuda_solvers import solve_gpu, solve_gpu_cg
from rwtools.graphtools.numba_solvers import solve_numba_cg

solvers = {"direct": direct_solver,
           "cholesky": cholesky_solver,
           "cg_mg": solve_cg_mg,
           "cg": solve_cg,
           "cuda": solve_gpu,
           "cuda_cg": solve_gpu_cg,
           "numba_cg": solve_numba_cg,
           }

