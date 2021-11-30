from rwtools.graphtools.basic_solvers import direct_solver, solve_cg, solve_mg_cg, cholesky_solver
from rwtools.graphtools.cuda_solvers import solve_gpu, solve_gpu_cg
from rwtools.graphtools.numba_solvers import solve_numba_cg
from enum import Enum


class ImplementedSolvers(Enum):
    # direct solvers
    direct = direct_solver
    cholesky = cholesky_solver
    cuda = solve_gpu
    # cg solves
    mg_cg = solve_mg_cg
    cg = solve_cg
    cuda_cg = solve_gpu_cg
    numba_cg = solve_numba_cg


class Solver:
    def __init__(self, mode='direct', kwargs=None):
        self.solver = ImplementedSolvers.__getattr__(mode)
        self.kwargs = kwargs

    def __call__(self, adj_mat, b):
        return self.solver(adj_mat, b, **self.kwargs)
