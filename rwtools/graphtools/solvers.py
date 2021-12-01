from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
import numpy as np


def direct_solver(adj_csc: csc_matrix, b: csc_matrix) -> np.ndarray:
    """ Simple wrapper around scipy spsolve """
    return spsolve(adj_csc, b, use_umfpack=True)


class Solver:
    def __init__(self):
        self.kwargs = dict()

    def __call__(self, adj_mat, b, mode='direct'):
        _solver = getattr(self, mode)()
        return _solver(adj_mat, b, **self.kwargs)

    @staticmethod
    def direct():
        return direct_solver

    @staticmethod
    def cholesky():
        from rwtools.graphtools.cholesky_solver import cholesky_solver
        return cholesky_solver

    @staticmethod
    def cuda():
        from rwtools.graphtools.cuda_solvers import solve_gpu
        return solve_gpu

    @staticmethod
    def cg():
        from rwtools.graphtools.scipy_cg import solve_cg
        return solve_cg

    @staticmethod
    def mg_cg():
        from rwtools.graphtools.scipy_cg import solve_mg_cg
        return solve_mg_cg

    @staticmethod
    def numba_cg():
        from rwtools.graphtools.numba_solvers import solve_numba_cg
        return solve_numba_cg

    @staticmethod
    def cuda_cg():
        from rwtools.graphtools.cuda_solvers import solve_gpu_cg
        return solve_gpu_cg


solver = Solver()
