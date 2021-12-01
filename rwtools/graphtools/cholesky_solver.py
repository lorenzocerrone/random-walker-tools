import warnings

import numpy as np
from scipy.sparse import csc_matrix

try:
    from sksparse.cholmod import cholesky
    sksparse_installed = True

except ImportError:
    warnings.warn("sksparse. Reverting to direct solver.")
    sksparse_installed = False


def cholesky_solver(adj_csc: csc_matrix, b: csc_matrix) -> np.ndarray:
    """ Solve rw using cholesky decomposition """
    adj_solver, pu = cholesky(adj_csc), np.empty_like(b)

    for i in range(b.shape[1]):
        _pu = adj_solver.solve_A(b[:, i])
        pu[:, i] = _pu if type(_pu) == np.ndarray else _pu.toarray()
    return pu
