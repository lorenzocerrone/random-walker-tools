import warnings

import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse.linalg import cg
from scipy.sparse.linalg import spsolve

try:
    import pyamg
    pyamg_installed = True

except ImportError:
    import pyamg
    warnings.warn("Pyamg not installed, performance for big images will be drastically reduced."
                  " Reverting to direct solver.")
    pyamg_installed = False

try:
    from sksparse.cholmod import cholesky
    sksparse_installed = True

except ImportError:
    warnings.warn("sksparse. Reverting to direct solver.")
    sksparse_installed = False


def direct_solver(adj_csc: csc_matrix, b: csc_matrix) -> np.ndarray:
    """ Simple wrapper around scipy spsolve """
    return spsolve(adj_csc, b, use_umfpack=True)


def cholesky_solver(adj_csc: csc_matrix, b: csc_matrix) -> np.ndarray:
    """ Solve rw using cholesky decomposition """
    if not sksparse_installed:
        return direct_solver(adj_csc, b)

    adj_solver, pu = cholesky(adj_csc), np.empty_like(b)

    for i in range(b.shape[-1]):
        _pu = adj_solver.solve_A(b[:, i])
        pu[:, i] = _pu if type(_pu) == np.ndarray else _pu.toarray()
    return pu


def get_mg_preconditioner(adj_csc: csc_matrix):
    ml = pyamg.ruge_stuben_solver(adj_csc, coarse_solver='gauss_seidel')
    m_preconditioner = ml.aspreconditioner(cycle='V')
    return m_preconditioner


def solve_mg_cg(adj_csc: csc_matrix, b: csc_matrix, tol: float = 1.e-3, pre_conditioner: bool = True) -> np.ndarray:
    """
    Implementation follows the source code of skimage:
    https://github.com/scikit-image/scikit-image/blob/master/skimage/segmentation/random_walker_segmentation.py
    it solves the linear system of equations: Ax = b,
    by conjugate gradient and using the Ruge-Stuben solver as pre-conditioner.
    Parameters
    ----------
    adj_csc: Sparse csr matrix (NxN)
    b: Sparse array or array (NxM)
    tol: result tolerance
    pre_conditioner: if false no pre-conditioner is used

    returns x array (NxM)
    -------
    """
    adj_csc = csr_matrix(adj_csc)
    b = b.astype(np.float32) if type(b) == np.ndarray else b.todense().astype(np.float32)

    # pre-conditioner
    m_preconditioner = get_mg_preconditioner(adj_csc) if pre_conditioner and pyamg_installed else None

    pu = np.zeros_like(b)
    for i in range(b.shape[-1]):
        pu[:, i] = cg(csc_matrix, b[:, i], tol=tol, M=m_preconditioner)[0].astype(np.float32)
    return pu


def solve_cg(adj_csc: csc_matrix, b: csc_matrix, tol: float = 1.e-3):
    """
    Implementation follows the source code of skimage:
    https://github.com/scikit-image/scikit-image/blob/master/skimage/segmentation/random_walker_segmentation.py
    it solves the linear system of equations: Ax = b,
    by conjugate gradient
    Parameters
    ----------
    adj_csc: Sparse csr matrix (NxN)
    b: Sparse array or array (NxM)
    tol: result tolerance

    returns x array (NxM)
    -------
    """
    return solve_mg_cg(adj_csc, b, tol=tol, pre_conditioner=False)
