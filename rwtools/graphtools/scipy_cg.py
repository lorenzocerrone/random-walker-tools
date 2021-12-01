import warnings

import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse.linalg import cg

try:
    import pyamg
    pyamg_installed = True

except ImportError:
    warnings.warn("Pyamg not installed, performance for big images will be drastically reduced."
                  " Reverting to direct solver.")
    pyamg_installed = False


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

    # pre-conditioner
    m_preconditioner = get_mg_preconditioner(adj_csc) if pre_conditioner and pyamg_installed else None
    pu = np.zeros_like(b)
    for i in range(b.shape[-1]):
        pu[:, i] = cg(adj_csc, b[:, i], tol=tol, M=m_preconditioner)[0].astype(np.float32)
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
