from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import cg
from scipy.sparse import csr_matrix
import numpy as np
import warnings

try:
    import pyamg
    use_direct_solver = False

except ImportError:
    warnings.warn("Pyamg not installed, performance for big images will be drastically reduced.")
    use_direct_solver = True


def direct_solver(A, b):
    """ Simple wrapper around scipy spsolve """
    return spsolve(A, b, use_umfpack=True)


def solve_cg_mg(A, b, tol=1e-4, preconditioner=True):
    """
    Implementation follows the source code of skimage:
    https://github.com/scikit-image/scikit-image/blob/master/skimage/segmentation/random_walker_segmentation.py
    it solves the linear system of equations: Ax = b,
    by conjugate gradient and using the Ruge Stuben solver as pre-conditioner
    Parameters
    ----------
    A: Sparse csr matrix (NxN)
    b: Sparse array or array (NxM)
    tol: result tollerance
    preconditioner: if false no pre-conditioner is used

    returns x array (NxM)
    -------

    """
    if use_direct_solver:
        return direct_solver(A, b)

    pu = []
    A = csr_matrix(A)

    # The actual cast will be performed slice by slice to reduce memory footprint
    check_type = True if type(b) == np.ndarray else False

    # pre-conditioner
    if preconditioner:
        ml = pyamg.ruge_stuben_solver(A, coarse_solver='gauss_seidel')
        M = ml.aspreconditioner(cycle='V')
    else:
        M = None

    for i in range(b.shape[-1]):
        _b = b[:, i].astype(np.float32) if check_type else b[:, i].todense().astype(np.float32)
        _pu = cg(A, _b, tol=tol, M=M)[0].astype(np.float32)
        pu.append(_pu)

    return np.array(pu, dtype=np.float32).T
