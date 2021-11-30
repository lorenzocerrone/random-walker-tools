import warnings

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import cg
from scipy.sparse.linalg import spsolve

try:
    import pyamg
    use_direct_solver_mg = False

except ImportError:
    warnings.warn("Pyamg not installed, performance for big images will be drastically reduced."
                  " Reverting to direct solver.")
    use_direct_solver_mg = True

try:
    from sksparse.cholmod import cholesky
    use_cholesky = False

except ImportError:
    warnings.warn("sksparse. Reverting to direct solver.")
    use_cholesky = True


def direct_solver(A, b):
    """ Simple wrapper around scipy spsolve """
    return spsolve(A, b, use_umfpack=True)


def cholesky_solver(A, b):
    """ Solve rw using cholesky decomposition """
    if use_cholesky:
        return direct_solver(A, b)

    A_solver, x = cholesky(A), np.empty_like(b)

    for i in range(b.shape[-1]):
        _x = A_solver.solve_A(b[:, i])
        x[:, i] = np.array(_x, dtype=np.float32) if type(_x) == np.ndarray else np.array(_x.toarray(), dtype=np.float32)
    return x


def solve_mg_cg(A, b, tol=1.e-3, pre_conditioner=True):
    """
    Implementation follows the source code of skimage:
    https://github.com/scikit-image/scikit-image/blob/master/skimage/segmentation/random_walker_segmentation.py
    it solves the linear system of equations: Ax = b,
    by conjugate gradient and using the Ruge Stuben solver as pre-conditioner.
    Parameters
    ----------
    A: Sparse csr matrix (NxN)
    b: Sparse array or array (NxM)
    tol: result tolerance
    pre_conditioner: if false no pre-conditioner is used

    returns x array (NxM)
    -------
    """
    pu = []
    A = csr_matrix(A)

    # The actual cast will be performed slice by slice to reduce memory footprint
    check_type = True if type(b) == np.ndarray else False

    # pre-conditioner
    if pre_conditioner and not use_direct_solver_mg:
        M = mg_preconditioner(A)
    else:
        M = None

    _pu_sum = np.ones(b.shape[0], dtype=np.float32)
    for i in range(b.shape[-1] - 1):
        _b = b[:, i].astype(np.float32) if check_type else b[:, i].todense().astype(np.float32)
        _pu = cg(A, _b, tol=tol, M=M)[0].astype(np.float32)
        _pu_sum -= _pu
        pu.append(_pu)

    pu.append(_pu_sum)
    return np.array(pu, dtype=np.float32).T


def mg_preconditioner(A):
    ml = pyamg.ruge_stuben_solver(A, coarse_solver='gauss_seidel')
    M = ml.aspreconditioner(cycle='V')
    return M


def solve_cg(A, b, tol=1.e-3):
    """
    Implementation follows the source code of skimage:
    https://github.com/scikit-image/scikit-image/blob/master/skimage/segmentation/random_walker_segmentation.py
    it solves the linear system of equations: Ax = b,
    by conjugate gradient
    Parameters
    ----------
    A: Sparse csr matrix (NxN)
    b: Sparse array or array (NxM)
    tol: result tolerance

    returns x array (NxM)
    -------
    """
    return solve_mg_cg(A, b, tol=tol, pre_conditioner=None)
