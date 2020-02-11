import warnings

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import cg
from scipy.sparse.linalg import spsolve

use_direct_solver_mg = False
use_direct_solver_cupy = False

try:
    import pyamg
except ImportError:
    warnings.warn("Pyamg not installed, performance for big images will be drastically reduced."
                  " Reverting to direct solver.")
    use_direct_solver_mg = True


try:
    from cupy.sparse import csc_matrix
    from cupy.sparse.linalg import lsqr
    import cupy as cp
    from cupy.cuda import cusolver
    from cupy.cuda import device
    from cupy.linalg import util
    import cupyx.scipy.sparse
except ImportError:
    warnings.warn("cupy not installed, GPU solver not available. Reverting to direct solver.")
    use_direct_solver_mg = True


def direct_solver(A, b):
    """ Simple wrapper around scipy spsolve """
    return spsolve(A, b, use_umfpack=True)


def solve_cg_mg(A, b, tol=1e-4, preconditioner=True):
    """
    Implementation follows the source code of skimage:
    https://github.com/scikit-image/scikit-image/blob/master/skimage/segmentation/random_walker_segmentation.py
    it solves the linear system of equations: Ax = b,
    by conjugate gradient and using the Ruge Stuben solver as pre-conditioner.
    Parameters
    ----------
    A: Sparse csr matrix (NxN)
    b: Sparse array or array (NxM)
    tol: result tollerance
    preconditioner: if false no pre-conditioner is used

    return x array (NxM)
    returns x array (NxM)
    -------

    """
    if use_direct_solver_mg:
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

    _pu_sum = np.ones(b.shape[0], dtype=np.float32)
    for i in range(b.shape[-1] - 1):
        _b = b[:, i].astype(np.float32) if check_type else b[:, i].todense().astype(np.float32)
        _pu = cg(A, _b, tol=tol, M=M)[0].astype(np.float32)
        _pu_sum -= _pu
        pu.append(_pu)

    pu.append(_pu_sum)
    return np.array(pu, dtype=np.float32).T


def solve_cg(A, b, tol=1e-4):
    """
    Implementation follows the source code of skimage:
    https://github.com/scikit-image/scikit-image/blob/master/skimage/segmentation/random_walker_segmentation.py
    it solves the linear system of equations: Ax = b,
    by conjugate gradient
    Parameters
    ----------
    A: Sparse csr matrix (NxN)
    b: Sparse array or array (NxM)
    tol: result tollerance

    return x array (NxM)
    returns x array (NxM)
    -------

    """
    return solve_cg_mg(A, b, tol=tol, preconditioner=None)



def solve_gpu(A, b):
    if use_direct_solver_cupy:
        return direct_solver(A, b)

    # The actual cast will be performed slice by slice to reduce memory footprint
    b = b.astype(np.float32) if type(b) == np.ndarray else b.todense().astype(np.float32)
    b_gpu = cp.asarray(np.array(b))

    cp_A_data = cp.asarray(A.data.ravel().astype(np.float32))
    cp_A_incices = cp.asarray(A.indices.ravel())
    cp_A_indptr = cp.asarray(A.indptr.ravel())

    A_gpu = csc_matrix((cp_A_data, cp_A_incices, cp_A_indptr), shape=A.shape)

    pu = []
    for i in range(b.shape[-1]):
        _pu = chol(A_gpu, b_gpu[:, i])[0]
        pu.append(cp.asnumpy(_pu))

    del A_gpu, b_gpu
    return np.array(pu, dtype=np.float32).T


def chol(A, b):
    """Solves linear system with QR decomposition.
    Find the solution to a large, sparse, linear system of equations.
    The function solves ``Ax = b``. Given two-dimensional matrix ``A`` is
    decomposed into ``Q * R``.
    Args:
        A (cupy.ndarray or cupyx.scipy.sparse.csr_matrix): The input matrix
            with dimension ``(N, N)``
        b (cupy.ndarray): Right-hand side vector.
    Returns:
        tuple:
            Its length must be ten. It has same type elements
            as SciPy. Only the first element, the solution vector ``x``, is
            available and other elements are expressed as ``None`` because
            the implementation of cuSOLVER is different from the one of SciPy.
            You can easily calculate the fourth element by ``norm(b - Ax)``
            and the ninth element by ``norm(x)``.
    .. seealso:: :func:`scipy.sparse.linalg.lsqr`
    """

    if not cupyx.scipy.sparse.isspmatrix_csr(A):
        A = cupyx.scipy.sparse.csr_matrix(A)
    util._assert_nd_squareness(A)
    util._assert_cupy_array(b)
    m = A.shape[0]
    if b.ndim != 1 or len(b) != m:
        raise ValueError('b must be 1-d array whose size is same as A')

    # Cast to float32 or float64
    if A.dtype == 'f' or A.dtype == 'd':
        dtype = A.dtype
    else:
        dtype = np.promote_types(A.dtype, 'f')

    handle = device.get_cusolver_sp_handle()
    nnz = A.nnz
    tol = 1.0
    reorder = 1
    x = cp.empty(m, dtype=dtype)
    singularity = np.empty(1, np.int32)

    if dtype == 'f':
        csrlsvqr = cusolver.scsrlsvchol
    else:
        csrlsvqr = cusolver.dcsrlsvchol

    csrlsvqr(
        handle, m, nnz, A._descr.descriptor, A.data.data.ptr,
        A.indptr.data.ptr, A.indices.data.ptr, b.data.ptr, tol, reorder,
        x.data.ptr, singularity.ctypes.data)

    # The return type of SciPy is always float64. Therefore, x must be casted.
    x = x.astype(np.float64)
    ret = (x, None, None, None, None, None, None, None, None, None)
    return ret