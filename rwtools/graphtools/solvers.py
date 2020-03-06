import warnings

import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
from scipy.sparse.linalg import cg
from scipy.sparse.linalg import spsolve

from concurrent import futures

try:
    import pyamg

    use_direct_solver_mg = False

except ImportError:
    warnings.warn("Pyamg not installed, performance for big images will be drastically reduced."
                  " Reverting to direct solver.")
    use_direct_solver_mg = True

try:
    from rwtools.graphtools.chol_cupy import chol
    import cupy as cp
    import cupyx.scipy.sparse

    use_direct_solver_cupy = False

except ImportError:
    warnings.warn("cupy not installed, GPU solver not available. Reverting to direct solver.")

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

    A_solver, x = cholesky(A), []

    for i in range(b.shape[-1]):
        _x = A_solver.solve_A(b[:, i])
        _x = np.array(_x, dtype=np.float32) if type(_x) == np.ndarray else np.array(_x.toarray(), dtype=np.float32)
        x.append(_x)

    return np.concatenate(x, axis=1)

import time

import numba
@numba.jit(nopython=True,
           fastmath=True,
           nogil=True)
def sp_dot(data, indptr, indices, b):
    x = np.empty(b.shape[0])
    for i in range(b.shape[0]):
        ind_b, ind_e, _x = indptr[i], indptr[i + 1], 0.0
        for j in range(ind_b, ind_e):
            b_ind = indices[j]
            _x += data[j] * b[b_ind]

        x[i] = _x
    return x


@numba.jit(nopython=True,
           fastmath=True,
           nogil=True)
def _dot(x, y):
    _res = 0
    for i in range(x.shape[0]):
        _res += x[i] * y[i]

    return _res

import math
import time
@numba.jit(nogil=True)
def _cg(input):
    b, a_value, a_indptr, a_indices, a_shape, x, x_temp, tol, max_iteration = input
    r = b - sp_dot(a_value, a_indptr, a_indices, x)
    p = r
    r_old = _dot(r, r)
    for _ in range(max_iteration):

        a_p = sp_dot(a_value, a_indptr, a_indices, p)

        alpha = r_old / _dot(p, a_p)

        x = x + alpha * p
        r = r - alpha * a_p

        r_new = _dot(r, r)
        if math.sqrt(r_new) < tol:
            return x

        p = r + (r_new/r_old) * p

        r_old = r_new

    return x


def mp_cg(A, b, tol=1e-8, max_workers=None):
    """Experimental"""
    acsr = csr_matrix(A)
    a_value = acsr.data
    a_shape = acsr.shape
    a_indptr = acsr.indptr
    a_indices = acsr.indices

    b = np.array(b.todense())
    import multiprocessing
    if max_workers is None:
        max_workers = multiprocessing.cpu_count()
        max_workers = 20

    executor = futures.ProcessPoolExecutor(max_workers=max_workers)

    iterator = []
    for i in range(b.shape[-1]):
        _iterator = (b[:, i].ravel(),
                     a_value,
                     a_indptr,
                     a_indices,
                     a_shape,
                     np.zeros(a_shape[0]) + 1/b.shape[-1],
                     np.empty(a_shape[0]),
                     tol,
                     int(1e6))

        iterator.append(_iterator)

    timer = time.time()
    _x = executor.map(_cg, iterator)
    x = list(_x)
    print(max_workers, time.time() - timer)
    return np.array(x).reshape(b.shape[-1], -1).T


def cg_torch_sparse(A, b, tol=1e-4, max_iteration=None, gpu=False):
    """Experimental"""
    import torch
    from torch_sparse import spmm
    b = np.array(b, dtype=np.float32) if type(b) == np.ndarray else np.array(b.toarray(), dtype=np.float32)
    if gpu:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    b = torch.tensor(b).to(device)

    acoo = coo_matrix(A)
    a_index = np.stack([acoo.row, acoo.col])
    a_index = torch.tensor(a_index)
    a_index = a_index.long().to(device)

    a_value = torch.tensor(acoo.data).to(device)
    a_shape = acoo.shape

    xk = torch.rand_like(b).to(device)

    rk = b - spmm(a_index, a_value, a_shape[0], a_shape[1], xk)
    pk = rk
    if max_iteration is None:
        max_iteration = 10 * b.shape[0]

    for i in range(max_iteration):
        # Precompute A * pk
        A_pk = spmm(a_index, a_value, a_shape[0], a_shape[1], pk)

        # Alpha numerator
        left = rk.T.view(rk.shape[1], 1, rk.shape[0])
        right = rk.T.view(rk.shape[1], rk.shape[0], 1)
        alpha_numerator = left.bmm(right).view(-1)

        # Alpha denominator
        left = pk.T.view(pk.shape[1], 1, pk.shape[0])
        right = A_pk.T.view(pk.shape[1], pk.shape[0], 1)
        alpha_denominator = left.bmm(right).view(-1)

        # Alpha
        alpha = alpha_numerator / (alpha_denominator + 1e-16)
        alpha = alpha.T

        xk = xk + alpha * pk
        rk_plus = rk - alpha * A_pk

        left = rk_plus.T.view(rk_plus.shape[1], 1, rk_plus.shape[0])
        right = rk_plus.T.view(rk_plus.shape[1], rk_plus.shape[0], 1)

        beta_numerator = left.bmm(right).view(-1)

        mask = beta_numerator > tol
        if mask.sum() < 1:
            print("exit after :", i)
            break
        """
        beta_numerator = beta_numerator[mask]
        alpha_numerator = alpha_numerator[mask]
        xk = xk[:, mask]
        pk = pk[:, mask]
        rk_plus = rk_plus[:, mask]
        """

        beta = beta_numerator / (alpha_numerator)

        beta = beta.T

        pk = rk_plus + beta * pk

        rk = rk_plus

        # print(xk, alpha, beta, rk_plus)

    return xk.cpu().numpy()


def solve_cg_mg(A, b, tol=1e-8, pre_conditioner=True):
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


def solve_cg(A, b, tol=1e-8):
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
    return solve_cg_mg(A, b, tol=tol, pre_conditioner=None)


def solve_gpu(A, b):
    """
    This function solves the linear system of equations: Ax = b, using chlomod solver on the GPU.
    Parameters
    ----------
    A: Sparse csr matrix (NxN)s
    b: Sparse array or array (NxM)

    returns x array (NxM)
    -------
    """
    if use_direct_solver_cupy:
        return direct_solver(A, b)

    # The actual cast will be performed slice by slice to reduce memory footprint
    b = b.astype(np.float32) if type(b) == np.ndarray else b.todense().astype(np.float32)
    b_gpu = cp.asarray(np.array(b))

    cp_A_data = cp.asarray(A.data.ravel().astype(np.float32))
    cp_A_incices = cp.asarray(A.indices.ravel())
    cp_A_indptr = cp.asarray(A.indptr.ravel())

    A_gpu = cupyx.scipy.sparse.csc_matrix((cp_A_data, cp_A_incices, cp_A_indptr), shape=A.shape)

    pu = []
    for i in range(b.shape[-1]):
        _pu = chol(A_gpu, b_gpu[:, i])[0]
        pu.append(cp.asnumpy(_pu))

    del A_gpu, b_gpu
    return np.array(pu, dtype=np.float32).T
