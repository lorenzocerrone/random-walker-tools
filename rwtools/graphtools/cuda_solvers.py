import numpy as np

from scipy.sparse import csc_matrix
import cupy as cp
import cupyx as cpx
import cupyx.scipy.sparse.linalg


def csc_to_gpu(adj_csc: csc_matrix) -> cpx.scipy.sparse.csc_matrix:
    cp_csc_data = cp.asarray(adj_csc.data.ravel().astype(np.float32))
    cp_csc_indices = cp.asarray(adj_csc.indices.ravel())
    cp_csc_indptr = cp.asarray(adj_csc.indptr.ravel())

    csc_gpu = cpx.scipy.sparse.csc_matrix((cp_csc_data, cp_csc_indices, cp_csc_indptr), shape=adj_csc.shape)
    return csc_gpu


def solve_gpu(adj_csc: csc_matrix, b: csc_matrix) -> np.ndarray:
    """
    This function solves the linear system of equations: Ax = b, using chlomod solver on the GPU.
    Parameters
    ----------
    adj_csc: Sparse csr matrix (NxN)s
    b: Sparse array or array (NxM)

    returns x array (NxM)
    -------
    """
    # The actual cast will be performed slice by slice to reduce memory footprint
    b = b.astype(np.float32) if type(b) == np.ndarray else b.todense().astype(np.float32)
    b_gpu = cp.asarray(np.array(b))

    adj_gpu = csc_to_gpu(adj_csc)
    adj_splu = cpx.scipy.sparse.linalg.splu(adj_gpu)

    pu = cp.zeros_like(b_gpu)
    for i in range(b.shape[-1]):
        pu[:, i] = adj_splu.solve(b_gpu[:, i])

    pu = cp.asnumpy(pu).astype('float32')
    return pu


def solve_gpu_cg(adj_csc: csc_matrix, b: csc_matrix, tol: float = 1e-3) -> np.ndarray:
    """
    This function solves the linear system of equations: Ax = b, using chlomod solver on the GPU.
    Parameters
    ----------
    adj_csc: Sparse csr matrix (NxN)s
    b: Sparse array or array (NxM)
    tol:

    returns x array (NxM)
    -------
    """
    # The actual cast will be performed slice by slice to reduce memory footprint
    b = b.astype(np.float32) if type(b) == np.ndarray else b.todense().astype(np.float32)
    b_gpu = cp.asarray(np.array(b))

    adj_gpu = csc_to_gpu(adj_csc)

    pu = cp.zeros_like(b_gpu)
    for i in range(b.shape[-1]):
        pu[:, i], _ = cpx.scipy.sparse.linalg.cg(adj_gpu, b_gpu[:, i], tol=tol)

    pu = cp.asnumpy(pu).astype('float32')
    return pu
