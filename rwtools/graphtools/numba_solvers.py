import math

import numba

import numpy as np
from scipy.sparse import csr_matrix, csc_matrix


@numba.jit(nogil=True)
def csr2csc(data_csr, indices_csr, indptr_csr, n):
    data_csc = np.zeros(data_csr.shape[0], dtype=data_csr.dtype)
    indices_csc = np.zeros(indices_csr.shape[0], dtype=np.int32)
    indptr_csc = np.zeros(indptr_csr.shape[0], dtype=np.int32)
    curr = np.zeros(n, dtype=np.int32)

    for i in range(n):
        j_b, j_e = indptr_csr[i], indptr_csr[i + 1]
        for j in range(j_b, j_e):
            indptr_csc[indices_csr[j] + 1] += 1

    for i in range(1, n + 1):
        indptr_csc[i] += indptr_csc[i - 1]

    for i in range(n):
        j_b, j_e = indptr_csr[i], indptr_csr[i + 1]
        for j in range(j_b, j_e):
            loc = indptr_csc[indices_csr[j]] + curr[indices_csr[j]]
            curr[indices_csr[j]] += 1
            indices_csc[loc] = i
            data_csc[loc] = data_csr[j]

    return data_csc, indices_csc, indptr_csc, n


@numba.jit(nogil=True)
def transpose(data, indices, indptr, n):
    return csr2csc(data, indices, indptr, n)


@numba.jit(nogil=True)
def csc2csr(data_csc, indices_csc, indptr_csc, n):
    return csr2csc(data_csc, indices_csc, indptr_csc, n)


@numba.jit(nopython=True,
           fastmath=True,
           nogil=True)
def numba_sp_dot(a_data, a_indices, a_indptr, b):
    dot_res = np.empty(b.shape[0])
    for i in range(b.shape[0]):
        ind_b, ind_e, _x = a_indptr[i], a_indptr[i + 1], 0.0
        for j in range(ind_b, ind_e):
            b_ind = a_indices[j]
            _x += a_data[j] * b[b_ind]

        dot_res[i] = _x
    return dot_res


@numba.jit(nopython=True,
           fastmath=True,
           nogil=True)
def numba_dot(x, y):
    dot_res = 0
    for i in range(x.shape[0]):
        dot_res += x[i] * y[i]

    return dot_res


@numba.njit(parallel=True, fastmath=True)
def numba_cg(adj_value, adj_indices, adj_indptr, b, x_out, tol, max_iteration):
    for i in numba.prange(b.shape[1]):
        x = np.zeros_like(b[:, i])
        r = b[:, i] - numba_sp_dot(adj_value, adj_indices, adj_indptr, x)
        p = r
        r_old = numba_dot(r, r)
        for _ in range(max_iteration):

            a_p = numba_sp_dot(adj_value, adj_indices, adj_indptr, p)

            alpha = r_old / numba_dot(p, a_p)

            x = x + alpha * p
            r = r - alpha * a_p

            r_new = numba_dot(r, r)

            if math.sqrt(r_new) < tol:
                x_out[:, i] = x
                break

            p = r + (r_new / r_old) * p

            r_old = r_new
        else:
            x_out[:, i] = x
    return x_out


def solve_numba_cg(adj_csc: csc2csr, b: csc_matrix, tol: float = 1.e-3, max_iteration: int = 1e6):
    """
    Args:
        adj_csc:
        b:
        tol:
        max_iteration:

    Returns:

    """
    adj_csr = csr_matrix(adj_csc)
    a_value, a_indices, a_indptr = adj_csr.data, adj_csr.indices, adj_csr.indptr
    x_out = np.zeros_like(b)
    pu = numba_cg(a_value, a_indices, a_indptr, b, x_out=x_out, tol=tol, max_iteration=int(max_iteration))
    return pu
