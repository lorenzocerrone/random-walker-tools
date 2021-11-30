import math

import numba
import numpy as np


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
def sp_dot(data, indices, indptr, b):
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


@numba.njit(parallel=True, fastmath=True)
def _cg(b, a_value, a_indices, a_indptr, tol, max_iteration):
    x_out = np.zeros_like(b)
    for i in numba.prange(b.shape[1]):
        x = np.zeros_like(b[:, i])
        r = b[:, i] - sp_dot(a_value, a_indices, a_indptr, x)
        p = r
        r_old = _dot(r, r)
        for _ in range(max_iteration):

            a_p = sp_dot(a_value, a_indices, a_indptr, p)

            alpha = r_old / _dot(p, a_p)

            x = x + alpha * p
            r = r - alpha * a_p

            r_new = _dot(r, r)

            if math.sqrt(r_new) < tol:
                x_out[:, i] = x
                break

            p = r + (r_new / r_old) * p

            r_old = r_new
        else:
            x_out[:, i] = x

    return x_out
