import math

import numba
import numpy as np


@numba.jit(nogil=True)
def ichol_csc(data, indices, indptr, n):
    """
    Incomple cholesky factorization of a sparse matrix.
    """
    for k in range(n):
        k_b, k_e = indptr[k], indptr[k + 1]
        _d = math.sqrt(data[k_b])
        data[k_b] = _d
        for i_k in range(k_b + 1, k_e):
            data[i_k] = data[i_k] / _d

        for j_k in range(k_b + 1, k_e):
            ind_j = indices[j_k]
            j_b, j_e = indptr[ind_j], indptr[ind_j + 1]
            for i_j in range(j_b, j_e):
                ind_i = indices[i_j]
                if ind_i == ind_j:
                    data[i_j] = data[i_j] - data[j_k] * data[j_k]

    return data


@numba.jit(nogil=True)
def solve_l_csr(data, indices, indptr, n, b):
    x = np.zeros_like(b)
    for k in range(n):
        k_b, k_e = indptr[k], indptr[k + 1]
        _x = 0
        for i in range(k_b, k_e):
            ind = indices[i]
            _x += data[i] * x[ind]
        x[k] = (b[k] - _x) / data[k_e - 1]
    return x


@numba.jit(nogil=True)
def solve_u_csr(data, indices, indptr, n, b):
    x = np.zeros_like(b)
    for k in range(n - 1, -1, -1):
        k_b, k_e = indptr[k + 1], indptr[k]
        _x = 0
        for i in range(k_e, k_b):
            ind = indices[i]
            _x += data[i] * x[ind]
        x[k] = (b[k] - _x) / data[k_e]
    return x


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


@numba.jit(nogil=True)
def ichol_solve(ichol_data, ichol_indices, ichol_indptr, n, b, is_csc):
    # U(low) dot U.T(up) x = b
    # U(low) dot y = b and U.T(up) x = y
    if is_csc:
        ichol_data, ichol_indices, ichol_indptr, n = csc2csr(ichol_data,
                                                             ichol_indices,
                                                             ichol_indptr,
                                                             n)

    x = solve_l_csr(ichol_data, ichol_indices, ichol_indptr, n, b)
    ichol_data, ichol_indices, ichol_indptr, n = transpose(ichol_data,
                                                           ichol_indices,
                                                           ichol_indptr,
                                                           n)
    x = solve_u_csr(ichol_data, ichol_indices, ichol_indptr, n, x)
    return x


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


@numba.jit(nogil=True)
def _cg(input):
    (b,
     a_value,
     a_indices,
     a_indptr,
     a_shape,
     x,
     tol,
     max_iteration) = input

    r = b - sp_dot(a_value, a_indices, a_indptr, x)
    p = r
    r_old = _dot(r, r)
    for _ in range(max_iteration):

        a_p = sp_dot(a_value, a_indices, a_indptr, p)

        alpha = r_old / _dot(p, a_p)

        x = x + alpha * p
        r = r - alpha * a_p

        r_new = _dot(r, r)
        if math.sqrt(r_new) < tol:
            return x

        p = r + (r_new / r_old) * p

        r_old = r_new

    return x

@numba.jit(nogil=True)
def _cg_ichol_preconditioned(input):
    (b,
     a_value,
     a_indices,
     a_indptr,
     a_shape,
     ichol_value,
     ichol_indices,
     ichol_indptr,
     ichol_shape,
     x,
     tol,
     max_iteration) = input

    r = b - sp_dot(a_value, a_indices, a_indptr, x)
    z = ichol_solve(ichol_value, ichol_indices, ichol_indptr, ichol_shape, r, False)
    p = z
    r_old = _dot(r, z)
    for _ in range(max_iteration):

        a_p = sp_dot(a_value, a_indices, a_indptr, p)

        alpha = r_old / _dot(p, a_p)

        x = x + alpha * p
        r = r - alpha * a_p

        z = ichol_solve(ichol_value, ichol_indices, ichol_indptr, ichol_shape, r, False)

        r_new = _dot(z, r)
        if math.sqrt(r_new) < tol:
            return x

        p = z + (r_new / r_old) * p

        r_old = r_new

    return x
