
import math

import numba
import numpy as np
from rwtools.graphtools.numba_cg import csc2csr, transpose, sp_dot, _dot


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
