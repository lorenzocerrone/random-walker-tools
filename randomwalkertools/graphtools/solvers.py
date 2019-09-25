from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import cg
from scipy.sparse import csr_matrix
import numpy as np
import warnings
use_direct_solver = False
from multiprocessing import Pool

try:
    import pyamg
except ImportError:
    warnings.warn("Pyamg not installed, performance for big images will be drastically reduced.")
    use_direct_solver = True


def direct_solver(A, b):
    """ Simple wrapper around scipy spsolve """
    return spsolve(A, b, use_umfpack=True)


def solve_cg_mg(A, b):
    """
    Implementation follows the source code of skimage: https://github.com/scikit-image/scikit-image/blob/master/skimage/segmentation/random_walker_segmentation.py
    it solves the linear system of equations: Ax = b, by coniugate gradient and using the Ruge Stuben solver as preconditioner
    Parameters
    ----------
    A Sparse csr matrix (NxN)
    b Sparse array or array (NxM):

    Returns: x array (NxM)
    -------

    """
    if use_direct_solver:
        return direct_solver(A, b)

    pu = []
    A = csr_matrix(A)
    b = b if type(b) == np.ndarray else b.todense()

    # preconditioner
    ml = pyamg.ruge_stuben_solver(A)

    # other options are all slower
    # ml = pyamg.coarse_grid_solver(A)
    # ml = pyamg.smoothed_aggregation_solver(A, diagonal_dominance=True)
    M = ml.aspreconditioner(cycle='V')
    for i in range(b.shape[-1]):
        pu.append(cg(A, b[:, i], tol=1e-4, M=M, maxiter=10)[0])

    return np.array(pu, dtype=np.float32).T


if __name__ == "__main__":
    import numpy as np
    import time
    from graphtools import make3d_lattice_graph, volumes2edges, graph2adjacency, adjacency2laplacian
    import vigra

    N = 40
    nx = 3
    x = np.zeros((N, N, N))
    x[:, N // 2] = 1

    graph = make3d_lattice_graph((N, N, N))

    edges = volumes2edges(x, graph, beta=1)

    A = graph2adjacency(graph, edges)
    L = adjacency2laplacian(A)

    b = np.random.rand(N*N*N, nx)

    timer = time.time()
    direct_solver(L, b)
    print(time.time() - timer)

    timer = time.time()
    solve_cg_mg(L, b)
    print(time.time() - timer)
