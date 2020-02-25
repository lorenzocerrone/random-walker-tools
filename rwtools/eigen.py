import numpy as np
from torch.autograd import Function
import torch

import numpy as np

from rwtools.graphtools import solvers
from rwtools.graphtools.graphtools import graph2adjacency, adjacency2laplacian
from rwtools.graphtools.graphtools import image2edges, volumes2edges
from rwtools.graphtools.graphtools import edges_tensor2graph, compute_randomwalker, make2d_lattice_graph
from rwtools.utils import sparse_pm, lap2lapu_bt, pu2p, seeds_list2mask, p2pu, pu_fill
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix
from scipy.sparse.linalg import lobpcg
from sksparse.cholmod import cholesky
import pyamg
import time


def laplacian_eigen(image,
                    seeds=None,
                    n_components=3,
                    beta=1,
                    divide_by_std=False, offsets=((0, 1), (1, 0))):
    """
    input : edges image 1 x C x H x W
    output: instances probability shape: C x H x W
    """
    # Pytorch Tensors to numpy
    graph = make2d_lattice_graph(size=(image.shape[0],
                                       image.shape[1]), offsets=offsets)

    edges = image2edges(image, graph, beta, divide_by_std=divide_by_std)

    A = graph2adjacency(graph, edges)
    L = adjacency2laplacian(A, mode=0)
    Lu, _ = lap2lapu_bt(L, seeds)

    ml = pyamg.ruge_stuben_solver(csr_matrix(Lu))
    M = ml.aspreconditioner(cycle='V')

    ex = np.random.rand(Lu.shape[0], n_components + 1).astype(np.float32)

    eigen_values, eigen_vectors = lobpcg(Lu, ex, largest=False, M=M, tol=1e-8)
    #eigen_values, eigen_vectors = lobpcg(Lu, ex, largest=False, tol=1e-8)
    eigen_values, eigen_vectors = eigen_values[1:], eigen_vectors[:, 1:]

    eigen_vectors = pu_fill(eigen_vectors, seeds)
    return eigen_values, eigen_vectors.reshape(image.shape[0],
                                               image.shape[1],
                                               -1)
