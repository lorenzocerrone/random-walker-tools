import numpy as np
from torch.autograd import Function
import torch

import numpy as np

from rwtools.graphtools import solvers
from rwtools.graphtools.graphtools import graph2adjacency, adjacency2laplacian
from rwtools.graphtools.graphtools import image2edges, volumes2edges
from rwtools.graphtools.graphtools import edges_tensor2graph, compute_randomwalker
from rwtools.utils import sparse_pm, lap2lapu_bt, pu2p, seeds_list2mask, p2pu
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import lobpcg
from sksparse.cholmod import cholesky
import pyamg

import time


def laplacian_eigen(edges_image, seeds=None, n_components=1, offsets=((0, 1), (1, 0), (1, 1))):
    """
    input : edges image 1 x C x H x W
    output: instances probability shape: C x H x W
    """
    # Pytorch Tensors to numpy
    np_edges_image = edges_image.clone().detach().numpy()
    np_edges_image = np.squeeze(np_edges_image)
    channels, image_shape = np_edges_image.shape[0], (np_edges_image.shape[1], np_edges_image.shape[2])

    np_edges_image = np_edges_image.reshape(channels, -1)

    np_seeds = seeds.numpy().astype(np.int)
    edges, edges_id, graph, graph_i, graph_j = edges_tensor2graph(np_edges_image, image_shape, offsets)

    A = graph2adjacency(graph, edges)
    L = adjacency2laplacian(A, mode=0)
    Lu, _ = lap2lapu_bt(L, np_seeds)

    ml = pyamg.ruge_stuben_solver(L)
    M = ml.aspreconditioner(cycle='V')

    ex = np.random.rand(Lu.shape[0], n_components)
    lamb = lobpcg(Lu, ex, largest=False, M=M)
    print(lamb)

    return None