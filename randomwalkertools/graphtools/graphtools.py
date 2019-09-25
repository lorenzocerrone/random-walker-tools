import numpy as np
import numexpr
from scipy.sparse import csc_matrix, diags, eye


def graph2adjacency(graph, edges=None, num_nodes=None, is_undirected=True):
    """
    This Function create a sparse csc adjacency matrix from a graph matrix.

    Parameters
    ----------
    graph (array 2 x d): Contains the tuples specifying the graph structure.
    edges (array d): A Optional set of weights for each edges.
    num_nodes (int): Number of nodes in the graph, if not given it will inferred from the graph.
    is_undirected (bool): If True the adjacency matrix will be symmetrized.

    Returns
    -------
     A (csc_matrix): adjacency matrix

    """

    assert graph.shape[0] == 2 and graph.ndim == 2, 'Input "graph" is expected to be 2 x #edges'

    if edges is None:  # if no edges is given fill the adjacency with unity edges
        edges = np.ones(graph.shape[-1])
    else:
        assert (edges.shape[0] == graph.shape[1] and
                edges.ndim == 1), 'Input "edges" is expected to be a vector (#edges)'

    if num_nodes is None:  # number of node inference
        num_nodes = graph.max() + 1
    else:
        assert type(num_nodes) == int, 'Num of nodes must be a integer'

    A = csc_matrix((edges, graph), shape=(num_nodes, num_nodes))  # create the basic adjacency

    if is_undirected:  # symmetrization
        diag = diags(A.diagonal())
        A = A + A.transpose() - diag

    return A


def adjacency2laplacian(A, D=None, mode=0):
    """
    This function create a graph laplacian matrix from the adjacency matrix.
    Parameters
    ----------
    A (sparse matrix): Adjacency matrix.
    D (Degree matrix): Optional, diagonal matrix containing the sum over the adjacency row.
    mode (int): 0 Returns the standard graph Laplacian, L = D - A.
                1 Returns the random walk normalized graph Laplacian, L = I - D^-1 * A.
                2 Returns the symmetric normalized graph Laplacian, L = I - D^-0.5 * A D^-0.5.

    Returns
    -------
    L (sparse matrix): graph Laplacian
    """
    if D is None:  # compute the degree matrix
        D = adjacency2degree(A)

    if mode == 0:  # standard graph Laplacian
        return D - A

    elif mode == 1:  # random walk graph Laplacian
        return eye(D.shape[0], format="csc") - D.power(-1) * A

    elif mode == 2:  # symmetric normalized graph Laplacian
        return eye(D.shape[0], format="csc") - D.power(-0.5) * A * D.power(-0.5)

    else:
        raise NotImplementedError


def adjacency2degree(A):
    """ Compute the degree matrix for a give adjacency matrix A"""
    return diags(np.asarray(A.sum(1)).reshape(-1), format="csc")


def adjacency2transition(A, D=None):
    """ Compute the transition matrix associated with the adjacency matrix A"""
    if D is None:
        D = adjacency2degree(A)
    return A * D.power(-1)


def make2d_lattice_graph(size=(3, 3), offsets=((1, 0), (0, 1))):
    """
    This functions create a graph matrix for a 2D square lattice. The indices will be raveled.
    Parameters
    ----------
    size ((int, int)): Size of the lattice.
    offsets (list of tuples): Defines the offsets. Example [[0, 1], [1, 0], [1, -1]].

    Returns
    -------
    graph matrix (array 2 x d)
    """
    assert len(size) == 2, "Size must be a tuple of length 2"
    g = np.arange((size[0] * size[1])).reshape(size[0], size[1])  # template matrix for the indices

    all_g0, all_g1 = [], []
    for offset in offsets:
        """
        Convoluted hack, slack offsets are needed for diagonal offsets. In case of negative offset the slack offset take
        the inverse value. 
        """
        _slack_offset = [-o if o < 0 else None for o in offset]
        _offset = [None if o == 0 or o < 0 else -o for o in offset]
        all_g0.append(g[_slack_offset[0]:_offset[0], _slack_offset[1]:_offset[1]].ravel())

        _slack_offset = [o if o < 0 else None for o in offset]
        _offset = [None if o == 0 or o < 0 else o for o in offset]
        all_g1.append(g[_offset[0]:_slack_offset[0], _offset[1]:_slack_offset[1]].ravel())

    all_g0 = np.hstack(all_g0)  # concatenate all results
    all_g1 = np.hstack(all_g1)

    return np.stack([all_g0, all_g1]).reshape(2, -1)  # Stack the matrix together


def make3d_lattice_graph(size=(3, 3, 3), offsets=((0, 0, 1), (0, 1, 0), (1, 0, 0))):
    """
    This functions create a graph matrix for a 3D square lattice. The indices will be raveled.
    Parameters
    ----------
    size ((int, int)): Size of the lattice.
    offsets (list of tuples): Defines the offsets. Example [[0, 0, 1], [1, 0, 0], [0, 1, -1]].

    Returns
    -------
    graph matrix (array 2 x d)
    """
    assert len(size) == 3, "Size must be a tuple of lenght 3"
    g = np.arange((size[0] * size[1] * size[2])).reshape(size[0], size[1], size[2])  # template matrix for the indices

    all_g0, all_g1 = [], []
    for offset in offsets:
        """
        Convoluted hack, slack offsets are needed for diagonal offsets. In case of negative offset the slack offset take
        the inverse value. 
        """
        _slack_offset = [-o if o < 0 else None for o in offset]
        _offset = [None if o == 0 or o < 0 else -o for o in offset]
        all_g0.append(g[_slack_offset[0]:_offset[0],
                      _slack_offset[1]:_offset[1],
                      _slack_offset[2]:_offset[2]].ravel())

        _slack_offset = [o if o < 0 else None for o in offset]
        _offset = [None if o == 0 or o < 0 else o for o in offset]
        all_g1.append(g[_offset[0]:_slack_offset[0],
                      _offset[1]:_slack_offset[1],
                      _offset[2]:_slack_offset[2]].ravel())

    all_g0 = np.hstack(all_g0)  # concatenate all results
    all_g1 = np.hstack(all_g1)

    return np.stack([all_g0, all_g1]).reshape(2, -1)  # stack the matrix together


def gaussian_kernel(x, y, beta, fast_big_images=128):
    if max(x.shape) > fast_big_images ** 2:  # numbexp optimization for images bigger than 128 x 128
        edges = numexpr.evaluate("sum((x - y) ** 2, axis = 1)")  # vompute the nodes L2 distance
        return numexpr.evaluate("exp(- beta * edges)")  # compute the negative exponential
    else:
        edges = ((x - y) ** 2).sum(1)  # compute the nodes L2 distance
        return np.exp(- beta * edges)  # compute the negative exponential


def image2edges(image, graph, beta, divide_by_std=True, fast_big_images=128, kernel=gaussian_kernel):
    """
    Given an image and a graph matrix creates a set of weights. The weights are calculate using a Gaussian kernel.

    Parameters
    ----------
    image (array H x W x Channels): Input image. Can be a gray image, rgb, or multichannel.
    graph (array 2 x d): Graph matrix.
    beta (float): Hyper parameter, define an effective temperature of the kernel.
    divide_by_std (Bool): Divide beta by the std.
    fast_big_images (int): Speed up can be achieved with numbexp for large graphs.

    Returns
    -------
    edges (array d): Array containing edge weights.

    """
    assert image.ndim in [2, 3]
    image_r = image.reshape(-1, image.shape[-1]) if image.ndim == 3 else image.reshape(-1, 1)
    image_x, image_y = image_r[graph[0]], image_r[graph[1]]

    if divide_by_std:  # beta regularization
        beta /= 10 * (image.std() + 1e-16)

    return kernel(image_x, image_y, beta, fast_big_images)


def volumes2edges(image, graph, beta, divide_by_std=True, fast_big_images=128, kernel=gaussian_kernel):
    """
    Given a volumetric image and a graph matrix creates a set of weights.
    The weights are calculate using a Gaussian kernel.

    Parameters
    ----------
    image (array H x W x Channels): Input image. Can be a gray image, rgb, or multichannel.
    graph (array 2 x d): Graph matrix.
    beta (float): Hyper parameter, define an effective temperature of the kernel.
    divide_by_std (Bool): Divide beta by the std.
    fast_big_images (int): Speed up can be achieved with numbexp for large graphs.

    Returns
    -------
    edges (array d): Array containing edge weights.

    """
    assert image.ndim in [3, 4]
    image_r = image.reshape(-1, image.shape[-1]) if image.ndim == 4 else image.reshape(-1, 1)
    image_x, image_y = image_r[graph[0]], image_r[graph[1]]

    if divide_by_std:  # beta regularization
        beta /= 10 * (image.std() + 1e-16)

    return kernel(image_x, image_y, beta, fast_big_images)


if __name__ == "__main__":
    import time

    shape = (300, 300, 30)
    _image = np.random.rand(shape[0], shape[1], shape[2], 5)

    timer = time.time()
    _graph = make3d_lattice_graph(shape, offsets=[[0, 0, 2],
                                                  [-1, 1, 1],
                                                  [0, 1, 0]])

    _edges = volumes2edges(_image, _graph, beta=10)
    _A = graph2adjacency(_graph, _edges)
    _L = adjacency2laplacian(_A)

    timer = time.time() - timer
    print("From building graph to graph Laplacian in %0.3f" % timer)

    shape = (300, 300)
    _image = np.random.rand(shape[0], shape[1], 3)

    timer = time.time()
    _graph = make2d_lattice_graph(shape, offsets=[[0, 2],
                                                  [-1, 1],
                                                  [1, 0]])

    _edges = image2edges(_image, _graph, beta=10)
    _A = graph2adjacency(_graph, _edges)
    _L = adjacency2laplacian(_A)

    timer = time.time() - timer
    print("From building graph to graph Laplacian in %0.3f" % timer)
