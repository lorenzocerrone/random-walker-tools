import numpy as np
from scipy.sparse import csc_matrix, diags, eye


def graph2adjacency(edges: np.ndarray,
                    edge_weights: np.ndarray = None,
                    num_nodes: int = None,
                    is_undirected: bool = True) -> csc_matrix:
    """
    This Function create a sparse csc adjacency matrix from a graph matrix.

    Parameters
    ----------
    graph (array 2 x d): Contains the tuples specifying the graph structure.
    edges (array d): A Optional set of weights for each edges.
    num_nodes (int): Number of nodes in the graph, if not given it will inferred from the graph.
    is_undirected (bool): If True the adjacency matrix will be symmetries.

    Returns
    -------
     A (csc_matrix): adjacency matrix

    """

    assert edges.shape[0] == 2 and edges.ndim == 2, 'Input "graph" is expected to be 2 x #edges'

    if edge_weights is None:  # if no edges is given fill the adjacency with unity edges
        edge_weights = np.ones(edges.shape[-1])
    else:
        assert (edge_weights.shape[0] == edges.shape[1] and
                edge_weights.ndim == 1), 'Input "edges" is expected to be a vector (#edges)'

    if num_nodes is None:  # number of node inference
        num_nodes = np.max(edges) + 1
    else:
        assert type(num_nodes) == int, 'Num of nodes must be a integer'

    adj_csc = csc_matrix((edge_weights, edges), shape=(num_nodes, num_nodes))  # create the basic adjacency

    if is_undirected:  # symmetrization
        diag = diags(adj_csc.diagonal())
        adj_csc = adj_csc + adj_csc.transpose() - diag

    return adj_csc


def adjacency2laplacian(adj: csc_matrix, degree: csc_matrix = None, mode: int = 0) -> csc_matrix:
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
    degree = adjacency2degree(adj) if degree is None else degree

    if mode == 0:  # standard graph Laplacian
        return degree - adj

    elif mode == 1:  # random walk graph Laplacian
        return eye(degree.shape[0], format="csc") - degree.power(-1) * adj

    elif mode == 2:  # symmetric normalized graph Laplacian
        return eye(degree.shape[0], format="csc") - degree.power(-0.5) * adj * degree.power(-0.5)

    else:
        raise NotImplementedError


def adjacency2degree(adj: csc_matrix) -> csc_matrix:
    """ Compute the degree matrix for a give adjacency matrix A"""
    return diags(np.asarray(adj.sum(1)).reshape(-1), format="csc")


def adjacency2transition(adj: csc_matrix, degree: csc_matrix = None) -> csc_matrix:
    """ Compute the transition matrix associated with the adjacency matrix A"""
    degree = adjacency2degree(adj) if degree is None else degree
    return adj * degree.power(-1)


def build_nd_grid_graph(size: tuple = (3, 3), offsets: tuple = ((1, 0), (0, 1))) -> np.ndarray:
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

    raveled_indices = np.arange(np.prod(size)).reshape(*size)  # template matrix for the indices

    edge_list = []
    for sign in (-1, 1):
        _edge_list = []
        for offset in offsets:
            """
            Convoluted hack, slack offsets are needed for diagonal offsets. 
            In case of negative offset the slack offset take the inverse value. 
            """
            slack_offset_list = [sign * o if o < 0 else None for o in offset]
            offset_list = [None if o == 0 or o < 0 else sign * o for o in offset]
            iterator = zip(slack_offset_list, offset_list) if sign == -1 else zip(offset_list, slack_offset_list)
            _edge_list.append(raveled_indices[tuple(slice(s, o) for s, o in iterator)].ravel())

        edge_list.append(np.hstack(_edge_list))  # concatenate all results

    return np.stack(edge_list).reshape(2, -1)  # stack the matrix together


def gaussian_kernel(x: np.ndarray, y: np.ndarray, beta: float = 1.) -> np.ndarray:
    edges = ((x - y) ** 2).sum(1)  # compute the nodes L2 distance
    return np.exp(- beta * edges)  # compute the negative exponential


def stack2edge_weights(stack: np.ndarray,
                       edges: np.ndarray,
                       beta: float = 100,
                       multichannel: bool = False,
                       divide_by_std: bool = True,
                       kernel=gaussian_kernel) -> np.ndarray:
    """
    Given an image and a graph matrix creates a set of weights. The weights are calculate using a Gaussian kernel.

    Parameters
    ----------
    stack: (array H x W x Channels) Input image. Can be a gray image, rgb, or multichannel.
    edges:  (array 2 x d) Graph matrix.
    beta: (float) Hyper parameter, define an effective temperature of the kernel.
    multichannel: (bool)
    divide_by_std: (bool) Divide beta by the std.
    kernel: function
    Returns
    -------
    edge_weights (array d): Array containing edge weights.
    """

    stack_raveled = stack.reshape(-1, stack.shape[-1]) if multichannel else stack.reshape(-1, 1)
    stack_i, stack_j = stack_raveled[edges[0]], stack_raveled[edges[1]]

    if divide_by_std:  # beta regularization
        beta /= 10 * (stack.std() + 1e-16)

    return kernel(stack_i, stack_j, beta)


def edges_tensor2graph(edges_tensor, image_shape, offsets):
    # TODO reimplement
    graph_i, graph_j = make2d_lattice_graph(size=(image_shape[0],
                                                  image_shape[1]), offsets=offsets, hstack=False)
    edges, edges_id = [], []
    for c, graph in enumerate(graph_i):
        _edges = edges_tensor[c, graph]
        _id = np.ones_like(_edges) * c + 1
        edges.append(_edges)
        edges_id.append(_id)

    # stack edges
    edges = np.hstack(edges)
    edges_id = np.hstack(edges_id)

    # stack graph
    graph_ii = np.hstack(graph_i)
    graph_jj = np.hstack(graph_j)
    graph = np.stack([graph_ii, graph_jj]).reshape(2, -1)
    return edges, edges_id, graph, graph_i, graph_j
