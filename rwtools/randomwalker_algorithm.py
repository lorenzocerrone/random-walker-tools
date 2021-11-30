import numpy as np

from rwtools.graphtools.solvers import Solver
from rwtools.graphtools.graphtools import graph2adjacency, adjacency2laplacian, stack2edge_weights
from rwtools.graphtools.graphtools import build_nd_grid_graph
from rwtools.utils import seeds_list2mask, lap2lapu_bt, sparse_pm, pu2p


def compute_random_walker(edge_weights: np.ndarray,
                          edges: np.ndarray,
                          seeds_mask: np.ndarray,
                          solving_mode: str = 'direct') -> np.ndarray:
    """
    Args:
        edge_weights:
        edges:
        seeds_mask:
        solving_mode:

    Returns:

    """
    adj_csc = graph2adjacency(edges, edge_weights)
    lap_csc = adjacency2laplacian(adj_csc, mode=0)
    lap_u_csc, b_t = lap2lapu_bt(lap_csc, seeds_mask)
    pm = sparse_pm(seeds_mask)
    pu = Solver(mode=solving_mode)(lap_u_csc, b_t.dot(pm))
    pu = np.array(pu, dtype=np.float32) if type(pu) == np.ndarray else np.array(pu.toarray(), dtype=np.float32)
    p = pu2p(pu, seeds_mask)
    return p


def random_walker_algorithm_nd(stack: np.ndarray,
                               beta: float = 130,
                               seeds_mask: np.ndarray = None,
                               seeds_list: list = None,
                               multichannel: bool = False,
                               offsets: tuple = ((0, 1, 0), (0, 0, 1), (1, 0, 0)),
                               divide_by_std: bool = True,
                               solving_mode: str = 'direct',
                               return_prob: bool = False) -> np.ndarray:
    """
    Implementation of the Random Walker Algorithm for 3D volumetric images
    Random walks for image segmentation, Leo Grady, 2006. IEEE Trans.

    Parameters
    ----------
    stack
    beta
    seeds_mask
    seeds_list
    multichannel
    offsets
    divide_by_std
    solving_mode
    return_prob

    Returns
    -------

    """

    assert (seeds_mask is not None or seeds_list is not None), 'A seeds mask or a seeds list is need'

    if seeds_mask is None:
        seeds_mask = seeds_list2mask(seeds_list, shape=stack.shape)

    # check_seeds(seeds_mask)
    if np.max(seeds_mask) < 2:
        p = np.ones(stack.shape).astype(np.float32)
        return p if return_prob else p[..., None]

    edges = build_nd_grid_graph(size=stack.shape, offsets=offsets)
    edge_weights = stack2edge_weights(stack, edges, beta, multichannel=multichannel, divide_by_std=divide_by_std)

    p = compute_random_walker(edge_weights, edges, seeds_mask, solving_mode)

    out_shape = stack.shape + (-1, )
    p = p.reshape(*out_shape)
    return p if return_prob else np.argmax(p, axis=-1)


def random_walker_algorithm_2d(image: np.ndarray,
                               beta: float = 130,
                               seeds_mask: np.ndarray = None,
                               seeds_list: list = None,
                               offsets: tuple = ((0, 1, 0), (0, 0, 1), (1, 0, 0)),
                               divide_by_std: bool = True,
                               solving_mode: str = 'direct',
                               return_prob: bool = False) -> np.ndarray:
    """
    Implementation of the Random Walker Algorithm for 2D images
    Random walks for image segmentation, Leo Grady, 2006. IEEE Trans.

    Parameters
    ----------
    image
    seeds_mask
    beta
    seeds_list
    offsets
    divide_by_std
    solving_mode
    return_prob

    Returns
    -------

    """
    assert image.ndim in [2, 3], 'Image must be a 2D gray scale (H, W) or multichannel (H, W, C)'
    multichannel = False if image.ndim == 2 else True
    return random_walker_algorithm_nd(image,
                                      beta=beta,
                                      seeds_mask=seeds_mask,
                                      seeds_list=seeds_list,
                                      multichannel=multichannel,
                                      offsets=offsets,
                                      divide_by_std=divide_by_std,
                                      solving_mode=solving_mode,
                                      return_prob=return_prob)


def random_walker_algorithm_3d(volume: np.ndarray,
                               beta: float = 130,
                               seeds_mask: np.ndarray = None,
                               seeds_list: list = None,
                               offsets: tuple = ((0, 1, 0), (0, 0, 1), (1, 0, 0)),
                               divide_by_std: bool = True,
                               solving_mode: str = 'direct',
                               return_prob: bool = False) -> np.ndarray:
    """
    Implementation of the Random Walker Algorithm for 3D volumetric images
    Random walks for image segmentation, Leo Grady, 2006. IEEE Trans.

    Parameters
    ----------
    volume
    beta
    seeds_mask
    seeds_list
    offsets
    divide_by_std
    solving_mode
    return_prob

    Returns
    -------

    """
    assert volume.ndim in [3, 4], 'Volume must be a 3D gray scale (D, H, W) or multichannel (D, H, W, C)'
    multichannel = False if volume.ndim == 2 else True
    return random_walker_algorithm_nd(volume,
                                      beta=beta,
                                      seeds_mask=seeds_mask,
                                      seeds_list=seeds_list,
                                      multichannel=multichannel,
                                      offsets=offsets,
                                      divide_by_std=divide_by_std,
                                      solving_mode=solving_mode,
                                      return_prob=return_prob)
