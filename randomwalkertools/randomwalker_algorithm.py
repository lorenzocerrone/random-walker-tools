import numpy as np
from .randomwalker_tools import sparse_pm, lap2lapu_bt, pu2p, seeds_list2mask
from .graphtools.graphtools import make2d_lattice_graph, make3d_lattice_graph
from .graphtools.graphtools import image2edges, volumes2edges
from .graphtools.graphtools import graph2adjacency, adjacency2laplacian
from .graphtools.solvers import direct_solver, solve_cg_mg, solve_cg, solve_gpu
solver = {"direct": direct_solver,
          "cg_mg": solve_cg_mg,
          "cg": solve_cg,
          "cuda": solve_gpu}


def random_walker_algorithm_2d(image,
                               beta=130,
                               seeds_mask=None,
                               seeds_list=None,
                               offsets=((0, 1), (1, 0)),
                               divide_by_std=True,
                               solving_mode="direct",
                               return_prob=False):
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
    assert (seeds_mask is not None or seeds_list is not None), "A seeds mask or a seeds list is need"
    if seeds_mask is None:
        seeds_mask = seeds_list2mask(seeds_list, (image.shape[0], image.shape[1]))

    assert image.ndim in [2, 3], "Image must be a 2D gray scale (H, W) or multichannel (H, W, C)"

    # check_seeds(seeds_mask)
    if seeds_mask.max() < 2:
        if return_prob:
            return np.ones((seeds_mask.shape[0], seeds_mask.shape[1], 1)).astype(np.float32)
        else:
            return np.ones((seeds_mask.shape[0], seeds_mask.shape[1])).astype(np.float32)

    graph = make2d_lattice_graph(size=(image.shape[0],
                                       image.shape[1]), offsets=offsets)

    edges = image2edges(image, graph, beta, divide_by_std=divide_by_std)
    A = graph2adjacency(graph, edges)
    L = adjacency2laplacian(A, mode=0)

    Lu, Bt = lap2lapu_bt(L, seeds_mask)
    pm = sparse_pm(seeds_mask)

    pu = solver[solving_mode](Lu, Bt.dot(pm))

    pu = np.array(pu, dtype=np.float32) if type(pu) == np.ndarray else np.array(pu.toarray(), dtype=np.float32)
    p = pu2p(pu, seeds_mask).reshape(image.shape[0],
                                     image.shape[1],
                                     -1)
    if return_prob:
        return p
    else:
        return np.argmax(p, axis=-1)


def random_walker_algorithm_3d(volume,
                               beta=130,
                               seeds_mask=None,
                               seeds_list=None,
                               offsets=((0, 0, 1), (0, 1, 0), (1, 0, 0)),
                               divide_by_std=True,
                               solving_mode="direct",
                               return_prob=False):
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

    assert (seeds_mask is not None or seeds_list is not None), "A seeds mask or a seeds list is need"

    if seeds_mask is None:
        seeds_mask = seeds_list2mask(seeds_list, (volume.shape[0], volume.shape[1], volume.shape[2]))

    assert volume.ndim in [3, 4], "Volume must be a 3D gray scale (D, H, W) or multichannel (D, H, W, C)"

    # check_seeds(seeds_mask)
    if seeds_mask.max() < 2:
        if return_prob:
            return np.ones((seeds_mask.shape[0], seeds_mask.shape[1], seeds_mask.shape[2], 1)).astype(np.float32)
        else:
            return np.ones((seeds_mask.shape[0], seeds_mask.shape[1], seeds_mask.shape[2])).astype(np.float32)

    graph = make3d_lattice_graph(size=(volume.shape[0],
                                       volume.shape[1],
                                       volume.shape[2]), offsets=offsets)
    edges = volumes2edges(volume, graph, beta,  divide_by_std=divide_by_std)

    A = graph2adjacency(graph, edges)
    L = adjacency2laplacian(A, mode=0)

    Lu, Bt = lap2lapu_bt(L, seeds_mask)
    pm = sparse_pm(seeds_mask)

    pu = solver[solving_mode](Lu, Bt.dot(pm))

    pu = np.array(pu, dtype=np.float32) if type(pu) == np.ndarray else np.array(pu.toarray(), dtype=np.float32)
    p = pu2p(pu, seeds_mask).reshape(volume.shape[0],
                                     volume.shape[1],
                                     volume.shape[2],
                                     -1)

    if return_prob:
        return p
    else:
        return np.argmax(p, axis=-1)
