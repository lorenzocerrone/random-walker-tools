import numpy as np
from .randomwalker_tools import sparse_pm, lap2lapu_bt, pu2p, seeds_list2mask
from randomwalkertools.graphtools.graphtools import make2d_lattice_graph, make3d_lattice_graph
from randomwalkertools.graphtools.graphtools import image2edges, volumes2edges
from randomwalkertools.graphtools.graphtools import graph2adjacency, adjacency2laplacian
from randomwalkertools.graphtools.solvers import direct_solver, solve_cg_mg
solver = {"direct": direct_solver, "multi_grid": solve_cg_mg}


def random_walker_algorithm_2d(image,
                               beta=130,
                               seeds_mask=None,
                               seeds_list=None,
                               offsets=((0, 1), (1, 0)),
                               divide_by_std=True,
                               solving_mode="direct"):
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

    Returns
    -------

    """
    assert (seeds_mask is not None or seeds_list is not None), "A seeds mask or a seeds list is need"
    if seeds_mask is None:
        seeds_mask = seeds_list2mask(seeds_list)

    assert image.ndim in [2, 3], "Image must be a 2D gray scale (H, W) or multichannel (H, W, C)"

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
    return p


def random_walker_algorithm_3d(volume,
                               beta=130,
                               seeds_mask=None,
                               seeds_list=None,
                               offsets=((0, 0, 1), (0, 1, 0), (1, 0, 0)),
                               divide_by_std=True,
                               solving_mode="direct"):
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

    Returns
    -------

    """

    assert (seeds_mask is not None or seeds_list is not None), "A seeds mask or a seeds list is need"
    if seeds_mask is None:
        seeds_mask = seeds_list2mask(seeds_list)

    assert volume.ndim in [3, 4], "Volume must be a 3D gray scale (D, H, W) or multichannel (D, H, W, C)"

    def generate_oversegmentation(volume):
        indx_signal = np.where(volume > 0.5)


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
    return p
