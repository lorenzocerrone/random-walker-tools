import numpy as np

from rwtools.graphtools.graphtools import compute_randomwalker
from rwtools.graphtools.graphtools import image2edges, volumes2edges
from rwtools.graphtools.graphtools import make2d_lattice_graph, make3d_lattice_graph
from rwtools.utils import seeds_list2mask


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
        p = np.ones(image.shape).astype(np.float32)
        return p if return_prob else p[..., None]

    graph = make2d_lattice_graph(size=(image.shape[0],
                                       image.shape[1]), offsets=offsets)

    edges = image2edges(image, graph, beta, divide_by_std=divide_by_std)

    p = compute_randomwalker(edges, graph, seeds_mask, solving_mode)
    p = p.reshape(image.shape[0], image.shape[1], -1)

    return p if return_prob else np.argmax(p, axis=-1)


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
        p = np.ones(volume.shape).astype(np.float32)
        return p if return_prob else p[..., None]

    graph = make3d_lattice_graph(size=(volume.shape[0],
                                       volume.shape[1],
                                       volume.shape[2]), offsets=offsets)
    edges = volumes2edges(volume, graph, beta, divide_by_std=divide_by_std)

    p = compute_randomwalker(edges, graph, seeds_mask, solving_mode)
    p = p.reshape(volume.shape[0], volume.shape[1], volume.shape[2], -1)

    return p if return_prob else np.argmax(p, axis=-1)
