import numpy as np
from scipy.sparse import csc_matrix


def seeds_list2mask(seeds_list, shape, seeds_labels=None):
    """
    Returns a array mask from a list of seeds.
    I.e. labels at seeds position and zeros everywhere else.

    Parameters
    ----------
    seeds_list (list of list): List of tuple containing the seeds.
    shape (tuple): Mask expected shape.
    seeds_labels (list of integers): Optional list of indices.
                                     If not given sequential labels will be assigned.

    Returns
    -------
    seeds_mask (array):  mask array

    """
    seeds_mask = np.zeros(shape, dtype=np.uint16)

    if seeds_labels is None:  # compute labels if not given
        seeds_labels = np.arange(1, len(seeds_list[0]) + 1)

    seeds_mask[tuple(seeds_list)] = seeds_labels  # placing the seeds
    return seeds_mask


def seeds_mask2list(seeds_mask):
    """ Returns a list of seeds from a mask."""
    return np.where(seeds_mask.ravel() != 0)[0]


def seeds_bool_mask(seed_mask):
    """ Returns a bool mask for unseeded nodes and one for seeded."""
    mask_u = seed_mask.ravel() == 0
    return mask_u


def pu2p(pu, seeds_mask):
    """ RW helper function. Returns full assignments matrix from unseeded assignments."""
    seeds_mask = seeds_mask.ravel()
    mask_u = seeds_bool_mask(seeds_mask)  # extract seeds bool mask
    p = np.zeros((seeds_mask.shape[0], pu.shape[-1]), dtype=np.float32)

    for s in range(seeds_mask.max()):  # fill the complete assignments matrix
        pos_s = np.where(seeds_mask == s + 1)
        p[pos_s, s] = 1
        p[mask_u, s] = pu[:, s]

    return p


def p2pu(p, seeds_mask):
    """ RW helper function. Returns the reduced (unseeded) assignment matrix."""
    mask_u = seeds_mask2list(seeds_mask)
    pu = p[mask_u]
    return pu


def lap2lapu_bt(lap, seeds_mask):
    """ RW helper function. Returns the unseeded x unseeded block and the unseeded x seeded."""
    mask_u = seeds_bool_mask(seeds_mask)
    return lap[mask_u][:, mask_u], - lap[mask_u][:, ~mask_u]


def sparse_pm(seeds_mask):
    """ RW helper function. Create a sparse matrix with the seeds information."""
    k = seeds_mask2list(seeds_mask)
    i_ind, j_ind = np.arange(k.shape[0]), (seeds_mask.ravel()[k] - 1).astype(np.uint16)
    val = np.ones_like(k, dtype=np.float)
    return csc_matrix((val, (i_ind, j_ind)), shape=(k.shape[0], j_ind.max() + 1))
