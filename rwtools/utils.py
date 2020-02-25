import numpy as np
from scipy.sparse import csc_matrix
from skimage.morphology import square, disk, erosion
from skimage.segmentation import find_boundaries
from scipy.ndimage.morphology import distance_transform_edt


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

    if p.shape[-1] == 0:
        return np.ones((seeds_mask.shape[0], 1), dtype=np.float32)

    for s in range(seeds_mask.max()):  # fill the complete assignments matrix
        pos_s = np.where(seeds_mask == s + 1)
        p[pos_s, s] = 1
        p[mask_u, s] = pu[:, s]

    return p


def p2pu(p, seeds_mask):
    """ RW helper function. Returns the reduced (unseeded) assignment matrix."""
    mask_u = seeds_bool_mask(seeds_mask)
    p = p.reshape(-1, p.shape[-1])
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


def seg2seeds(segmentation, beta=0.1, max_radius=10):
    boundary = find_boundaries(segmentation, connectivity=2)

    adjusted_seg = segmentation + 1
    adjusted_seg[boundary] = 0

    ids, counts = np.unique(adjusted_seg, return_counts=True)

    new_seg = np.zeros_like(adjusted_seg)
    seeds = np.zeros_like(adjusted_seg)

    for i, (_ids, _counts) in enumerate(zip(ids[1:], counts[1:])):
        mask = adjusted_seg == _ids

        radius = min(int(np.sqrt((beta * _counts) / np.pi)), max_radius)

        eroded_mask = erosion(mask, disk(radius))

        if np.sum(eroded_mask) == 0:
            dt_mask = distance_transform_edt(mask)
            mask_max = np.argmax(dt_mask)
            x, y = np.unravel_index(mask_max, mask.shape)
            seeds[x, y] = i + 1
        else:
            seeds[eroded_mask] = i + 1

        new_seg[(segmentation + 1) == _ids] = i

    return seeds, new_seg


def adjust_pmaps(pmaps):
    pmaps = (pmaps - pmaps.min()) / (pmaps.max() - pmaps.min())
    pmaps = 1 - 2 * pmaps
    return pmaps
