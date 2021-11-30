import numpy as np
from torch.autograd import Function
import torch

import numpy as np

from rwtools.graphtools import solvers
from rwtools.graphtools.graphtools import graph2adjacency, adjacency2laplacian
from rwtools.graphtools.graphtools import image2edge_weights, volume2edge_weights
from rwtools.graphtools.graphtools import edges_tensor2graph
from rwtools.randomwalker_algorithm import compute_random_walker
from rwtools.utils import sparse_pm, lap2lapu_bt, pu2p, seeds_list2mask, p2pu
from scipy.sparse import coo_matrix, csc_matrix
from sksparse.cholmod import cholesky

import time


class DifferentiableRandomWalker2D(torch.autograd.Function):

    @staticmethod
    def forward(ctx,
                edges_image,
                seeds=None,
                num_grad=1000,
                max_backprop=True,
                offsets=((0, 1), (1, 0), (1, 1)),
                mode_forward="direct",
                mode_backward="cholesky"):
        """
        input : edges image 1 x C x H x W
        output: instances probability shape: C x H x W
        """
        ctx.num_grad = num_grad
        ctx.max_backprop = max_backprop
        ctx.offsets = offsets
        ctx.mode_backward = mode_backward

        # Pytorch Tensors to numpy
        np_edges_image = edges_image.clone().detach().numpy()
        np_edges_image = np.squeeze(np_edges_image)
        ctx.channels, ctx.image_shape = np_edges_image.shape[0], (np_edges_image.shape[1], np_edges_image.shape[2])
        np_edges_image = np_edges_image.reshape(ctx.channels, -1)

        np_seeds = seeds.numpy().astype(np.int)
        ctx.edges, ctx.edges_id, ctx.graph, ctx.graph_i, ctx.graph_j = edges_tensor2graph(np_edges_image,
                                                                                          ctx.image_shape,
                                                                                          offsets)
        np_p = compute_random_walker(ctx.edges, ctx.graph, np_seeds, solving_mode=mode_forward)
        np_p = np_p.reshape(ctx.image_shape[0], ctx.image_shape[1], -1)
        p = torch.from_numpy(np_p)

        ctx.save_for_backward(seeds, p)
        return p

    @staticmethod
    def backward(ctx, grad_output):
        """
        input : grad from loss
        output: grad from the laplacian backprop
        """
        timer = time.time()
        seeds, p = ctx.saved_tensors
        np_seeds, np_p = seeds.numpy(), p.detach().numpy()

        pu = p2pu(np_p, np_seeds)

        # Pytorch Tensors to numpy
        np_gradout = grad_output.numpy()
        np_gradout = np_gradout.reshape(-1, np_gradout.shape[-1])
        np_gradout = p2pu(np_gradout, np_seeds)

        A = graph2adjacency(ctx.graph, ctx.edges)
        L = adjacency2laplacian(A)
        Lu, _ = lap2lapu_bt(L, np_seeds)

        A = graph2adjacency(ctx.graph, ctx.edges_id)
        Lu_id, _ = lap2lapu_bt(A, np_seeds)

        Lu_id = coo_matrix(Lu_id)
        ind_i, ind_j = Lu_id.col, Lu_id.row
        ind_e = np.array((Lu_id.tocsr()[ind_i, ind_j] - 1), dtype=np.int)[0, ...]

        mask = (ind_j - ind_i) > 0
        ind_i, ind_j, ind_e = ind_i[mask], ind_j[mask], ind_e[mask]

        # Sampling
        if ctx.num_grad < np.unique(ind_i).shape[0]:
            u_ind = np.unique(ind_i)
            grad2do = np.random.choice(u_ind, size=ctx.num_grad, replace=False)
        else:
            grad2do = np.unique(ind_i)

        # find maxgrad for each region
        if ctx.max_backprop:
            c_max = np.argmax(np.abs(np_gradout), axis=1)
        else:
            c_max = np.ones(np_gradout.shape[0])[:, None].dot(np.arange(np_gradout.shape[-1])[None, :])
            c_max = c_max.astype(np.int)

        if ctx.mode_backward == "cholesky":
            grad_input = cholesky_backprop_solver(ind_i, ind_j, ind_e,
                                                  grad2do, pu, c_max, Lu, np_gradout, ctx.channels)
        else:
            grad_input = standard_backprop_solver(ind_i, ind_j, ind_e,
                                                  grad2do, pu, c_max, Lu, np_gradout, ctx.channels, ctx.mode_backward)

        grad_input = grad_fill(grad_input, np_seeds, edges=ctx.channels).reshape((1,
                                                                                 ctx.channels,
                                                                                 ctx.image_shape[0],
                                                                                 ctx.image_shape[1]))
        return torch.from_numpy(grad_input), None, None, None, None


def standard_backprop_solver(ind_i, ind_j, ind_e, grad2do, pu, c_max, Lu, np_gradout, channels, mode):
    grad_input = np.zeros((1, channels, np_gradout.shape[0]))
    # Loops around all the edges
    all_dl, all_k, all_e = [], [], []
    for k, l, e in zip(ind_i, ind_j, ind_e):
        if k in grad2do:
            dl = np.zeros_like(pu)
            dl[l] = pu[k] - pu[l]
            dl[k] = pu[l] - pu[k]

            all_dl.append(dl[:, c_max[k]])
            all_k.append(k)
            all_e.append(e)

    partial_grad = solvers[mode](Lu, np.array(all_dl).T)
    grad = np.sum(np_gradout[:, c_max[all_k]] * partial_grad, axis=0)
    grad_input[0, all_e, all_k] = grad
    return grad_input


def cholesky_backprop_solver(ind_i, ind_j, ind_e, grad2do, pu, c_max, Lu, np_gradout, channels):
    ch_lap_solver = cholesky(csc_matrix(Lu))
    grad_input = np.zeros((1, channels, np_gradout.shape[0]))

    for k, l, e in zip(ind_i, ind_j, ind_e):
        if k in grad2do:
            dl = np.zeros_like(pu)
            dl[l] = pu[k] - pu[l]
            dl[k] = pu[l] - pu[k]

            partial_grad = ch_lap_solver.solve_A(dl[:, c_max[k]])
            grad = np.sum(np_gradout[:, c_max[k]] * partial_grad)
            grad_input[0, e, k] = grad
    return grad_input


def grad_fill(gradu, seeds, edges=2):
    """
    :param gradu: unseeded output probability
    :param seeds: RW seeds, must be the same size as the image
    :param edges: number of affinities for each pixel
    :return: p: the complete output probability
    """
    seeds_r = seeds.ravel()
    mask_u = seeds_r == 0
    grad = np.zeros((edges, seeds_r.shape[0]))
    grad[:, mask_u] = gradu

    return grad
