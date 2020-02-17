
from rwtools import random_walker_algorithm_2d
import numpy as np


class TestRWAlgorithm:
    def test_rw_2d(self):
        x = np.ones((16, 16))
        x[8, :] = 0
        seeds = np.zeros_like(x).astype(np.int)

        seeds[0, 0] = 1
        seeds[-1, -1] = 2

        _x = random_walker_algorithm_2d(x, seeds_mask=seeds, beta=1, offsets=((1, 0), (0, 1)), solving_mode="direct",
                                        return_prob=True, divide_by_std=False)

        _y = np.load("./resources/2d_2seeds_16.npy")
        assert np.allclose(_x, _y)