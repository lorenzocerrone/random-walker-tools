import numpy as np

from rwtools.randomwalker_algorithm import random_walker_algorithm_2d

x = np.ones((128, 128))
x[64, :] = 1

seeds = np.zeros_like(x).astype(np.int)

seeds[0, 0] = 1
seeds[-1, -1] = 2

_x = random_walker_algorithm_2d(x, seeds_mask=seeds, beta=1, offsets=((1, 0), (0, 1)), solving_mode="direct",
                                return_prob=False, divide_by_std=False)
print(_x)



