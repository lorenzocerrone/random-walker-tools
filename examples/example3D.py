import numpy as np

from rwtools.randomwalker_algorithm import random_walker_algorithm_3d

x = np.random.rand(10, 100, 100)
seeds = np.zeros_like(x).astype(np.int)

seeds[0, 0, 0] = 1
seeds[-1, -1, -1] = 2

prob = random_walker_algorithm_3d(x, seeds_mask=seeds, beta=1, solving_mode="cg_mg", return_prob=True)
