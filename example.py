import numpy as np
from randomwalkertools.randomwalker_algorithm import random_walker_algorithm_2d, random_walker_algorithm_3d
import time

x = np.ones((128, 128))
seeds = np.zeros_like(x).astype(np.int)

seeds[0, 0] = 1
seeds[-1, -1] = 2

timer = time.time()
seg = random_walker_algorithm_2d(x, seeds_mask=seeds, beta=0.1, offsets=((1, 0), (0, 1)))
print(time.time() - timer)


x = np.random.rand(12, 12, 20)
seeds = np.zeros_like(x).astype(np.int)

seeds[0, 0, 0] = 1
seeds[-1, -1, -1] = 2

timer = time.time()
seg = random_walker_algorithm_3d(x, seeds_mask=seeds, beta=0.1, solving_mode="multi_grid")
print(time.time() - timer)