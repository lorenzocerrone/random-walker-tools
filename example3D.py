import numpy as np
from randomwalkertools.randomwalker_algorithm import random_walker_algorithm_2d, random_walker_algorithm_3d
import time
import matplotlib.pyplot as plt

x = np.random.rand(100, 100, 100)
seeds = np.zeros_like(x).astype(np.int)

seeds[0, 0, 0] = 1
seeds[-1, -1, -1] = 2

timer = time.time()
prob = random_walker_algorithm_3d(x, seeds_mask=seeds, beta=1, solving_mode="multi_grid", return_prob=True)
print(time.time() - timer)
print(prob.shape)

print(prob.sum(), np.prod(x.shape))