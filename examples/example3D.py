import numpy as np

from rwtools.randomwalker_algorithm import random_walker_algorithm_3d
from rwtools.graphtools import solvers
import time
import matplotlib.pyplot as plt


shape = (32, 32, 32)
n = 3

offsets = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
x = np.ones((shape[0], shape[1], shape[2]))
seeds = np.zeros_like(x).astype(np.int)
np.random.seed(0)
s = np.random.choice(np.arange(shape[0] * shape[1] * shape[2]), size=n)
sx, sy, sz = np.unravel_index(s, shape=shape)
seeds[sx, sy, sz] = np.arange(1, n+1)

for solver in solvers.keys():
    timer = time.time()
    _x = random_walker_algorithm_3d(x, seeds_mask=seeds, beta=10, offsets=offsets, solving_mode=solver,
                                    return_prob=False, divide_by_std=False)
    plt.imshow(_x[shape[0]//2], cmap="prism", interpolation="nearest")
    plt.axis("off")
    plt.show()
    print(f"{solver}, timer: {time.time() - timer}")