import numpy as np

from rwtools.randomwalker_algorithm import random_walker_algorithm_2d
from rwtools.graphtools import solvers
import time
import matplotlib.pyplot as plt

shape = (100, 100)
n = 300

offsets = ((1, 0), (0, 1))
x = np.ones((shape[0], shape[1]))
seeds = np.zeros_like(x).astype(np.int)
np.random.seed(0)
s = np.random.choice(np.arange(shape[0] * shape[1]), size=n)
sx, sy = np.unravel_index(s, shape=shape)
seeds[sx, sy] = np.arange(1, n+1)

for solver in solvers.keys():
    timer = time.time()
    _x = random_walker_algorithm_2d(x, seeds_mask=seeds, beta=10, offsets=offsets, solving_mode="sp_cg",
                                    return_prob=False, divide_by_std=False)
    plt.imshow(_x, cmap="prism", interpolation="nearest")
    plt.show()
    print(f"{solver}, timer: {time.time() - timer}")
