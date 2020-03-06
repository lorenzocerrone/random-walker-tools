import numpy as np

from rwtools.randomwalker_algorithm import random_walker_algorithm_3d
import time
import matplotlib.pyplot as plt

shape = (16, 64, 64)
np.random.seed(0)
x = np.random.rand(shape[0], shape[1], shape[2])
seeds = np.zeros_like(x).astype(np.int)

#seeds[0, 0, 0] = 1
#seeds[-1, -1, -1] = 2
n = 80

#for n in [2, 4, 8, 16, 32, 64, 128]:
for n in [20]:
    print(n)
    s = np.random.choice(np.arange(np.prod(shape)), size=n)
    sx, sy, sz = np.unravel_index(s, shape=shape)
    seeds[sx, sy, sz] = np.arange(1, n+1)

    timer = time.time()
    _x = random_walker_algorithm_3d(x, seeds_mask=seeds, beta=1, solving_mode="cg", return_prob=False)

    plt.imshow(_x[shape[0]//2])
    plt.show()
    timer = (time.time() - timer)
    timer_s = timer / n
    print("timer cg: ", timer, timer_s)


    timer = time.time()
    _x = random_walker_algorithm_3d(x, seeds_mask=seeds, beta=1, solving_mode="mp_cg", return_prob=False)

    plt.imshow(_x[shape[0]//2])
    plt.show()
    timer = (time.time() - timer)
    timer_s = timer / n
    print("timer mp_cg: ", timer, timer_s)
