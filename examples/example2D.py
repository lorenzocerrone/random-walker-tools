import numpy as np

from rwtools.randomwalker_algorithm import random_walker_algorithm_2d
import time
import matplotlib.pyplot as plt

#shape = (100, 100)
shape = (500, 500)
offsets = ((1, 0), (0, 1))
x = np.ones((shape[0], shape[1]))
x[shape[0]//2, :] = 1

seeds = np.zeros_like(x).astype(np.int)
n = 3

np.random.seed(0)
s = np.random.choice(np.arange(np.prod(shape)), size=n)
sx, sy = np.unravel_index(s, shape=shape)
seeds[sx, sy] = np.arange(1, n+1)

timer = time.time()
_x = random_walker_algorithm_2d(x, seeds_mask=seeds, beta=10, offsets=offsets, solving_mode="mp_cg",
                                return_prob=False, divide_by_std=False)
plt.imshow(_x, cmap="prism")
plt.show()
print("timer: ", time.time() - timer)

timer = time.time()
_x = random_walker_algorithm_2d(x, seeds_mask=seeds, beta=10, offsets=offsets, solving_mode="mp_cg_ichol",
                                return_prob=False, divide_by_std=False)
plt.imshow(_x, cmap="prism")
plt.show()
print("timer: ", time.time() - timer)


timer = time.time()
_x = random_walker_algorithm_2d(x, seeds_mask=seeds, beta=10, offsets=offsets, solving_mode="cg",
                                return_prob=False, divide_by_std=False)
plt.imshow(_x, cmap="prism")
plt.show()
print("timer: ", time.time() - timer)

timer = time.time()
_x = random_walker_algorithm_2d(x, seeds_mask=seeds, beta=10, offsets=offsets, solving_mode="cg_mg",
                                return_prob=False, divide_by_std=False)
plt.imshow(_x)
plt.show()
print("timer: ", time.time() - timer)

_x = random_walker_algorithm_2d(x, seeds_mask=seeds, beta=1, offsets=offsets, solving_mode="direct",
                                return_prob=False, divide_by_std=False)
#plt.imshow(_x)
#plt.show()
print("timer: ", time.time() - timer)


