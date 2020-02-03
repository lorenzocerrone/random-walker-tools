import numpy as np
from randomwalkertools.randomwalker_algorithm import random_walker_algorithm_2d, random_walker_algorithm_3d
import time

all_sizes = 2**np.arange(6, 10)
all_sizes = np.sqrt(np.arange(1, 1000, 2) * 64 * 64)

all_times = []
for size in all_sizes:
    size = int(size)
    x = np.ones((size, size))
    seeds = np.zeros_like(x).astype(np.int)

    seeds[0, 0] = 1
    seeds[-1, -1] = 2

    timer = time.time()

    random_walker_algorithm_2d(x, seeds_mask=seeds, beta=0.1, offsets=((1, 0), (0, 1)))

    print(size, size*size, time.time() - timer)
    all_times.append([size, time.time() - timer])

    np.save("2d", np.array(all_times))
