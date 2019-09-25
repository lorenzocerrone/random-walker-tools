import numpy as np
from skimage.segmentation import watershed
from randomwalkertools.randomwalker_algorithm import random_walker_algorithm_2d, random_walker_algorithm_3d

import time

solver = ["multi_grid"]

all_results = []
np.save("test", np.array(all_results))
for s in solver:
    for i in 2**np.arange(5, 11):
        x = np.random.rand(i, i)
        seeds = np.zeros_like(x).astype(np.int)

        seeds[0, 0] = 1
        seeds[-1, -1] = 2

        timer = time.time()
        for _ in range(3):
            seg = random_walker_algorithm_2d(x, seeds_mask=seeds, beta=10, solving_mode=s)
        timer2d = (time.time() - timer) / 3

        ii = int((i * i)**(1/3))
        x = np.random.rand(ii, ii, ii)
        seeds = np.zeros_like(x).astype(np.int)

        seeds[0, 0, 0] = 1
        seeds[-1, -1, -1] = 2

        timer = time.time()
        for _ in range(3):
            seg = random_walker_algorithm_3d(x, seeds_mask=seeds, beta=10, solving_mode=s)
        timer3d = (time.time() - timer) / 3

        timer = time.time()
        for _ in range(3):
            watershed(-x, markers=seeds)
        timer3dw = (time.time() - timer) / 3

        print(s, i, timer2d, ii, timer3d, ii, timer3dw)
        all_results.append([s, i, timer2d, ii, timer3d])


np.save("test", np.array(all_results))