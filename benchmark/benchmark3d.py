import numpy as np
from randomwalkertools.randomwalker_algorithm import random_walker_algorithm_3d
import time


all_sizes, beta = 2**np.arange(8, 12), 1
print(all_sizes)
t_size = {}
for size in all_sizes:
    t_solver = {}
    for solver in ["cg_mg"]:#["direct", "cg_mg", "cg", "cuda"]:
        t_seeds = {}
        for n_seeds in [2, 4, 6, 12, 24]:
            size = int(size)
            x = np.ones((size, size, size))
            lap, i = 0, 0
            for i in range(1, 100):

                seeds = np.zeros_like(x).astype(np.int)

                s = np.random.choice(x.size, size=n_seeds, replace=False)
                for ii, _s in enumerate(s):
                    _xx, _yy, _zz = np.unravel_index(_s, x.shape)
                    seeds[_xx, _yy, _zz] = ii + 1

                timer = time.time()
                try:
                    random_walker_algorithm_3d(x,
                                               seeds_mask=seeds,
                                               beta=beta,
                                               solving_mode=solver,
                                               )
                    lap += time.time() - timer

                except:
                    lap = np.nan
                    break

                if lap > 1:
                    break
            print(f"size: {size} {solver} time: {lap / i:.3f} time/instance: {lap / (i * n_seeds):.3f}")

            t_seeds[n_seeds] = (i, lap)
        t_solver[solver] = t_seeds
    t_size[size] = t_solver
    np.save('3d.npy', t_size)
