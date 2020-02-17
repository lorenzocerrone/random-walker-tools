from randomwalkertools.graphtools.solvers import direct_solver, solve_cg_mg, solve_cg, solve_gpu
from .randomwalker_algorithm import random_walker_algorithm_2d, random_walker_algorithm_3d

solvers = {"direct": direct_solver,
          "cg_mg": solve_cg_mg,
          "cg": solve_cg,
          "cuda": solve_gpu}