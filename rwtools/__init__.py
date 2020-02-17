from randomwalkertools.graphtools.solvers import direct_solver, solve_cg_mg, solve_cg, solve_gpu
from randomwalkertools.randomwalker_algorithm import random_walker_algorithm_2d, random_walker_algorithm_3d
from randomwalkertools.differentiable_random_walker_algorithm import DifferentiableRandomWalker2D

solvers = {"direct": direct_solver,
          "cg_mg": solve_cg_mg,
          "cg": solve_cg,
          "cuda": solve_gpu}
