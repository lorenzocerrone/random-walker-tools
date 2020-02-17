from scipy.sparse.linalg import eigsh, eigs, lobpcg
from scipy.sparse import diags, eye
import pyamg
import numpy as np
import numpy as np
from randomwalkertools.graphtools.graphtools import image2edges, make2d_lattice_graph, graph2adjacency, adjacency2laplacian
import time
import matplotlib.pyplot as plt
N = 20
nx = 40

ex = np.random.rand(N*N, nx)
x = np.zeros((N, N))
x[:, N//2] = 1

graph = make2d_lattice_graph((N, N))

edges = image2edges(x, graph, beta=1)

A = graph2adjacency(graph, edges)
L = adjacency2laplacian(A)

timer = time.time()


ml = pyamg.ruge_stuben_solver(L)
#ml = pyamg.coarse_grid_solver(L)
#ml = pyamg.smoothed_aggregation_solver(L, diagonal_dominance=True)
M = ml.aspreconditioner(cycle='V')
lamb, q = lobpcg(L, ex, largest=False, M=M)
print(time.time() - timer)


Lamb = diags(1/lamb[1:])
L_plus = q[:, 1:].dot(Lamb.dot(q[:, 1:].T))
#print(L_plus)
# q[:, 0].reshape(N*N, 1).dot(q[:, 0].reshape(N*N, 1).T)
I = L.dot(L_plus) + q[:, 0].reshape(N*N, 1).dot(q[:, 0].reshape(N*N, 1).T)
print()


plt.imshow(I)
plt.colorbar()
plt.show()

plt.imshow(L.toarray())
plt.colorbar()
plt.show()


Lamb = diags(lamb[1:])
L_plus = q[:, 1:].dot(Lamb.dot(q[:, 1:].T))
plt.imshow(L_plus)
plt.colorbar()
plt.show()

