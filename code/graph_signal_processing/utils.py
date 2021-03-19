import numpy as np

"""
Toutes ces formules viennent d'ici: https://github.com/statsmodels/statsmodels/pull/4143/files
"""

def L_(n):
    k = sum(u(int(np.trunc((j)*n + i - 0.5* (j + 1)*(j))), n*(n+1)//2)[:, None].dot(vec(E(i, j, n, n))[None, :])
             for i in range(n) for j in range(i+1))
    return k

def K(n):
     k = sum(np.kron(E(i, j, n, n), E(i, j, n, n).T)
             for i in range(n) for j in range(n))
     return k

def u(i, n):
     u_ = np.zeros(n, np.int64)
     u_[i] = 1
     return u_

def E(i, j, nr, nc):
     x = np.zeros((nr, nc), np.int64)
     x[i, j] = 1
     return x

def vec(x):
     return np.ravel(x, order='F')

def duplication_matrix(n):
     l = L_(n)
     ltl = l.T.dot(l)
     k = K(n)
     d = l.T + k.dot(l.T) - ltl.dot(k).dot(l.T)
     return d

def vech(x):
     if x.ndim == 2:
         idx = np.triu_indices_from(x.T)
         return x.T[idx[0], idx[1]] #, x[idx[1], idx[0]]
     elif x.ndim > 2:
         #try ravel last two indices
         #idx = np.triu_indices(x.shape[-2::-1])
         n_rows, n_cols = x.shape[-2:]
         idr, idc = np.array([[i, j] for j in range(n_cols)
                                     for i in range(j, n_rows)]).T
         return x[..., idr, idc]
