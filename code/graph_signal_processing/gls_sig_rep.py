
import numpy as np
from cvxopt import solvers, matrix

from utils import duplication_matrix, vech, E

def objective_function(laplacian, y, alpha, beta):
    return(alpha * np.trace(y @ laplacian @ y.T) + beta * np.linalg.norm(laplacian, ord='fro')**2)

def optimization_problem_wrt_L(y, alpha, beta):
    n = y.shape[1]
    m = n * (n+1) // 2
    d = duplication_matrix(n)
    q = alpha * d.T @ (y.T @ y).flatten()
    q = matrix(q, tc='d')
    P = matrix(2 * beta * d.T @ d, tc='d')
    # contraintes égalité
    A = []
    b = []
    ## contrainte de trace
    c = np.zeros((m, ))
    c[0] = 1
    index = 0
    for i in range(n-1):
        index += (n-i)
        c[index] = 1
    A.append(np.copy(c))
    b.append(n)
    ## contrainte 0 vp de vc 1
    for i in range(n):
        c = np.zeros((n, n))
        c[i, :] = 1
        c = vech(np.sign(c + c.T))
        A.append(np.copy(c))
        b.append(0)
    A = matrix(np.array(A), tc='d')
    b = matrix(np.array(b), tc='d')
    # contraintes inégalité
    G = []
    h = []
    for i in range(n):
        for j in range(i+1, n):
            c = vech(E(i, j, n, n) + E(j, i, n, n))
            G.append(np.copy(c))
            h.append(0)
    G = matrix(np.array(G), tc='d')
    h = matrix(np.array(h), tc='d')
    sol = np.ravel(solvers.qp(P, q, G, h, A, b)['x'])
    return d.dot(sol).reshape((n ,n))


def GLSigRep(samples, alpha, beta, n_step=1000, log_step=None):
    num_samples, num_vertices = samples.shape
    y = np.copy(samples)
    for i in range(n_step):
    # update L
        laplacian = optimization_problem_wrt_L(y, alpha, beta)
    # update H
        y = samples @ np.linalg.inv(np.eye(num_vertices)+alpha*laplacian).T
    # log the objective value
        if log_step is not None and i % log_step == 0:
            obj = objective_function(laplacian, y, alpha, beta)
            print(obj)
        try:
            if np.allclose(old_laplacian, laplacian) and np.allclose(old_y, y):
                break
        except:
            pass
        old_laplacian = laplacian
        old_y = y
    return laplacian, y
