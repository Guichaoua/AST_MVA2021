import numpy as np


def line_graph(num_vertices, normalize=True):
    W = np.zeros(shape=(num_vertices, num_vertices))
    for i in range(num_vertices-1):
        W[i+1, i] = W[i, i+1] = 1
    D = np.diag(np.sum(W, axis=1))
    L = D - W
    if normalize:
        L = L / np.trace(L) * num_vertices
    return L


def grid_graph(num_vertices, normalize=True):
    W = np.zeros(shape=(num_vertices, num_vertices))
    length = int(np.sqrt(n))
    for i in range(0, length):
        for j in range(0, length):
            if i>0:
                W[i+j*length,(i-1)+j*length] = 1
            if j>0:
                W[i+j*length,i+(j-1)*length] = 1
            if i<m-1:
                W[i+j*length,(i+1)+j*length] = 1
            if j<m-1:
                W[i+j*length,i+(j+1)*length] = 1
    D = np.diag(np.sum(W, axis=1))
    L = D - W
    if normalize:
        L = L / np.trace(L) * num_vertices
    return L


def rbf_random_graph(num_vertices, sigma=0.5, kappa=0.6, normalize=True):
    vertices = np.random.random(size=(num_vertices, 2))
    W = np.zeros((num_vertices, num_vertices))
    for i in range(0, num_vertices):
        for j in range(i+1, num_vertices):
            dist = np.linalg.norm(vertices[i]-vertices[j])
            if dist < kappa:
                W[j, i] = W[i, j] = np.exp(-dist**2/(2*sigma**2))
    D = np.diag(np.sum(W, axis=1))
    L = D - W
    if normalize:
        L = L / np.trace(L) * num_vertices
    return L

def er_random_graph(num_vertices, p=0.2, normalize=True):
    W = np.zeros(shape=(num_vertices, num_vertices))
    for i in range(0, num_vertices):
        for j in range(i+1, num_vertices):
            W[j, i] = W[i, j] = np.random.binomial(n=1, p=p)
    D = np.diag(np.sum(W, axis=1))
    L = D - W
    if normalize:
        L = L / np.trace(L) * num_vertices
    return L
