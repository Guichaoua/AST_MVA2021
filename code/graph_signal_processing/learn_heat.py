
import numpy as np
from cvxopt import solvers, matrix

from graph_signal_generator import computing_dynamic_matrix
from graph_generator import rbf_random_graph

def reconstruction_loss(samples,
                        laplacian,
                        activations,
                        diffusion_factors):
    D = computing_dynamic_matrix(laplacian, diffusion_factors)
    return np.linalg.norm(samples-activations.dot(D.T), ord='fro')**2

def objective_function(samples,
                      laplacian,
                      activations,
                      diffusion_factors,
                      alpha,
                      beta):
    """
    alpha : L1 regularization of H (activations)
    beta : L2 regularization of L (laplacian)
    """
    obj = reconstruction_loss(samples, laplacian, activations, diffusion_factors)
    obj += beta * np.linalg.norm(laplacian, ord='fro')
    obj += alpha * np.linalg.norm(np.ravel(activations), ord=1)
    return obj

def grad_tr_A_exp_L(eigenvalues, eigenvectors, A):
    """grad of tr(Aexp(L)) wrt L when L is symmetric"""
    num_vertices = len(eigenvalues)
    B = np.zeros(shape=(num_vertices, num_vertices))
    for i in range(num_vertices):
        for j in range(i, num_vertices):
            if eigenvalues[i] == eigenvalues[j]:
                B[i, i] = np.exp(eigenvalues[i])
            else:
                B[i, j] = (np.exp(eigenvalues[i]) - np.exp(eigenvalues[j])) / (eigenvalues[i] - eigenvalues[j])
                B[j, i] = B[i, j]
    grad = eigenvectors @ ((eigenvectors.T @ A.T @ eigenvectors) * B) @ eigenvectors.T
    return grad

def grad_wrt_L(samples, laplacian, activations, diffusion_factors):
    """gradient of the reconstruction error with respect to L"""
    num_vertices = samples.shape[1]
    grad = np.zeros(shape=(num_vertices, num_vertices))
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    for i, tau in enumerate(diffusion_factors):
        H_tau = activations[:, num_vertices*i:num_vertices*(i+1)]
        A = H_tau.T @ samples
        grad -= 2 * (- tau * grad_tr_A_exp_L(- tau * eigenvalues, eigenvectors, A))
        for j, tau_bis in enumerate(diffusion_factors):
            H_tau_bis = activations[:, num_vertices*j:num_vertices*(j+1)]
            A = H_tau_bis.T @ H_tau
            grad += - ((tau+tau_bis) *  grad_tr_A_exp_L(-(tau+tau_bis)*eigenvalues, eigenvectors, A))
    return grad

def grad_wrt_tau(samples,
                 laplacian,
                 activations,
                 diffusion_factors):
    N = num_vertices = samples.shape[1]
    grad = np.zeros(shape=len(diffusion_factors))
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    for i, tau in enumerate(diffusion_factors):
        grad[i] += 2 * np.trace(activations[:, N*i:N*(i+1)].T @ samples @ laplacian @ \
                                eigenvectors @ np.diag(np.exp(- tau * eigenvalues)) @ eigenvectors.T)
        for j, tau_bis in enumerate(diffusion_factors):
            grad[i] -= 2 * np.trace(activations[:, N*j:N*(j+1)].T @ activations[:, N*i:N*(i+1)] @ laplacian @ \
                                    eigenvectors @ np.diag(np.exp(-(tau + tau_bis) * eigenvalues)) @ eigenvectors.T)
    return grad

def op_proximal_norm1(z, c, alpha):
    """proximity operator of x -> alpha * ||x||_1"""
    return np.sign(z) * np.maximum(np.abs(z)-alpha/c, np.zeros(shape=len(z)))

def cond_descent_lemma(samples,
                        new_laplacian,
                        laplacian,
                        activations,
                        diffusion_factors,
                        C2):
    left_side = reconstruction_loss(samples, new_laplacian, activations, diffusion_factors)
    right_side = reconstruction_loss(samples, laplacian, activations, diffusion_factors)
    grad = grad_wrt_L(samples, laplacian, activations, diffusion_factors)
    right_side +=  np.dot(grad.flatten(), (new_laplacian-laplacian).flatten())
    right_side += C2 / 2 * np.linalg.norm(new_laplacian-laplacian, ord='fro')**2
    return left_side <= right_side

def proximal_min_prob_L(laplacian, grad, d, beta):
    num_vertices = laplacian.shape[0]
    q = matrix(grad.flatten() - d * laplacian.flatten(), tc='d')
    P = matrix((d + 2 * beta * d) * np.eye(num_vertices**2), tc='d')
    A = []
    b = []
    # contrainte trace
    c = np.zeros((num_vertices**2,))
    for i in range(num_vertices):
        c[i*num_vertices+i] = 1
    A.append(np.copy(c))
    b.append(num_vertices)
    # contrainte symétrie
    for i in range(num_vertices):
        for j in range(i+1, num_vertices):
            c = np.zeros((num_vertices**2,))
            c[i*num_vertices+j] = 1
            c[j*num_vertices+i] = -1
            A.append(np.copy(c))
            b.append(0)
    # contrainte 1 vecteur propre pour la vp 0
    for i in range(num_vertices):
        c = np.zeros((num_vertices**2,))
        c[num_vertices*i:num_vertices*(i+1)] = 1
        A.append(np.copy(c))
        b.append(0)
    A = matrix(np.array(A), tc='d')
    b = matrix(np.array(b), tc='d')
    G = []
    h = []
    # contraintes termes négatifs
    for i in range(num_vertices):
        for j in range(i+1, num_vertices):
            c = np.zeros((num_vertices**2,))
            c[i*num_vertices+j] = 1
            G.append(np.copy(c))
            h.append(0)
    G = matrix(np.array(G), tc='d')
    h = matrix(np.array(h), tc='d')
    sol = np.ravel(solvers.qp(P, q, G, h, A, b)['x'])
    return sol.reshape((num_vertices, num_vertices))

def lipschitz_constant_wrt_H(dynamic_matrix):
    return np.linalg.norm(2 * dynamic_matrix.T @ dynamic_matrix, ord='fro')

def lipschitz_constant_wrt_tau(samples,
                               laplacian,
                               activations,
                               diffusion_factors):
    num_vertices = laplacian.shape[0]
    res = np.zeros(shape=(len(diffusion_factors,)))
    for i, _ in enumerate(diffusion_factors):
        res[i] += 2 * np.linalg.norm(activations[:, num_vertices*i:num_vertices*(i+1)]) * np.linalg.norm(samples)
        for j, _ in enumerate(diffusion_factors):
            res[i] += 4 * np.linalg.norm(activations[:, num_vertices*i:num_vertices*(i+1)]) * \
            np.linalg.norm(activations[:, num_vertices*j:num_vertices*(j+1)])
    return np.linalg.norm(laplacian@laplacian, ord=2)**2 * np.max(res)


def learn_heat(samples, n_step=20,  alpha=1e-6, beta=1e-2,
               gamma1=1.5, gamma2=1.5, gamma3=1.5, s=2,
               log_step=10, true_L=None, true_H=None, true_tau=None,
               tol=1e-2):
    num_samples, num_vertices = samples.shape
    # initialisation
    diffusion_factors = np.sort(np.array([2+np.random.random() for _ in range(s)]))
    diffusion_factors = np.array(true_tau)
    laplacian = rbf_random_graph(num_vertices)
    dynamic = computing_dynamic_matrix(laplacian, diffusion_factors)
    activations = np.random.normal(size=(num_samples, s*num_vertices))
    # updating loops
    old_obj = np.inf
    for step in range(n_step):

        obj = objective_function(samples,
                                   laplacian,
                                   activations,
                                   diffusion_factors,
                                   alpha,
                                   beta)
        if np.abs(obj-old_obj) < tol:
            break
        old_obj = obj

        if step % log_step == 0 and true_L is not None:

            diff_w_gt_L = np.linalg.norm(true_L - laplacian, ord='fro')
            diff_w_gt_H = np.linalg.norm(true_H - activations, ord='fro')
            diff_w_gt_tau = np.linalg.norm(true_tau - diffusion_factors, ord=2)
            print("step {0} : obj = {1:2f}, diff gt L = {2:4f}, diff gt H = {3:4f},\
             diff gt tau = {4:.2f}".format(step, obj, diff_w_gt_L, diff_w_gt_H, diff_w_gt_tau))
        # updating H
        C1 = lipschitz_constant_wrt_H(dynamic)
        c_ = gamma1 * C1
        for j in range(num_samples):
            z = activations[j] - 1 / c_ * (-2 * dynamic.T @ (samples[j] - dynamic @ activations[j]))
            activations[j] = op_proximal_norm1(z, c_, alpha)
        # updating L
        eta = 2
        C2_init = 1
        descent_lemma_satisfied = False
        k = 0
        ### backtracking line search (for the constant C2)
        while not(descent_lemma_satisfied):
            C2 = C2_init * eta**k
            d_ = gamma2 * C2
            grad_L = grad_wrt_L(samples, laplacian, activations, diffusion_factors)
            new_laplacian = proximal_min_prob_L(laplacian, grad_L, d_, beta)
            k += 1
            descent_lemma_satisfied = cond_descent_lemma(samples,
                                                        new_laplacian,
                                                        laplacian,
                                                        activations,
                                                        diffusion_factors,
                                                        C2)
        laplacian = new_laplacian
        dynamic = computing_dynamic_matrix(laplacian, diffusion_factors)
        # updating tau
        #e_ = gamma3 * lipschitz_constant_wrt_tau(samples, laplacian, activations, diffusion_factors)
        #grad_tau = grad_wrt_tau(samples, laplacian, activations, diffusion_factors)
        #diffusion_factors = np.maximum(- 1 / e_ * (grad_tau - e_ * diffusion_factors),
        #                               np.zeros(shape=(s,)))
        #dynamic = computing_dynamic_matrix(laplacian, diffusion_factors)

    return laplacian, activations, diffusion_factors
