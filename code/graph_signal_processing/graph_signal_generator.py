import numpy as np

def computing_dynamic_matrix(laplacian, diffusion_factors):
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    for i, tau in enumerate(diffusion_factors):
        if i == 0:
            D = eigenvectors @ np.diag(np.exp(-tau * eigenvalues)) @ eigenvectors.T
        else:
            D = np.concatenate([D, eigenvectors @ np.diag(np.exp(-tau * eigenvalues)) @ eigenvectors.T], axis=1)
    return D

def heat_signal_generator(laplacian, diffusion_factors, n_samples):
    num_vertices = laplacian.shape[0]
    num_diffusions = len(diffusion_factors)
    dynamic_matrix = computing_dynamic_matrix(laplacian, diffusion_factors)
    samples = np.zeros(shape=(n_samples, num_vertices))
    activations = np.zeros(shape=(n_samples, num_vertices*num_diffusions))
    for i in range(n_samples):
        # pour chaque échantillon, nous choisissons des activations au hasard
        indices = np.random.choice(np.arange(num_vertices * num_diffusions), 3, replace=False)
        # nous choississons des températures initiales gaussiennes
        init_heat = np.random.normal(size=(3,))
        activations[i, indices] = init_heat
        # nous calculons finalement le signal sur le graph
        samples[i] = np.ravel(dynamic_matrix @ activations[i])
    return samples, activations
