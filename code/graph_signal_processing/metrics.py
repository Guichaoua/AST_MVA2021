
import numpy as np

from sklearn.metrics import f1_score

def F1_score(true_L, pred_L, threshold=1e-1):
    assert true_L.shape == pred_L.shape
    n = true_L.shape[0]
    true_edges = []
    pred_edges = []
    for i in range(0, n):
        for j in range(i+1, n):
            true_edges.append(int(np.sign(-true_L[i, j])))
            if pred_L[i, j] > - threshold:
                pred_edges.append(0)
            else:
                pred_edges.append(1)
    return f1_score(true_edges, pred_edges)
