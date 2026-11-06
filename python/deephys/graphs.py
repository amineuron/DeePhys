"""Exact-neighbor fuzzy graph and clamped label propagation from DeePhys."""

from dataclasses import dataclass
import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors


def fuzzy_knn_graph(X, k=50):
    X = np.asarray(X, float)
    n = len(X)
    k = min(int(k), n - 1)
    if X.ndim != 2 or not np.isfinite(X).all():
        raise ValueError("finite 2-D feature matrix required")
    if k < 1:
        return sparse.csr_matrix((n, n))
    search = NearestNeighbors(n_neighbors=k + 1, algorithm="brute", n_jobs=1).fit(X)
    d, idx = search.kneighbors(X)
    neighbors = np.empty((n, k), int)
    distances = np.empty((n, k), float)
    for i in range(n):
        order = np.lexsort((idx[i], d[i]))
        ii = idx[i, order]
        dd = d[i, order]
        keep = ii != i
        if keep.all():
            keep[-1] = False
        neighbors[i] = ii[keep][:k]
        distances[i] = dd[keep][:k]
    positive = np.where(distances > 0, distances, np.inf)
    rho = positive.min(axis=1)
    rho[~np.isfinite(rho)] = 0
    dr = np.maximum(0, distances - rho[:, None])
    target = np.log2(k)
    lo = np.zeros(n)
    hi = np.full(n, np.inf)
    sigma = np.ones(n)
    for _ in range(64):
        big = np.exp(-dr / sigma[:, None]).sum(axis=1) > target
        hi[big] = sigma[big]
        lo[~big] = sigma[~big]
        finite = np.isfinite(hi)
        sigma[finite] = (lo[finite] + hi[finite]) / 2
        sigma[~finite] *= 2
    sigma = np.maximum(sigma, 0.001 * distances.mean(axis=1))
    sigma[sigma <= 0] = np.finfo(float).eps
    vals = np.exp(-dr / sigma[:, None])
    A = sparse.csr_matrix(
        (vals.ravel(), (np.repeat(np.arange(n), k), neighbors.ravel())), shape=(n, n)
    )
    W = A + A.T - A.multiply(A.T)
    W.setdiag(0)
    W.eliminate_zeros()
    return W.tocsr()


@dataclass
class PropagationResult:
    scores: np.ndarray
    unreached: np.ndarray
    converged: bool
    iterations: int


def propagate_labels(graph, seed_indices, seed_labels, max_iter=1000, tol=1e-6):
    """Port of CellTypeClassifier.propagateLabels; 1=E, 2=I.

    These absorption scores are not calibrated cell-identity probabilities.
    Unseeded/disconnected components stay unlabelled in the public classifier.
    """
    if max_iter < 1 or tol <= 0:
        raise ValueError("positive iteration limit and tolerance required")
    G = sparse.csr_matrix(graph, dtype=float)
    n = G.shape[0]
    if G.shape != (n, n) or not np.isfinite(G.data).all() or np.any(G.data < 0):
        raise ValueError("invalid graph")
    idx = np.asarray(seed_indices, int)
    labels = np.asarray(seed_labels, int)
    if (
        len(idx) != len(labels)
        or len(np.unique(idx)) != len(idx)
        or np.any((idx < 0) | (idx >= n))
        or np.any(~np.isin(labels, [1, 2]))
    ):
        raise ValueError("unique valid seed indices and labels 1/2 required")
    rows = np.asarray(G.sum(axis=1)).ravel()
    rows[rows == 0] = 1
    W = sparse.diags(1 / rows) @ G
    clamp = np.c_[labels == 1, labels == 2].astype(float)
    F = np.full((n, 2), 0.5)
    F[idx] = clamp
    converged = False
    for iteration in range(1, max_iter + 1):
        new = W @ F
        r = new.sum(axis=1)
        r[r == 0] = 1
        new /= r[:, None]
        new[idx] = clamp
        change = np.max(np.abs(new - F)) if n else 0
        F = new
        if change < tol:
            converged = True
            break
    unreached = np.all(np.abs(F - 0.5) < 1e-12, axis=1) | np.all(F == 0, axis=1)
    unreached[idx] = False
    ns = clamp.sum(axis=0)
    if np.all(ns > 0) and ns[0] != ns[1]:
        F /= ns
        r = F.sum(axis=1)
        r[r == 0] = 1
        F /= r[:, None]
    return PropagationResult(F, unreached, converged, iteration)


def global_normalize(X):
    """Native no-feature-selection global z-score and maximum-absolute scale."""
    X = np.asarray(X, float)
    mean = X.mean(axis=0)
    scale = X.std(axis=0, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        Z = (X - mean) / scale
    keep = np.isfinite(Z).all(axis=0)
    Z = Z[:, keep]
    maxabs = np.max(np.abs(Z), axis=0) if Z.shape[1] else np.array([])
    maxabs[maxabs == 0] = 1
    return Z / maxabs, dict(mean=mean, std=scale, keep=keep, maxabs=maxabs)


def weight_feature_groups(X, names, acg_weight=1.0, waveform_weight=1.0):
    """Native dimension-normalized ACG/waveform contribution to graph distances."""
    result = np.array(X, float, copy=True)
    acg = np.array([n.startswith(("FullACG", "ACG")) for n in names])
    wf = np.array([n.startswith(("Waveform", "ReferenceWaveform")) for n in names])
    for mask, weight in [(acg, acg_weight), (wf, waveform_weight)]:
        if mask.any():
            result[:, mask] *= np.sqrt(weight / mask.sum())
    return result


def leiden_communities(graph, *, seed=42, resolution=1.0, restarts=30):
    """Python Leiden on the exact graph; stochastic partitions need not match MATLAB."""
    import igraph as ig
    import leidenalg

    W = sparse.triu(sparse.csr_matrix(graph), k=1).tocoo()
    g = ig.Graph(
        n=graph.shape[0],
        edges=list(zip(W.row.tolist(), W.col.tolist())),
        directed=False,
    )
    g.es["weight"] = W.data.tolist()
    best = None
    partitions = []
    scores = []
    for i in range(restarts):
        part = leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            weights="weight",
            resolution_parameter=resolution,
            seed=seed + i,
        )
        partitions.append(np.asarray(part.membership) + 1)
        scores.append(part.quality())
        if best is None or part.quality() > best.quality():
            best = part
    return np.asarray(best.membership) + 1, np.asarray(partitions), np.asarray(scores)
