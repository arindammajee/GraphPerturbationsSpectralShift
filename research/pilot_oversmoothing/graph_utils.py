"""Normalised-Laplacian statistics and sparse helpers (no self-loops unless noted)."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, ArpackNoConvergence


def to_symmetric_simple(A: sp.spmatrix) -> sp.csr_matrix:
    """Undirected, unweighted, no self-loops."""
    A = A.tocsr(copy=True).astype(np.float64)
    A = A + A.T
    A.data = np.ones_like(A.data)
    A.setdiag(0)
    A.eliminate_zeros()
    return A


def degrees(A: sp.spmatrix) -> np.ndarray:
    return np.asarray(A.sum(axis=1)).ravel().astype(np.float64)


def largest_connected_component(A: sp.spmatrix) -> np.ndarray:
    n, labels = sp.csgraph.connected_components(A, directed=False, return_labels=True)
    counts = np.bincount(labels, minlength=n)
    keep = np.where(labels == int(np.argmax(counts)))[0]
    return keep


def subgraph(A: sp.spmatrix, idx: np.ndarray) -> sp.csr_matrix:
    return A[idx][:, idx].tocsr()


def sigma2_and_mu(A: sp.spmatrix) -> tuple[float, float, int]:
    """T1 closed form. Isolates use L_ii = 0, so mu = 1 - n0/n."""
    A = A.tocsr()
    n = A.shape[0]
    d = degrees(A)
    n0 = int((d == 0).sum())
    mu = 1.0 - n0 / n
    acc = 0.0
    coo = A.tocoo()
    for i, j, v in zip(coo.row, coo.col, coo.data):
        if i < j and v != 0:
            acc += 1.0 / (d[i] * d[j])
    fro2 = (n - n0) + 2.0 * acc
    var = fro2 / n - mu * mu
    return float(var), float(mu), n0


def mean_degree(A: sp.spmatrix) -> float:
    n = A.shape[0]
    return float(A.nnz) / n  # already symmetric, nnz = 2|E|, so 2|E|/n


def edge_homophily(A: sp.spmatrix, y: np.ndarray) -> float:
    coo = A.tocoo()
    same = 0
    total = 0
    for i, j, v in zip(coo.row, coo.col, coo.data):
        if i < j and v != 0:
            total += 1
            if y[i] == y[j]:
                same += 1
    if total == 0:
        return float("nan")
    return same / total


def normalised_adjacency(A: sp.spmatrix, self_loops: bool) -> sp.csr_matrix:
    A = A.tocsr(copy=True).astype(np.float64)
    if self_loops:
        A = A.copy()
        A.setdiag(0)
        A = A + sp.eye(A.shape[0], format="csr", dtype=np.float64)
    d = degrees(A)
    d_inv_sqrt = np.zeros_like(d)
    pos = d > 0
    d_inv_sqrt[pos] = 1.0 / np.sqrt(d[pos])
    D = sp.diags(d_inv_sqrt)
    return (D @ A @ D).tocsr()


def normalised_laplacian(A: sp.spmatrix, self_loops: bool = False) -> sp.csr_matrix:
    n = A.shape[0]
    S = normalised_adjacency(A, self_loops=self_loops)
    # Isolates: S row is 0 and we want L_ii = 0 (no self-loops case).
    # With self-loops every node has degree >= 1, L = I - S.
    if self_loops:
        return (sp.eye(n, format="csr", dtype=np.float64) - S).tocsr()
    d = degrees(A)
    diag = (d > 0).astype(np.float64)
    return (sp.diags(diag) - S).tocsr()


def spectral_gap(A: sp.spmatrix, self_loops: bool = False) -> float:
    """lambda_2 of the normalised Laplacian of a (hopefully connected) graph."""
    n = A.shape[0]
    if n <= 2:
        return float("nan")
    S = normalised_adjacency(A, self_loops=self_loops)
    # Largest eigenvalues of S: 1 = 1-lambda_1, next is 1-lambda_2.
    try:
        ev = eigsh(S, k=min(3, n - 2), which="LA", return_eigenvectors=False, maxiter=10000)
        ev = np.sort(ev)[::-1]
        # ev[0] should be ~1
        return float(max(0.0, ev[0] - ev[1]))
    except (ArpackNoConvergence, ValueError, Exception):
        if n <= 4000:
            ev = np.linalg.eigvalsh(S.toarray())
            ev = np.sort(ev)[::-1]
            return float(max(0.0, ev[0] - ev[1]))
        return float("nan")


def dirichlet_energy(X: np.ndarray, L: sp.spmatrix) -> float:
    """(1/(n f)) Tr(X^T L X)."""
    n, f = X.shape
    LX = L @ X
    return float(np.sum(X * LX) / (n * f))
