"""Feature propagation, Dirichlet energy, and pairwise cosine similarity."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LinearRegression

from graph_utils import dirichlet_energy, normalised_adjacency, normalised_laplacian


def mean_pairwise_cosine(X: np.ndarray, rng: np.random.Generator, n_pairs: int = 8000) -> tuple[float, float]:
    """Return (mean cosine similarity, mean cosine distance = 1-sim) over sampled pairs."""
    n = X.shape[0]
    if n < 2:
        return float("nan"), float("nan")
    # row-normalise for cosine
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    U = X / nrm
    if n * (n - 1) / 2 <= n_pairs:
        # all pairs
        sim = U @ U.T
        iu = np.triu_indices(n, k=1)
        vals = sim[iu]
        mean_sim = float(vals.mean())
        return mean_sim, float(1.0 - mean_sim)
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    # resample collisions
    mask = i == j
    while mask.any():
        j[mask] = rng.integers(0, n, size=int(mask.sum()))
        mask = i == j
    sims = np.sum(U[i] * U[j], axis=1)
    mean_sim = float(np.mean(sims))
    return mean_sim, float(1.0 - mean_sim)


def propagate_curve(
    A: sp.spmatrix,
    X: np.ndarray,
    ks: list[int],
    self_loops: bool,
    rng: np.random.Generator,
    n_pairs: int = 8000,
) -> dict:
    S = normalised_adjacency(A, self_loops=self_loops)
    L = normalised_laplacian(A, self_loops=self_loops)
    H = np.asarray(X, dtype=np.float64)
    k_max = max(ks)
    want = set(ks)
    records = []
    # k=0
    if 0 in want:
        e0 = dirichlet_energy(H, L)
        sim, mad = mean_pairwise_cosine(H, rng, n_pairs)
        records.append({"k": 0, "dirichlet": e0, "mean_cos": sim, "mad": mad})
    for k in range(1, k_max + 1):
        H = S @ H
        if k in want:
            e = dirichlet_energy(H, L)
            sim, mad = mean_pairwise_cosine(H, rng, n_pairs)
            records.append({"k": k, "dirichlet": e, "mean_cos": sim, "mad": mad})
    # Decay rate from log Dirichlet energy vs k, using k>=1 while energy stays positive.
    ks_fit, logs = [], []
    e1 = None
    for r in records:
        if r["k"] < 1:
            continue
        if e1 is None:
            e1 = r["dirichlet"]
        if r["dirichlet"] <= 1e-16 or (e1 is not None and e1 > 0 and r["dirichlet"] / e1 < 1e-8):
            break
        if r["dirichlet"] > 0:
            ks_fit.append(r["k"])
            logs.append(np.log(r["dirichlet"]))
    if len(ks_fit) >= 3:
        reg = LinearRegression()
        reg.fit(np.array(ks_fit).reshape(-1, 1), np.array(logs))
        alpha = float(-reg.coef_[0])  # energy ~ C * exp(-alpha k)
        r2 = float(reg.score(np.array(ks_fit).reshape(-1, 1), np.array(logs)))
        n_fit = len(ks_fit)
    else:
        alpha, r2, n_fit = float("nan"), float("nan"), len(ks_fit)
    # half life relative to k=1
    half_k = None
    e_ref = None
    for r in records:
        if r["k"] == 1:
            e_ref = r["dirichlet"]
    if e_ref and e_ref > 0:
        for r in records:
            if r["k"] >= 1 and r["dirichlet"] <= 0.5 * e_ref:
                half_k = r["k"]
                break
    by_k = {r["k"]: r for r in records}
    return {
        "records": records,
        "alpha": alpha,
        "alpha_r2": r2,
        "n_fit": n_fit,
        "half_life_k": half_k,
        "dirichlet_k1": by_k.get(1, {}).get("dirichlet"),
        "dirichlet_k8": by_k.get(8, {}).get("dirichlet"),
        "dirichlet_k16": by_k.get(16, {}).get("dirichlet"),
        "dirichlet_k32": by_k.get(32, {}).get("dirichlet"),
        "dirichlet_k64": by_k.get(64, {}).get("dirichlet"),
        "mad_k1": by_k.get(1, {}).get("mad"),
        "mad_k16": by_k.get(16, {}).get("mad"),
        "mad_k64": by_k.get(64, {}).get("mad"),
        "cos_k1": by_k.get(1, {}).get("mean_cos"),
        "cos_k16": by_k.get(16, {}).get("mean_cos"),
        "cos_k64": by_k.get(64, {}).get("mean_cos"),
        "self_loops": self_loops,
    }
