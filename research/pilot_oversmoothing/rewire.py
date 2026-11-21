"""Edge-count-preserving rewires scored by the exact local Delta sigma^2 update."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from collections import deque

from graph_utils import sigma2_and_mu, to_symmetric_simple


class GraphState:
    def __init__(self, A: sp.spmatrix):
        A = to_symmetric_simple(A)
        self.n = A.shape[0]
        self.nbrs = [set() for _ in range(self.n)]
        coo = A.tocoo()
        for i, j, v in zip(coo.row, coo.col, coo.data):
            if i < j and v != 0:
                self.nbrs[int(i)].add(int(j))
                self.nbrs[int(j)].add(int(i))
        self.deg = np.array([len(s) for s in self.nbrs], dtype=np.int64)

    @property
    def m(self) -> int:
        return int(self.deg.sum() // 2)

    def has_edge(self, u, v) -> bool:
        return v in self.nbrs[u]

    def c_i(self, i: int, exclude: int | None = None) -> float:
        s = 0.0
        for j in self.nbrs[i]:
            if j == exclude:
                continue
            dj = self.deg[j]
            if dj > 0:
                s += 1.0 / dj
        return s

    def sigma2(self) -> float:
        acc = 0.0
        n0 = 0
        for i, nbr in enumerate(self.nbrs):
            di = self.deg[i]
            if di == 0:
                n0 += 1
                continue
            for j in nbr:
                if i < j:
                    acc += 1.0 / (di * self.deg[j])
        mu = 1.0 - n0 / self.n
        fro2 = (self.n - n0) + 2.0 * acc
        return fro2 / self.n - mu * mu

    def delta_add(self, u: int, v: int) -> float:
        if u == v or self.has_edge(u, v):
            return 0.0
        du, dv = int(self.deg[u]), int(self.deg[v])
        if du < 1 or dv < 1:
            # T2 formula does not apply (isolate). Return nan to reject.
            return float("nan")
        cu, cv = self.c_i(u), self.c_i(v)
        hu = 1.0 / (du + 1) - 1.0 / du
        hv = 1.0 / (dv + 1) - 1.0 / dv
        return (2.0 / self.n) * (cu * hu + cv * hv + du * dv * hu * hv)

    def delta_del(self, u: int, v: int) -> float:
        if u == v or not self.has_edge(u, v):
            return 0.0
        du, dv = int(self.deg[u]), int(self.deg[v])
        if du < 2 or dv < 2:
            return float("nan")
        cu, cv = self.c_i(u, exclude=v), self.c_i(v, exclude=u)
        hu = 1.0 / (du - 1) - 1.0 / du
        hv = 1.0 / (dv - 1) - 1.0 / dv
        return (2.0 / self.n) * (cu * hu + cv * hv - 1.0 / (du * dv))

    def add_edge(self, u, v):
        assert u != v and not self.has_edge(u, v)
        self.nbrs[u].add(v)
        self.nbrs[v].add(u)
        self.deg[u] += 1
        self.deg[v] += 1

    def del_edge(self, u, v):
        assert self.has_edge(u, v)
        self.nbrs[u].remove(v)
        self.nbrs[v].remove(u)
        self.deg[u] -= 1
        self.deg[v] -= 1

    def is_connected_no_isolates(self) -> bool:
        if (self.deg == 0).any():
            return False
        start = 0
        seen = {start}
        q = deque([start])
        while q:
            u = q.popleft()
            for v in self.nbrs[u]:
                if v not in seen:
                    seen.add(v)
                    q.append(v)
        return len(seen) == self.n

    def to_csr(self) -> sp.csr_matrix:
        rows, cols = [], []
        for i, nbr in enumerate(self.nbrs):
            for j in nbr:
                if i < j:
                    rows.extend([i, j])
                    cols.extend([j, i])
        data = np.ones(len(rows), dtype=np.float64)
        return sp.csr_matrix((data, (rows, cols)), shape=(self.n, self.n))

    def edges(self) -> list[tuple[int, int]]:
        return [(i, j) for i, nbr in enumerate(self.nbrs) for j in nbr if i < j]


def _sample_deletable(state: GraphState, rng: np.random.Generator, k: int) -> list[tuple[int, int]]:
    edges = state.edges()
    if not edges:
        return []
    # Prefer endpoints with deg >= 2
    ok = [(u, v) for u, v in edges if state.deg[u] >= 2 and state.deg[v] >= 2]
    pool = ok if ok else edges
    if len(pool) <= k:
        return pool
    idx = rng.choice(len(pool), size=k, replace=False)
    return [pool[i] for i in idx]


def _sample_addable(state: GraphState, rng: np.random.Generator, k: int, tries_mult: int = 40, bias: str | None = None) -> list[tuple[int, int]]:
    n = state.n
    seen = set()
    out = []
    deg = state.deg.astype(np.float64)
    if bias == "up":
        # prefer low-degree endpoints (T4/T5: leaf-leaf of hubs can raise sigma2)
        w = 1.0 / np.maximum(deg, 1.0)
    elif bias == "down":
        w = np.maximum(deg, 1.0)
    else:
        w = np.ones(n)
    w = w / w.sum()
    for _ in range(k * tries_mult):
        u = int(rng.choice(n, p=w))
        v = int(rng.choice(n, p=w))
        if u == v:
            continue
        if u > v:
            u, v = v, u
        if (u, v) in seen or state.has_edge(u, v):
            continue
        if state.deg[u] < 1 or state.deg[v] < 1:
            continue
        seen.add((u, v))
        out.append((u, v))
        if len(out) >= k:
            break
    return out


def delta_swap(state: GraphState, del_e, add_e) -> float:
    """Composition: delete then add, evaluated with the exact T2 formulae."""
    u, v = del_e
    x, y = add_e
    if len({u, v, x, y}) < 3:
        # loops or deleting and adding the same edge
        if {u, v} == {x, y}:
            return 0.0
    ddel = state.delta_del(u, v)
    if not np.isfinite(ddel):
        return float("nan")
    state.del_edge(u, v)
    dadd = state.delta_add(x, y)
    state.add_edge(u, v)  # revert
    if not np.isfinite(dadd):
        return float("nan")
    return ddel + dadd


def _apply_swap(state: GraphState, del_e, add_e):
    state.del_edge(*del_e)
    state.add_edge(*add_e)


def rewire_target(
    A: sp.spmatrix,
    direction: str,
    rel_target: float = 0.20,
    max_swaps: int = 400,
    sample_del: int = 40,
    sample_add: int = 40,
    seed: int = 0,
    require_connected: bool = True,
) -> dict:
    """
    Greedy delete+add rewires. direction is 'up' or 'down'.
    rel_target is the desired relative change (0.20 = 20%).
    Mean degree is invariant because |E| is fixed.
    """
    rng = np.random.default_rng(seed)
    state = GraphState(A)
    s0 = state.sigma2()
    if direction == "up":
        target = s0 * (1.0 + rel_target)
        def better(d): return np.isfinite(d) and d > 0
        def reached(): return state.sigma2() >= target
        pick = max
    elif direction == "down":
        target = s0 * (1.0 - rel_target)
        def better(d): return np.isfinite(d) and d < 0
        def reached(): return state.sigma2() <= target
        pick = min
    else:
        raise ValueError(direction)

    applied = []
    attempted = 0
    rejected_disc = 0
    for _ in range(max_swaps):
        if reached():
            break
        dels = _sample_deletable(state, rng, sample_del)
        adds = _sample_addable(state, rng, sample_add, bias=direction)
        best = None  # (delta, del, add)
        for de in dels:
            for ae in adds:
                if len({de[0], de[1], ae[0], ae[1]}) < 4 and {de[0], de[1]} == {ae[0], ae[1]}:
                    continue
                if de[0] in ae and de[1] in ae:
                    continue
                dlt = delta_swap(state, de, ae)
                if not better(dlt):
                    continue
                if best is None or (pick is max and dlt > best[0]) or (pick is min and dlt < best[0]):
                    best = (dlt, de, ae)
        attempted += 1
        if best is None:
            # escalate the sample once in a while rather than giving up
            sample_del = min(sample_del + 5, 80)
            sample_add = min(sample_add + 5, 80)
            continue
        _, de, ae = best
        _apply_swap(state, de, ae)
        if require_connected and not state.is_connected_no_isolates():
            # revert
            state.del_edge(*ae)
            state.add_edge(*de)
            rejected_disc += 1
            continue
        applied.append({"del": [int(de[0]), int(de[1])], "add": [int(ae[0]), int(ae[1])], "delta": float(best[0])})
        if len(applied) % 25 == 0 or reached():
            print(f"      {direction}: {len(applied)} swaps, sigma2 {s0:.5f}->{state.sigma2():.5f} ({(state.sigma2()-s0)/s0:+.1%})", flush=True)

    s1 = state.sigma2()
    A2 = state.to_csr()
    # verify against T1
    s_chk, _, n0 = sigma2_and_mu(A2)
    return {
        "direction": direction,
        "sigma2_before": float(s0),
        "sigma2_after": float(s1),
        "sigma2_check": float(s_chk),
        "rel_change": float((s1 - s0) / s0) if s0 else float("nan"),
        "n_swaps": len(applied),
        "n_rejected_disconnect": rejected_disc,
        "n_rounds": attempted,
        "n0_after": int(n0),
        "n_edges_before": int(A.nnz // 2),
        "n_edges_after": int(A2.nnz // 2),
        "A": A2,
        "reached_target": bool(reached()),
        "target": float(target),
    }


def random_rewire(
    A: sp.spmatrix,
    n_swaps: int,
    seed: int = 0,
    require_connected: bool = True,
    max_tries: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    state = GraphState(A)
    s0 = state.sigma2()
    if max_tries is None:
        max_tries = max(200, n_swaps * 40)
    applied = 0
    tries = 0
    rejected = 0
    while applied < n_swaps and tries < max_tries:
        tries += 1
        dels = _sample_deletable(state, rng, 1)
        adds = _sample_addable(state, rng, 1)
        if not dels or not adds:
            continue
        de, ae = dels[0], adds[0]
        if {de[0], de[1]} == {ae[0], ae[1]}:
            continue
        if not np.isfinite(state.delta_del(*de)) or not np.isfinite(
            # add formula after delete may need deg>=1 which is ok
            0.0
        ):
            continue
        # tentative
        if state.has_edge(*ae):
            continue
        state.del_edge(*de)
        if state.has_edge(*ae) or ae[0] == ae[1] or state.deg[ae[0]] < 1 or state.deg[ae[1]] < 1:
            state.add_edge(*de)
            continue
        state.add_edge(*ae)
        if require_connected and not state.is_connected_no_isolates():
            state.del_edge(*ae)
            state.add_edge(*de)
            rejected += 1
            continue
        applied += 1
    s1 = state.sigma2()
    A2 = state.to_csr()
    s_chk, _, n0 = sigma2_and_mu(A2)
    return {
        "direction": "random",
        "sigma2_before": float(s0),
        "sigma2_after": float(s1),
        "sigma2_check": float(s_chk),
        "rel_change": float((s1 - s0) / s0) if s0 else float("nan"),
        "n_swaps": applied,
        "n_rejected_disconnect": rejected,
        "n_rounds": tries,
        "n0_after": int(n0),
        "n_edges_before": int(A.nnz // 2),
        "n_edges_after": int(A2.nnz // 2),
        "A": A2,
        "reached_target": True,
        "target": float("nan"),
    }
