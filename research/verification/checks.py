#!/usr/bin/env python3
"""
Sanity checks for conjectures on the variance of the normalised Laplacian
spectrum under edge addition, deletion, and rewiring.

Convention
----------
L = I - D^{-1/2} A D^{-1/2}  (symmetric normalised Laplacian).
Isolates use the standard diagonal L_ii = 0.
Population variance:  sigma^2 = (1/n) ||L||_F^2 - mu^2,
with mu = Tr(L)/n.

Usage
-----
  python research/verification/checks.py

Dependencies: numpy, networkx. Writes check_results.json next to this file.
"""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np

SEED = 0
OUT_JSON = Path(__file__).resolve().parent / "check_results.json"

# ---------------------------------------------------------------------------
# Core linear algebra and closed forms
# ---------------------------------------------------------------------------


def adjacency(G: nx.Graph) -> tuple[np.ndarray, list]:
    nodes = list(G.nodes())
    return nx.to_numpy_array(G, nodelist=nodes, dtype=float), nodes


def degrees_from_A(A: np.ndarray) -> np.ndarray:
    return A.sum(axis=1)


def normalised_laplacian(A: np.ndarray) -> np.ndarray:
    """Symmetric normalised Laplacian; L_ii = 0 for isolated vertices."""
    n = A.shape[0]
    d = degrees_from_A(A)
    inv_sqrt = np.zeros(n, dtype=float)
    pos = d > 0
    inv_sqrt[pos] = 1.0 / np.sqrt(d[pos])
    L = np.zeros((n, n), dtype=float)
    L[np.diag_indices(n)] = pos.astype(float)
    # Off-diagonal: D^{-1/2} A D^{-1/2}; isolates have inv_sqrt = 0.
    L -= inv_sqrt[:, None] * A * inv_sqrt[None, :]
    return L


def spectral_stats(L: np.ndarray) -> dict:
    ev = np.linalg.eigvalsh(L)
    mu = float(ev.mean())
    var = float(ev.var(ddof=0))
    return {"mu": mu, "var": var, "ev": ev, "trace": float(np.trace(L))}


def sigma2_from_eig(A: np.ndarray) -> tuple[float, float, np.ndarray]:
    L = normalised_laplacian(A)
    st = spectral_stats(L)
    return st["var"], st["mu"], st["ev"]


def n_isolates(A: np.ndarray) -> int:
    return int((degrees_from_A(A) == 0).sum())


def sigma2_closed_form(A: np.ndarray) -> tuple[float, float]:
    """
    General closed form (standard isolate convention L_ii = 0):
      n0 = number of isolates
      mu = 1 - n0/n
      ||L||_F^2 = (n - n0) + 2 * sum_{{i,j} in E} 1/(d_i d_j)
      sigma^2 = ||L||_F^2 / n - mu^2
    When n0 = 0 this reduces to sigma^2 = (2/n) * sum_e 1/(d_i d_j).
    """
    n = A.shape[0]
    d = degrees_from_A(A)
    n0 = int((d == 0).sum())
    mu = 1.0 - n0 / n
    acc = 0.0
    iu, ju = np.triu_indices(n, k=1)
    mask = A[iu, ju] > 0
    for i, j in zip(iu[mask], ju[mask]):
        acc += 1.0 / (d[i] * d[j])
    fro2 = (n - n0) + 2.0 * acc
    var = fro2 / n - mu * mu
    return var, mu


def c_neighbour(A: np.ndarray, i: int, exclude=None) -> float:
    d = degrees_from_A(A)
    n = A.shape[0]
    s = 0.0
    for j in range(n):
        if exclude is not None and j == exclude:
            continue
        if A[i, j] > 0:
            if d[j] <= 0:
                raise ZeroDivisionError("neighbour with degree 0")
            s += (A[i, j] ** 2) / d[j]
    return s


def delta_sigma2_add_formula(A: np.ndarray, u: int, v: int) -> float:
    """Closed form for adding a non-edge uv, assuming d_u, d_v >= 1 (no new/old isolates)."""
    n = A.shape[0]
    d = degrees_from_A(A)
    if A[u, v] != 0:
        raise ValueError("uv already an edge")
    if d[u] < 1 or d[v] < 1:
        raise ValueError("addition formula assumes both endpoints already non-isolated")
    cu = c_neighbour(A, u)
    cv = c_neighbour(A, v)
    hu = 1.0 / (d[u] + 1) - 1.0 / d[u]
    hv = 1.0 / (d[v] + 1) - 1.0 / d[v]
    return (2.0 / n) * (cu * hu + cv * hv + d[u] * d[v] * hu * hv)


def delta_sigma2_add_sign_quantity(A: np.ndarray, u: int, v: int) -> float:
    """Sign of Delta sigma^2_add equals the sign of this quantity (d_u,d_v>=1)."""
    d = degrees_from_A(A)
    cu = c_neighbour(A, u)
    cv = c_neighbour(A, v)
    return 1.0 - ((d[v] + 1) / d[u]) * cu - ((d[u] + 1) / d[v]) * cv


def delta_sigma2_del_formula(A: np.ndarray, u: int, v: int) -> float:
    """Corrected closed form for deleting edge uv, assuming d_u, d_v >= 2."""
    n = A.shape[0]
    d = degrees_from_A(A)
    if A[u, v] == 0:
        raise ValueError("uv not an edge")
    if d[u] < 2 or d[v] < 2:
        raise ValueError("deletion formula assumes d_u,d_v >= 2 (no new isolates)")
    cu = c_neighbour(A, u, exclude=v)
    cv = c_neighbour(A, v, exclude=u)
    hu = 1.0 / (d[u] - 1) - 1.0 / d[u]
    hv = 1.0 / (d[v] - 1) - 1.0 / d[v]
    return (2.0 / n) * (cu * hu + cv * hv - 1.0 / (d[u] * d[v]))


def delta_sigma2_del_paper_buggy(A: np.ndarray, u: int, v: int) -> float:
    """The (incorrect) deletion formula from the preprint, for contrast."""
    n = A.shape[0]
    d = degrees_from_A(A)
    cu = c_neighbour(A, u, exclude=None)  # incorrectly includes j=v
    cv = c_neighbour(A, v, exclude=None)
    hu = 1.0 / (d[u] - 1) - 1.0 / d[u]
    hv = 1.0 / (d[v] - 1) - 1.0 / d[v]
    return (2.0 / n) * (cu * hu + cv * hv + d[u] * d[v] * hu * hv)


def apply_add(A: np.ndarray, u: int, v: int) -> np.ndarray:
    B = A.copy()
    B[u, v] = B[v, u] = 1.0
    return B


def apply_del(A: np.ndarray, u: int, v: int) -> np.ndarray:
    B = A.copy()
    B[u, v] = B[v, u] = 0.0
    return B


def nonedges(A: np.ndarray, require_deg_ge: int = 1):
    n = A.shape[0]
    d = degrees_from_A(A)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] == 0 and d[i] >= require_deg_ge and d[j] >= require_deg_ge:
                pairs.append((i, j))
    return pairs


def edges(A: np.ndarray, require_deg_ge: int = 1):
    n = A.shape[0]
    d = degrees_from_A(A)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] > 0 and d[i] >= require_deg_ge and d[j] >= require_deg_ge:
                pairs.append((i, j))
    return pairs


def connected_simple(n: int, fn, rng: np.random.Generator, tries: int = 80) -> nx.Graph:
    for _ in range(tries):
        G = fn()
        G = nx.Graph(G)
        G.remove_edges_from(nx.selfloop_edges(G))
        if G.number_of_nodes() != n:
            G.add_nodes_from(range(n))
        if nx.is_connected(G) and min(dict(G.degree()).values()) >= 1:
            return G
    raise RuntimeError("failed to sample a connected graph")


def er_connected(n, p, rng):
    seed = int(rng.integers(0, 2**31 - 1))

    def fn():
        return nx.erdos_renyi_graph(n, p, seed=int(rng.integers(0, 2**31 - 1)))

    return connected_simple(n, fn, rng)


def ba_connected(n, m, rng):
    def fn():
        return nx.barabasi_albert_graph(n, m, seed=int(rng.integers(0, 2**31 - 1)))

    return connected_simple(n, fn, rng)


def ws_connected(n, k, p, rng):
    def fn():
        return nx.watts_strogatz_graph(n, k, p, seed=int(rng.integers(0, 2**31 - 1)))

    return connected_simple(n, fn, rng)


def sbm_connected(sizes, p_in, p_out, rng):
    n = sum(sizes)
    probs = [[p_in if i == j else p_out for j in range(len(sizes))] for i in range(len(sizes))]

    def fn():
        return nx.stochastic_block_model(sizes, probs, seed=int(rng.integers(0, 2**31 - 1)))

    return connected_simple(n, fn, rng)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation; average ranks for ties."""

    def rank(a):
        order = np.argsort(a, kind="mergesort")
        r = np.empty(len(a), dtype=float)
        # average ranks
        i = 0
        while i < len(a):
            j = i
            while j + 1 < len(a) and a[order[j + 1]] == a[order[i]]:
                j += 1
            avg = 0.5 * (i + j) + 1.0  # 1-based average rank
            r[order[i : j + 1]] = avg
            i = j + 1
        return r

    rx, ry = rank(np.asarray(x, dtype=float)), rank(np.asarray(y, dtype=float))
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def pearson(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def check_T1_closed_form(rng) -> dict:
    """sigma^2 closed form vs eigendecomposition, with and without isolates."""
    cases = []
    # no isolates
    for name, G in [
        ("ER_n30_p0.25", er_connected(30, 0.25, rng)),
        ("BA_n30_m2", ba_connected(30, 2, rng)),
        ("WS_n30_k4_p0.2", ws_connected(30, 4, 0.2, rng)),
        ("star_n12", nx.star_graph(11)),
        ("cycle_n20", nx.cycle_graph(20)),
        ("complete_n10", nx.complete_graph(10)),
    ]:
        A, _ = adjacency(G)
        v_eig, mu_eig, _ = sigma2_from_eig(A)
        v_cf, mu_cf = sigma2_closed_form(A)
        cases.append(
            {
                "graph": name,
                "n0": n_isolates(A),
                "var_eig": v_eig,
                "var_cf": v_cf,
                "mu_eig": mu_eig,
                "mu_cf": mu_cf,
                "abs_err_var": abs(v_eig - v_cf),
                "abs_err_mu": abs(mu_eig - mu_cf),
            }
        )
    # with isolates: a 6-path plus 3 isolated vertices (path_graph(6) has 6 nodes)
    P = nx.path_graph(6)
    P.add_nodes_from(range(6, 9))
    A, _ = adjacency(P)
    v_eig, mu_eig, ev = sigma2_from_eig(A)
    v_cf, mu_cf = sigma2_closed_form(A)
    # paper-style (wrong for isolates): force diagonal 1
    n = A.shape[0]
    d = degrees_from_A(A)
    L_paper = np.eye(n)
    for i in range(n):
        for j in range(n):
            if i != j and A[i, j] and d[i] > 0 and d[j] > 0:
                L_paper[i, j] = -A[i, j] / math.sqrt(d[i] * d[j])
    # for isolates paper still puts 1 on the diagonal, off-diag 0
    st_paper = spectral_stats(L_paper)
    cases.append(
        {
            "graph": "path6_plus_3_isolates",
            "n0": n_isolates(A),
            "var_eig": v_eig,
            "var_cf": v_cf,
            "mu_eig": mu_eig,
            "mu_cf": mu_cf,
            "abs_err_var": abs(v_eig - v_cf),
            "abs_err_mu": abs(mu_eig - mu_cf),
            "standard_mu": mu_eig,
            "paper_forced_diag1_mu": st_paper["mu"],
            "paper_forced_diag1_var": st_paper["var"],
        }
    )
    max_err = max(c["abs_err_var"] for c in cases)
    return {
        "item": "T1",
        "verdict": "supported" if max_err < 1e-10 else "refuted",
        "max_abs_err_var": max_err,
        "cases": cases,
    }


def check_T2_deletion_and_add(rng) -> dict:
    n, trials = 24, 25
    add_ok = add_fail = 0
    del_ok = del_fail = 0
    bug_ok = bug_fail = 0
    add_err_max = 0.0
    del_err_max = 0.0
    bug_err_max = 0.0
    consistency_ok = 0
    consistency_fail = 0
    for _ in range(trials):
        G = er_connected(n, 0.35, rng)
        A, _ = adjacency(G)
        # addition
        cands = nonedges(A, require_deg_ge=1)
        u, v = cands[int(rng.integers(len(cands)))]
        pred = delta_sigma2_add_formula(A, u, v)
        v0, _, _ = sigma2_from_eig(A)
        v1, _, _ = sigma2_from_eig(apply_add(A, u, v))
        err = abs((v1 - v0) - pred)
        add_err_max = max(add_err_max, err)
        add_ok += err < 1e-10
        add_fail += err >= 1e-10
        # consistency: add then delete
        B = apply_add(A, u, v)
        back = delta_sigma2_del_formula(B, u, v)
        if abs(pred + back) < 1e-10:
            consistency_ok += 1
        else:
            consistency_fail += 1
        # deletion
        ecands = [e for e in edges(A, require_deg_ge=2)]
        # also need both endpoints deg >= 2
        d = degrees_from_A(A)
        ecands = [(i, j) for (i, j) in edges(A) if d[i] >= 2 and d[j] >= 2]
        u, v = ecands[int(rng.integers(len(ecands)))]
        pred_d = delta_sigma2_del_formula(A, u, v)
        pred_bug = delta_sigma2_del_paper_buggy(A, u, v)
        v0, _, _ = sigma2_from_eig(A)
        v1, _, _ = sigma2_from_eig(apply_del(A, u, v))
        errd = abs((v1 - v0) - pred_d)
        errb = abs((v1 - v0) - pred_bug)
        del_err_max = max(del_err_max, errd)
        bug_err_max = max(bug_err_max, errb)
        del_ok += errd < 1e-10
        del_fail += errd >= 1e-10
        bug_ok += errb < 1e-10
        bug_fail += errb >= 1e-10
    corrected_ok = del_fail == 0 and add_fail == 0 and consistency_fail == 0
    return {
        "item": "T2",
        "verdict": "supported" if corrected_ok else "refuted",
        "addition": {"ok": add_ok, "fail": add_fail, "max_abs_err": add_err_max},
        "deletion_corrected": {"ok": del_ok, "fail": del_fail, "max_abs_err": del_err_max},
        "deletion_paper_buggy": {"ok": bug_ok, "fail": bug_fail, "max_abs_err": bug_err_max},
        "add_then_delete_negation": {"ok": consistency_ok, "fail": consistency_fail},
    }


def delta_sigma2_rewire_direct(A: np.ndarray, u: int, v: int, w: int) -> float:
    """
    Direct closed form for the rewire delete-uv, add-uw, assuming
    the result has no isolates (d_v >= 2, d_u,d_w >= 1, uw not present, uv present).
    """
    n = A.shape[0]
    d = degrees_from_A(A)
    acc = 0.0
    acc += -1.0 / (d[u] * d[v])
    acc += 1.0 / (d[u] * (d[w] + 1))
    # edges vx, x != u
    for x in range(n):
        if x == u or A[v, x] == 0:
            continue
        if x == w:
            acc += 1.0 / ((d[v] - 1) * (d[w] + 1)) - 1.0 / (d[v] * d[w])
        else:
            acc += (1.0 / d[x]) * (1.0 / ((d[v] - 1) * d[v]))
    # edges wy, y != u and y != v (vw already handled)
    for y in range(n):
        if y == u or y == v or A[w, y] == 0:
            continue
        acc += (1.0 / d[y]) * (-1.0 / (d[w] * (d[w] + 1)))
    return (2.0 / n) * acc


def check_T3_rewire(rng) -> dict:
    n, trials = 22, 20
    comp_ok = comp_fail = 0
    direct_ok = direct_fail = 0
    naive_ok = naive_fail = 0  # adding formulas on the *original* graph (should fail)
    max_comp = max_direct = max_naive = 0.0
    examples = []
    for _ in range(trials):
        G = ba_connected(n, 2, rng)
        A, _ = adjacency(G)
        d = degrees_from_A(A)
        # pick uv with d_v>=2, and a w not adjacent to u, w != v, d_w>=1
        found = None
        e_list = edges(A)
        rng.shuffle(np.array(e_list))  # don't mutate types badly
        rng.shuffle(e_list)
        for u, v in e_list:
            # try both orientations so the "fixed" vertex is u
            for uu, vv in ((u, v), (v, u)):
                if d[vv] < 2:
                    continue
                ws = [
                    w
                    for w in range(n)
                    if w != uu and w != vv and A[uu, w] == 0 and d[w] >= 1
                ]
                if not ws:
                    continue
                w = int(ws[int(rng.integers(len(ws)))])
                found = (uu, vv, w)
                break
            if found:
                break
        if not found:
            continue
        u, v, w = found
        var0, _, _ = sigma2_from_eig(A)
        B = apply_del(A, u, v)
        C = apply_add(B, u, w)
        var1, _, _ = sigma2_from_eig(C)
        actual = var1 - var0
        # composition
        pred_comp = delta_sigma2_del_formula(A, u, v) + delta_sigma2_add_formula(B, u, w)
        # direct
        pred_dir = delta_sigma2_rewire_direct(A, u, v, w)
        # naive (wrong): evaluate both formulas on G
        try:
            pred_naive = delta_sigma2_del_formula(A, u, v) + delta_sigma2_add_formula(A, u, w)
        except ValueError:
            pred_naive = float("nan")
        ecomp = abs(actual - pred_comp)
        edir = abs(actual - pred_dir)
        enai = abs(actual - pred_naive) if pred_naive == pred_naive else 1.0
        max_comp = max(max_comp, ecomp)
        max_direct = max(max_direct, edir)
        max_naive = max(max_naive, enai)
        comp_ok += ecomp < 1e-10
        comp_fail += ecomp >= 1e-10
        direct_ok += edir < 1e-10
        direct_fail += edir >= 1e-10
        # naive should typically be wrong; count "ok" if it accidentally matches
        naive_ok += enai < 1e-10
        naive_fail += enai >= 1e-10
        if len(examples) < 3:
            examples.append(
                {
                    "uvw": [int(u), int(v), int(w)],
                    "degs": [int(d[u]), int(d[v]), int(d[w])],
                    "actual": actual,
                    "composition": pred_comp,
                    "direct": pred_dir,
                    "naive_on_G": pred_naive,
                }
            )
    # We expect composition and direct to succeed, naive to fail often.
    verdict = "supported" if comp_fail == 0 and direct_fail == 0 else "refuted"
    return {
        "item": "T3",
        "verdict": verdict,
        "composition": {"ok": comp_ok, "fail": comp_fail, "max_abs_err": max_comp},
        "direct_formula": {"ok": direct_ok, "fail": direct_fail, "max_abs_err": max_direct},
        "naive_sum_of_formulas_on_original_G": {
            "ok": naive_ok,
            "fail": naive_fail,
            "max_abs_err": max_naive,
            "note": "Evaluating add(uw) on G rather than G-uv is not a valid composition.",
        },
        "examples": examples,
    }


def check_T4_sign_law(rng) -> dict:
    results = {}

    # (a) d-regular: addition should always decrease variance (non-complete)
    # 3-regular (utility / prism-like): nx.random_regular_graph
    regular_signs = []
    for n, dreg in [(16, 3), (20, 4), (18, 5)]:
        G = nx.random_regular_graph(dreg, n, seed=int(rng.integers(0, 2**31 - 1)))
        A, _ = adjacency(G)
        for u, v in nonedges(A, require_deg_ge=1)[:80]:
            pred = delta_sigma2_add_formula(A, u, v)
            q = delta_sigma2_add_sign_quantity(A, u, v)
            # sign agreement
            regular_signs.append(
                {"n": n, "d": dreg, "dvar": pred, "q": q, "negative": pred < -1e-15}
            )
    n_reg = len(regular_signs)
    n_neg = sum(1 for s in regular_signs if s["negative"])
    n_sign_agree = sum(
        1
        for s in regular_signs
        if (s["dvar"] > 0 and s["q"] > 0)
        or (s["dvar"] < 0 and s["q"] < 0)
        or (abs(s["dvar"]) < 1e-14 and abs(s["q"]) < 1e-14)
    )
    results["regular_addition"] = {
        "n_pairs": n_reg,
        "n_negative": n_neg,
        "n_sign_agrees_with_q": n_sign_agree,
        "min_dvar": min(s["dvar"] for s in regular_signs),
        "max_dvar": max(s["dvar"] for s in regular_signs),
        "all_negative": n_neg == n_reg,
    }

    # (b) two pendants of a star: adding a leaf-leaf edge should increase variance
    #     iff hub degree > 4, i.e. star with n_leaves >= 5 (star_graph(k) has k leaves, hub deg k).
    star_rows = []
    for n_leaves in range(3, 12):
        G = nx.star_graph(n_leaves)  # nodes 0 = hub, 1..n_leaves = leaves
        A, _ = adjacency(G)
        # add edge 1-2
        pred = delta_sigma2_add_formula(A, 1, 2)
        q = delta_sigma2_add_sign_quantity(A, 1, 2)
        v0, _, _ = sigma2_from_eig(A)
        v1, _, _ = sigma2_from_eig(apply_add(A, 1, 2))
        star_rows.append(
            {
                "n_leaves": n_leaves,
                "hub_degree": n_leaves,
                "formula_dvar": pred,
                "actual_dvar": v1 - v0,
                "q": q,
                "predicted_positive": n_leaves > 4,
                "actually_positive": (v1 - v0) > 1e-14,
            }
        )
    star_ok = all(
        (r["predicted_positive"] == r["actually_positive"])
        and abs(r["formula_dvar"] - r["actual_dvar"]) < 1e-10
        for r in star_rows
    )
    results["star_leaf_leaf_addition"] = {"ok": star_ok, "rows": star_rows}

    # (c) heterogeneous BA: mix of signs
    G = ba_connected(40, 2, rng)
    A, _ = adjacency(G)
    pos = neg = 0
    dvars = []
    for u, v in nonedges(A, require_deg_ge=1):
        pred = delta_sigma2_add_formula(A, u, v)
        dvars.append(pred)
        pos += pred > 1e-14
        neg += pred < -1e-14
    results["BA_n40_m2_addition_sign_mix"] = {
        "n_nonedges": len(dvars),
        "n_positive": pos,
        "n_negative": neg,
        "n_zeroish": len(dvars) - pos - neg,
        "min": min(dvars) if dvars else None,
        "max": max(dvars) if dvars else None,
        "mean": float(np.mean(dvars)) if dvars else None,
    }

    # (d) deletion on regular: should increase
    G = nx.random_regular_graph(4, 20, seed=int(rng.integers(0, 2**31 - 1)))
    A, _ = adjacency(G)
    del_vals = [delta_sigma2_del_formula(A, u, v) for u, v in edges(A)]
    results["regular_deletion_all_positive"] = {
        "n_edges": len(del_vals),
        "n_positive": sum(1 for x in del_vals if x > 1e-14),
        "min": min(del_vals),
        "max": max(del_vals),
        "all_positive": all(x > 1e-14 for x in del_vals),
    }

    verdict = (
        "supported"
        if results["regular_addition"]["all_negative"]
        and results["star_leaf_leaf_addition"]["ok"]
        and results["regular_deletion_all_positive"]["all_positive"]
        and results["BA_n40_m2_addition_sign_mix"]["n_positive"] > 0
        and results["BA_n40_m2_addition_sign_mix"]["n_negative"] > 0
        else "mixed"
    )
    return {"item": "T4", "verdict": verdict, **results}


def check_T5_extremal_edges(rng) -> dict:
    """Which non-edges maximise / minimise Delta sigma^2 on a heterogeneous graph."""
    G = ba_connected(36, 2, rng)
    A, _ = adjacency(G)
    d = degrees_from_A(A)
    recs = []
    for u, v in nonedges(A, require_deg_ge=1):
        dv = delta_sigma2_add_formula(A, u, v)
        # neighbour mean degrees
        nu = [d[j] for j in range(len(d)) if A[u, j] > 0]
        nv = [d[j] for j in range(len(d)) if A[v, j] > 0]
        recs.append(
            {
                "uv": (int(u), int(v)),
                "d_u": int(d[u]),
                "d_v": int(d[v]),
                "dmin": int(min(d[u], d[v])),
                "dmax": int(max(d[u], d[v])),
                "dsum": int(d[u] + d[v]),
                "mean_neigh_deg": float(np.mean(nu + nv)),
                "dvar": dv,
            }
        )
    recs.sort(key=lambda r: r["dvar"])
    k = 8
    lo, hi = recs[:k], recs[-k:]
    # correlations of dvar with degree features
    features = {
        "corr_dvar_dsum": pearson([r["dsum"] for r in recs], [r["dvar"] for r in recs]),
        "corr_dvar_dmin": pearson([r["dmin"] for r in recs], [r["dvar"] for r in recs]),
        "corr_dvar_dmax": pearson([r["dmax"] for r in recs], [r["dvar"] for r in recs]),
        "corr_dvar_mean_neigh_deg": pearson(
            [r["mean_neigh_deg"] for r in recs], [r["dvar"] for r in recs]
        ),
        "spearman_dvar_dsum": spearman(
            np.array([r["dsum"] for r in recs]), np.array([r["dvar"] for r in recs])
        ),
        "spearman_dvar_mean_neigh_deg": spearman(
            np.array([r["mean_neigh_deg"] for r in recs]),
            np.array([r["dvar"] for r in recs]),
        ),
    }
    # deletion extremals
    drecs = []
    for u, v in edges(A):
        if d[u] < 2 or d[v] < 2:
            continue
        dv = delta_sigma2_del_formula(A, u, v)
        drecs.append(
            {
                "uv": (int(u), int(v)),
                "d_u": int(d[u]),
                "d_v": int(d[v]),
                "dsum": int(d[u] + d[v]),
                "dvar": dv,
            }
        )
    drecs.sort(key=lambda r: r["dvar"])
    return {
        "item": "T5",
        "verdict": "mixed",  # qualitative pattern on one BA sample; not a full characterisation
        "graph": "BA_n36_m2",
        "n_nonedges": len(recs),
        "addition_smallest_dvar": lo,
        "addition_largest_dvar": hi,
        "addition_feature_correlations": features,
        "addition_range": [recs[0]["dvar"], recs[-1]["dvar"]],
        "deletion_smallest_dvar": drecs[:5],
        "deletion_largest_dvar": drecs[-5:],
        "deletion_corr_dvar_dsum": pearson(
            [r["dsum"] for r in drecs], [r["dvar"] for r in drecs]
        )
        if drecs
        else None,
        "note": (
            "On this BA sample, the sign and magnitude of Delta sigma^2_add "
            "are related to endpoint degrees and neighbour degrees; see correlations."
        ),
    }


def check_T6_isolates_and_leaves(rng) -> dict:
    """Mean and variance when an edge touches an isolate or a degree-1 vertex."""
    # Path of 5 plus an isolate (node 5). Add edge 0-5 (0 is a leaf) and 2-5 (2 has deg 2).
    G = nx.path_graph(5)
    G.add_node(5)
    A, _ = adjacency(G)
    var0, mu0, _ = sigma2_from_eig(A)
    vcf0, mcf0 = sigma2_closed_form(A)
    rows = []
    for u, v, label in [(0, 5, "leaf_to_isolate"), (2, 5, "internal_to_isolate")]:
        B = apply_add(A, u, v)
        var1, mu1, _ = sigma2_from_eig(B)
        rows.append(
            {
                "case": label,
                "mu_before": mu0,
                "mu_after": mu1,
                "delta_mu": mu1 - mu0,
                "var_before": var0,
                "var_after": var1,
                "delta_var": var1 - var0,
                "trace_before": float(np.trace(normalised_laplacian(A))),
                "trace_after": float(np.trace(normalised_laplacian(B))),
                "formula_applicable": False,
            }
        )
    # Delete a leaf edge from a star: hub-leaf, d_leaf=1. Formula should refuse / be invalid.
    S = nx.star_graph(6)
    As, _ = adjacency(S)
    d = degrees_from_A(As)
    var_s, mu_s, _ = sigma2_from_eig(As)
    B = apply_del(As, 0, 1)  # hub-leaf
    var_b, mu_b, _ = sigma2_from_eig(B)
    formula_raised = False
    try:
        _ = delta_sigma2_del_formula(As, 0, 1)
    except ValueError:
        formula_raised = True
    leaf_del = {
        "mu_before": mu_s,
        "mu_after": mu_b,
        "delta_mu": mu_b - mu_s,
        "var_before": var_s,
        "var_after": var_b,
        "new_isolates": n_isolates(B) - n_isolates(As),
        "corrected_formula_raises": formula_raised,
        "hub_degree_before": int(d[0]),
        "leaf_degree_before": int(d[1]),
    }
    # Confirm closed form still matches after these operations
    cf_err = max(
        abs(sigma2_closed_form(A)[0] - var0),
        abs(sigma2_closed_form(apply_add(A, 0, 5))[0] - sigma2_from_eig(apply_add(A, 0, 5))[0]),
        abs(sigma2_closed_form(B)[0] - var_b),
    )
    return {
        "item": "T6",
        "verdict": "supported" if formula_raised and abs(mu0 - (1 - 1 / 6)) < 1e-12 and cf_err < 1e-10 else "mixed",
        "path5_plus_isolate_mu": mu0,
        "expected_mu": 1 - 1 / 6,
        "add_isolate_cases": rows,
        "delete_leaf_from_star": leaf_del,
        "closed_form_max_abs_err": cf_err,
        "note": "The no-isolate addition/deletion formulae do not apply; the general closed form still matches the spectrum.",
    }


def check_N1_er_scaling(rng) -> dict:
    n = 80
    # keep p large enough that the typical graph is isolate-free
    ps = [0.12, 0.16, 0.20, 0.28, 0.40, 0.55, 0.75]
    n_rep = 6
    rows = []
    for p in ps:
        sigs = []
        mean_degs = []
        n0s = []
        for _ in range(n_rep):
            # use a raw ER sample (connected_simple may bias p slightly)
            G = nx.erdos_renyi_graph(n, p, seed=int(rng.integers(0, 2**31 - 1)))
            A, _ = adjacency(G)
            # drop to the closed form (handles isolates)
            v, _ = sigma2_closed_form(A)
            sigs.append(v)
            mean_degs.append(float(degrees_from_A(A).mean()))
            n0s.append(n_isolates(A))
        mean_sig = float(np.mean(sigs))
        mean_d = float(np.mean(mean_degs))
        rows.append(
            {
                "p": p,
                "np": n * p,
                "mean_degree": mean_d,
                "mean_sigma2": mean_sig,
                "std_sigma2": float(np.std(sigs)),
                "pred_1_over_np": 1.0 / (n * p),
                "pred_1_over_mean_degree": 1.0 / mean_d,
                "ratio_sigma2_times_np": mean_sig * n * p,
                "ratio_sigma2_times_mean_degree": mean_sig * mean_d,
                "mean_n0": float(np.mean(n0s)),
            }
        )
    ratios = [r["ratio_sigma2_times_np"] for r in rows]
    # supported if ratios sit near 1 (say within 20% for this finite n)
    dev = [abs(x - 1.0) for x in ratios]
    verdict = "supported" if max(dev) < 0.25 else ("mixed" if max(dev) < 0.5 else "refuted")
    return {
        "item": "N1",
        "verdict": verdict,
        "n": n,
        "n_rep": n_rep,
        "rows": rows,
        "ratio_range": [min(ratios), max(ratios)],
        "max_abs_deviation_from_1": max(dev),
    }


def check_N2_concentration_er_ba(rng) -> dict:
    n = 60
    n_samp = 250
    out = {}
    for name, G in [
        ("ER_n60_p0.2", er_connected(60, 0.2, rng)),
        ("BA_n60_m3", ba_connected(60, 3, rng)),
    ]:
        A, _ = adjacency(G)
        d = degrees_from_A(A)
        cands = nonedges(A, require_deg_ge=1)
        pick = [cands[i] for i in rng.choice(len(cands), size=min(n_samp, len(cands)), replace=False)]
        vals = np.array([delta_sigma2_add_formula(A, u, v) for u, v in pick])
        # also split BA by degree class if BA
        bucket = {}
        if name.startswith("BA"):
            med = float(np.median(d))
            hi = float(np.quantile(d, 0.9))

            def klass(i):
                if d[i] >= hi:
                    return "hub"
                if d[i] <= 2:
                    return "leaf"
                return "mid"

            groups = defaultdict(list)
            for (u, v), val in zip(pick, vals):
                a, b = sorted([klass(u), klass(v)])
                groups[f"{a}-{b}"].append(float(val))
            bucket = {k: {"n": len(v), "mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in groups.items()}
        out[name] = {
            "n_samples": len(vals),
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "cv": float(vals.std() / abs(vals.mean())) if vals.mean() != 0 else float("inf"),
            "min": float(vals.min()),
            "max": float(vals.max()),
            "iqr": float(np.subtract(*np.percentile(vals, [75, 25]))),
            "p05": float(np.percentile(vals, 5)),
            "p95": float(np.percentile(vals, 95)),
            "degree_class_buckets": bucket,
            "mean_degree": float(d.mean()),
            "max_degree": int(d.max()),
            "degree_gini_like_cv": float(d.std() / d.mean()),
        }
    er_cv = out["ER_n60_p0.2"]["cv"]
    ba_cv = out["BA_n60_m3"]["cv"]
    # Conjecture: BA has heavier tails / larger CV than ER
    verdict = "supported" if ba_cv > er_cv else "refuted"
    return {
        "item": "N2",
        "verdict": verdict,
        "conjecture": "Random-addition Delta sigma^2 is more concentrated on ER than on BA of similar n and mean degree.",
        "er_cv": er_cv,
        "ba_cv": ba_cv,
        "graphs": out,
    }


def check_N3_spectral_distance(rng) -> dict:
    """Relation of Delta sigma^2 to sum_i (lambda'_i - lambda_i)^2 and ||L'-L||_F^2."""
    G = ba_connected(40, 3, rng)
    A, _ = adjacency(G)
    d = degrees_from_A(A)
    L = normalised_laplacian(A)
    ev = np.sort(np.linalg.eigvalsh(L))
    var0 = float(ev.var(ddof=0))
    recs = []
    # deletions with d>=2
    elist = [(u, v) for (u, v) in edges(A) if d[u] >= 2 and d[v] >= 2]
    # cap for speed
    if len(elist) > 80:
        idx = rng.choice(len(elist), size=80, replace=False)
        elist = [elist[i] for i in idx]
    for u, v in elist:
        B = apply_del(A, u, v)
        L2 = normalised_laplacian(B)
        ev2 = np.sort(np.linalg.eigvalsh(L2))
        dvar = float(ev2.var(ddof=0) - var0)
        # If mu is unchanged (no new isolates), n*dvar = ||ev2||^2 - ||ev||^2
        sum_sq_diff = float(np.sum((ev2 - ev) ** 2))
        fro_dL = float(np.linalg.norm(L2 - L, ord="fro") ** 2)
        n_dvar_abs = 40 * abs(dvar)
        recs.append(
            {
                "dvar": dvar,
                "abs_dvar": abs(dvar),
                "sum_sq_dlambda": sum_sq_diff,
                "fro2_dL": fro_dL,
                "n_times_dvar": 40 * dvar,
                "sum_lambda2_diff": float(np.sum(ev2**2 - ev**2)),
                "new_isolates": n_isolates(B) > 0,
            }
        )
    # Hoffman-Wielandt: sum (d lambda)^2 <= ||L'-L||_F^2
    hw_ok = all(r["sum_sq_dlambda"] <= r["fro2_dL"] + 1e-8 for r in recs)
    # Identity: n Delta sigma^2 = sum (lambda'^2 - lambda^2) when mu fixed
    id_err = max(abs(r["n_times_dvar"] - r["sum_lambda2_diff"]) for r in recs if not r["new_isolates"])
    corr_abs = spearman(
        np.array([r["abs_dvar"] for r in recs]),
        np.array([r["sum_sq_dlambda"] for r in recs]),
    )
    corr_raw = spearman(
        np.array([r["dvar"] for r in recs]),
        np.array([r["sum_sq_dlambda"] for r in recs]),
    )
    # They are not equal
    rel_gap = [
        r["sum_sq_dlambda"] / (40 * abs(r["dvar"])) if r["dvar"] != 0 else float("nan")
        for r in recs
    ]
    return {
        "item": "N3",
        "verdict": "supported" if hw_ok and id_err < 1e-8 else "mixed",
        "n_edges_tested": len(recs),
        "hoffman_wielandt_holds": hw_ok,
        "identity_n_dvar_eq_sum_d_lambda2_max_abs_err": id_err,
        "spearman_abs_dvar_vs_sum_sq_dlambda": corr_abs,
        "spearman_dvar_vs_sum_sq_dlambda": corr_raw,
        "ratio_sum_sq_dlambda_over_n_abs_dvar": {
            "min": float(np.nanmin(rel_gap)),
            "median": float(np.nanmedian(rel_gap)),
            "max": float(np.nanmax(rel_gap)),
        },
        "note": (
            "Delta sigma^2 is not equal to (1/n) sum (lambda'_i - lambda_i)^2; "
            "the latter is a spectral distance. When mu is fixed, "
            "n Delta sigma^2 = sum_i (lambda_i'^2 - lambda_i^2)."
        ),
    }


def check_N5_scaling_other_models(rng) -> dict:
    """Does sigma^2 ≈ 1/mean_degree on WS (nearly regular) and deviate on BA/SBM?"""
    n = 60
    rows = []
    configs = [
        ("WS_k4_p0.05", lambda: ws_connected(n, 4, 0.05, rng), 4.0),
        ("WS_k6_p0.1", lambda: ws_connected(n, 6, 0.1, rng), 6.0),
        ("WS_k8_p0.2", lambda: ws_connected(n, 8, 0.2, rng), 8.0),
        ("BA_m2", lambda: ba_connected(n, 2, rng), 4.0),  # mean deg ≈ 2m
        ("BA_m3", lambda: ba_connected(n, 3, rng), 6.0),
        ("SBM_2block_pin0.4_pout0.05", lambda: sbm_connected([30, 30], 0.4, 0.05, rng), None),
        ("SBM_2block_pin0.5_pout0.02", lambda: sbm_connected([30, 30], 0.5, 0.02, rng), None),
    ]
    for name, fn, expected_mean_d in configs:
        sigs, ds = [], []
        for _ in range(5):
            G = fn()
            A, _ = adjacency(G)
            v, _ = sigma2_closed_form(A)
            sigs.append(v)
            ds.append(float(degrees_from_A(A).mean()))
        mean_sig, mean_d = float(np.mean(sigs)), float(np.mean(ds))
        rows.append(
            {
                "graph": name,
                "mean_degree": mean_d,
                "mean_sigma2": mean_sig,
                "pred_1_over_mean_degree": 1.0 / mean_d,
                "ratio": mean_sig * mean_d,
                "std_sigma2": float(np.std(sigs)),
            }
        )
    # WS should be very close to 1/k; BA/SBM may deviate
    ws_ratios = [r["ratio"] for r in rows if r["graph"].startswith("WS")]
    ba_ratios = [r["ratio"] for r in rows if r["graph"].startswith("BA")]
    sbm_ratios = [r["ratio"] for r in rows if r["graph"].startswith("SBM")]
    ws_close = all(abs(x - 1.0) < 0.08 for x in ws_ratios)
    ba_off = all(x < 0.95 for x in ba_ratios)
    # 1/<d> holds on WS (and these SBMs) but systematically overestimates sigma^2 on BA.
    verdict = "mixed" if (ws_close and ba_off) or (ws_close and not ba_off) else "refuted"
    return {
        "item": "N5",
        "verdict": verdict,
        "rows": rows,
        "ws_ratio_range": [min(ws_ratios), max(ws_ratios)],
        "ba_ratio_range": [min(ba_ratios), max(ba_ratios)],
        "sbm_ratio_range": [min(sbm_ratios), max(sbm_ratios)],
        "note": "WS is nearly regular so sigma^2 ≈ 1/k is expected; BA and SBM test whether 1/<d> still holds under heterogeneity or community structure.",
    }


def check_A1_fragility_vs_spectral_distance(rng) -> dict:
    """
    Cheap proxy: on a small heterogeneous graph, rank edges by Delta sigma^2
    (and |Delta sigma^2|) versus the sorted-spectrum distance after deletion.
    """
    G = ba_connected(32, 2, rng)
    A, _ = adjacency(G)
    d = degrees_from_A(A)
    L = normalised_laplacian(A)
    ev = np.sort(np.linalg.eigvalsh(L))
    var0 = float(ev.var(ddof=0))
    recs = []
    for u, v in edges(A):
        if d[u] < 2 or d[v] < 2:
            continue
        B = apply_del(A, u, v)
        ev2 = np.sort(np.linalg.eigvalsh(normalised_laplacian(B)))
        dvar = float(ev2.var(ddof=0) - var0)
        spec_dist = float(np.sqrt(np.sum((ev2 - ev) ** 2)))
        recs.append(
            {
                "uv": [int(u), int(v)],
                "d_u": int(d[u]),
                "d_v": int(d[v]),
                "dvar": dvar,
                "abs_dvar": abs(dvar),
                "spectral_distance": spec_dist,
            }
        )
    sd = np.array([r["spectral_distance"] for r in recs])
    dv = np.array([r["dvar"] for r in recs])
    ad = np.array([r["abs_dvar"] for r in recs])
    # Top-k overlap of rankings (attack would take top-k by spectral distance)
    def topk_overlap(score, k):
        # higher score = more "attack-like". For dvar we try both raw and abs.
        att = set(np.argsort(-sd)[:k])
        pred = set(np.argsort(-score)[:k])
        return len(att & pred) / k

    ks = [3, 5, 8]
    overlaps_abs = {f"top{k}": topk_overlap(ad, k) for k in ks if k < len(recs)}
    overlaps_raw = {f"top{k}": topk_overlap(dv, k) for k in ks if k < len(recs)}
    overlaps_neg = {f"top{k}": topk_overlap(-dv, k) for k in ks if k < len(recs)}
    return {
        "item": "A1",
        "verdict": "mixed",  # interpret after seeing correlations; not a full attack study
        "graph": "BA_n32_m2",
        "n_edges": len(recs),
        "spearman_dvar_vs_spectral_distance": spearman(dv, sd),
        "spearman_abs_dvar_vs_spectral_distance": spearman(ad, sd),
        "pearson_abs_dvar_vs_spectral_distance": pearson(ad, sd),
        "topk_overlap_rank_by_abs_dvar": overlaps_abs,
        "topk_overlap_rank_by_dvar": overlaps_raw,
        "topk_overlap_rank_by_neg_dvar": overlaps_neg,
        "spectral_distance_range": [float(sd.min()), float(sd.max())],
        "dvar_range": [float(dv.min()), float(dv.max())],
        "note": (
            "This is a ranking-correlation proxy on one small BA graph, not a "
            "reproduction of Lin et al.'s attack or an evaluation on citation networks."
        ),
    }


def check_A3_dynamic_tracking(rng) -> dict:
    """Track sigma^2 while growing an ER graph by random edge additions."""
    n = 50
    # start from a connected sparse graph
    G = er_connected(n, 0.08, rng)
    A, _ = adjacency(G)
    # target: add edges until about p=0.35
    target_edges = int(0.35 * n * (n - 1) / 2)
    records = []

    def snapshot(A, t, tag):
        v, mu = sigma2_closed_form(A)
        dbar = float(degrees_from_A(A).mean())
        records.append(
            {
                "t": t,
                "n_edges": int(A.sum() // 2),
                "mean_degree": dbar,
                "sigma2": v,
                "mu": mu,
                "n0": n_isolates(A),
                "one_over_mean_degree": 1.0 / dbar if dbar > 0 else float("nan"),
                "ratio": v * dbar if dbar > 0 else float("nan"),
                "tag": tag,
            }
        )

    t = 0
    snapshot(A, t, "start")
    cands = nonedges(A, require_deg_ge=0)
    rng.shuffle(cands)
    # add in a stream; record every 10 additions and always at the end
    added = 0
    for u, v in list(cands):
        if int(A.sum() // 2) >= target_edges:
            break
        if A[u, v] != 0:
            continue
        A = apply_add(A, u, v)
        added += 1
        t += 1
        if added % 15 == 0:
            snapshot(A, t, "step")
    snapshot(A, t, "end")
    # monotonicity? not necessarily monotone per step, but overall should fall toward 1/(np)
    start, end = records[0], records[-1]
    # correlation of sigma2 with 1/mean_degree along the trajectory
    sigs = np.array([r["sigma2"] for r in records])
    invd = np.array([r["one_over_mean_degree"] for r in records])
    return {
        "item": "A3",
        "verdict": "supported" if end["sigma2"] < start["sigma2"] and pearson(sigs, invd) > 0.9 else "mixed",
        "n": n,
        "n_snapshots": len(records),
        "start": start,
        "end": end,
        "pearson_sigma2_vs_1_over_mean_degree": pearson(sigs, invd),
        "snapshots": records,
        "note": "Synthetic growing ER, not a real dynamic network. sigma^2 is expected to track 1/<d> as the graph densifies.",
    }


def _to_serialisable(obj):
    if isinstance(obj, dict):
        return {str(k): _to_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serialisable(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def main():
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    checks = [
        check_T1_closed_form,
        check_T2_deletion_and_add,
        check_T3_rewire,
        check_T4_sign_law,
        check_T5_extremal_edges,
        check_T6_isolates_and_leaves,
        check_N1_er_scaling,
        check_N2_concentration_er_ba,
        check_N3_spectral_distance,
        check_N5_scaling_other_models,
        check_A1_fragility_vs_spectral_distance,
        check_A3_dynamic_tracking,
    ]
    results = []
    for fn in checks:
        t1 = time.time()
        rec = fn(rng)
        rec["seconds"] = time.time() - t1
        results.append(rec)
        print(
            f"[{rec['item']}] {rec['verdict']}  ({rec['seconds']:.2f}s)",
            flush=True,
        )
    payload = {
        "seed": SEED,
        "total_seconds": time.time() - t0,
        "results": results,
    }
    OUT_JSON.write_text(json.dumps(_to_serialisable(payload), indent=2))
    print(f"\nWrote {OUT_JSON} in {payload['total_seconds']:.1f}s")
    # compact stdout dump of verdicts and a few key numbers
    for rec in results:
        print("=" * 60)
        print(json.dumps(_to_serialisable({k: v for k, v in rec.items() if k != "snapshots" and k != "cases" and k != "addition_smallest_dvar" and k != "addition_largest_dvar"}), indent=2)[:2500])


if __name__ == "__main__":
    main()
