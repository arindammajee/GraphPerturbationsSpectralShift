#!/usr/bin/env python3
"""Pilot: does normalised-Laplacian eigenvalue variance predict GNN over-smoothing?"""

from __future__ import annotations

import json
import math
import time
import traceback
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from scipy.stats import spearmanr

from datasets import GraphData, load_all
from gnn import sgc_accuracy, train_gcn
from graph_utils import edge_homophily, mean_degree, sigma2_and_mu, spectral_gap
from oversmooth import propagate_curve
from rewire import random_rewire, rewire_target

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
FIG = ROOT / "figures"
RES.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

KS = [0, 1, 2, 4, 8, 12, 16, 24, 32, 48, 64]
INTERVENTION_DATASETS = {"Cora", "Chameleon", "Texas"}
# Full GNN accuracy on the original graph (skip the largest ones on CPU).
TRAIN_ORIGINAL = {"Cora", "CiteSeer", "Texas", "Wisconsin", "Cornell", "Chameleon"}  # Squirrel skipped: SGC too slow on CPU
SKIP_TRAIN_NAMES = {"PubMed", "Roman-empire", "Amazon-ratings"}
N_SEEDS = 5
GCN_LAYERS = (2, 8)  # 16 only on graphs with n < 500 if time remains
SGC_KS = (2, 10)
TIME_BUDGET_SEC = 38 * 60
HIDDEN = 32
EPOCHS = 200
PATIENCE = 25


def _jsonify(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _jsonify(obj.tolist())
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if obj is None:
        return None
    return obj


def dataset_stats(g: GraphData) -> dict:
    A = g.A
    s2, mu, n0 = sigma2_and_mu(A)
    dbar = mean_degree(A)
    h = edge_homophily(A, g.y)
    t0 = time.time()
    gap = spectral_gap(A, self_loops=False)
    gap_loop = spectral_gap(A, self_loops=True)
    return {
        "name": g.name,
        "family": g.family,
        "n_full": g.n_full,
        "n_lcc": g.n_lcc,
        "n": g.n,
        "n_edges": g.n_edges,
        "n_features": g.n_features,
        "n_classes": g.n_classes,
        "n0_full": g.n0_full,
        "n0_lcc": n0,
        "mu": mu,
        "sigma2": s2,
        "mean_degree": dbar,
        "sigma2_times_mean_degree": s2 * dbar if dbar else None,
        "inv_mean_degree": (1.0 / dbar) if dbar else None,
        "edge_homophily": h,
        "spectral_gap": gap,
        "spectral_gap_self_loops": gap_loop,
        "gap_seconds": time.time() - t0,
        "notes": g.notes,
    }


def run_oversmooth(g: GraphData, rng: np.random.Generator) -> dict:
    out = {"name": g.name}
    for loops in (False, True):
        key = "with_self_loops" if loops else "no_self_loops"
        t0 = time.time()
        curve = propagate_curve(g.A, g.X, KS, self_loops=loops, rng=rng)
        curve["seconds"] = time.time() - t0
        out[key] = curve
    return out


def summarise_mean_std(rows: list[dict], field: str) -> dict:
    vals = [r[field] for r in rows if r.get(field) is not None and np.isfinite(r[field])]
    if not vals:
        return {"mean": None, "std": None, "n": 0}
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0), "n": len(vals)}


def train_suite(name: str, A: sp.spmatrix, g: GraphData, variant: str, seeds: int, layers, remaining) -> list[dict]:
    rows = []
    splits = g.splits
    n_splits = min(seeds, len(splits)) if len(splits) > 1 else 1
    # Planetoid: one official split, vary init seed. Others: use first n_splits official splits.
    jobs = []
    if g.family == "planetoid":
        for seed in range(seeds):
            jobs.append((splits[0], seed))
    else:
        for i in range(n_splits):
            jobs.append((splits[i], i))

    for n_layers in layers:
        for split, seed in jobs:
            if remaining() < 45 and n_layers >= 8 and name not in {"Texas", "Cornell", "Wisconsin"}:
                print(f"  time: skip remaining GCN-{n_layers} on {name}/{variant}", flush=True)
                return rows
            t0 = time.time()
            try:
                rec = train_gcn(A, g.X, g.y, split, n_layers=n_layers, seed=seed, hidden=HIDDEN, epochs=EPOCHS, patience=PATIENCE)
                rec.update({"dataset": name, "variant": variant, "seconds": time.time() - t0})
                rows.append(rec)
                print(f"  {name}/{variant} {rec['model']} seed={seed} test={rec['test_acc']:.3f} val={rec['val_acc']:.3f} ({rec['seconds']:.1f}s)", flush=True)
            except Exception as e:
                print(f"  FAIL GCN-{n_layers} {name}/{variant} seed={seed}: {e}", flush=True)
                traceback.print_exc()
    for k in SGC_KS:
        for split, seed in jobs:
            t0 = time.time()
            try:
                rec = sgc_accuracy(A, g.X, g.y, split, k=k, seed=seed)
                rec.update({"dataset": name, "variant": variant, "seconds": time.time() - t0})
                rows.append(rec)
                print(f"  {name}/{variant} {rec['model']} seed={seed} test={rec['test_acc']:.3f} ({rec['seconds']:.1f}s)", flush=True)
            except Exception as e:
                print(f"  FAIL SGC-{k} {name}/{variant} seed={seed}: {e}", flush=True)
                # SGC is deterministic on a fixed split: only need one planetoid run per k, but keep loop.
                traceback.print_exc()
                break
    return rows


def correlate(xs, ys):
    a = np.array(xs, dtype=float)
    b = np.array(ys, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if len(a) < 3:
        return {"n": int(len(a)), "spearman": None, "pvalue": None}
    r, p = spearmanr(a, b)
    return {"n": int(len(a)), "spearman": float(r), "pvalue": float(p)}


def residual_corr(y, pred_vars: dict, of_interest, names):
    """Spearman between residuals of y~controls and residuals of of_interest~controls. Small-n, exploratory."""
    y = np.array(y, dtype=float)
    z = np.array(of_interest, dtype=float)
    cols = [np.array(pred_vars[k], dtype=float) for k in names]
    mask = np.isfinite(y) & np.isfinite(z)
    for c in cols:
        mask &= np.isfinite(c)
    if mask.sum() < 4:
        return {"n": int(mask.sum()), "partial_spearman": None, "note": "too few points"}
    Y = y[mask]
    Z = z[mask]
    X = np.column_stack([c[mask] for c in cols] + [np.ones(int(mask.sum()))])
    try:
        beta_y, *_ = np.linalg.lstsq(X, Y, rcond=None)
        beta_z, *_ = np.linalg.lstsq(X, Z, rcond=None)
        ry = Y - X @ beta_y
        rz = Z - X @ beta_z
        r, p = spearmanr(rz, ry)
        return {"n": int(mask.sum()), "partial_spearman": float(r), "pvalue": float(p), "controls": names}
    except Exception as e:
        return {"n": int(mask.sum()), "partial_spearman": None, "error": str(e)}


def make_plots(stats, oversmooth, interventions, train_rows):
    # Figure 1: across-dataset scatter of sigma2 vs oversmoothing / homophily
    names = [s["name"] for s in stats]
    s2 = np.array([s["sigma2"] for s in stats], dtype=float)
    invd = np.array([s["inv_mean_degree"] for s in stats], dtype=float)
    homo = np.array([s["edge_homophily"] for s in stats], dtype=float)
    gap = np.array([s["spectral_gap"] for s in stats], dtype=float)
    # use with-self-loops curves (closer to GCN)
    alpha = []
    e16 = []
    for s in stats:
        ev = next(o for o in oversmooth if o["name"] == s["name"])
        alpha.append(ev["with_self_loops"]["alpha"])
        e16.append(ev["with_self_loops"]["dirichlet_k16"])
    alpha = np.array(alpha, dtype=float)
    e16 = np.array(e16, dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4))
    ax = axes[0]
    sc = ax.scatter(s2, alpha, c=homo, cmap="coolwarm", vmin=0, vmax=1, s=70, edgecolors="k", linewidths=0.4)
    for i, nm in enumerate(names):
        ax.annotate(nm, (s2[i], alpha[i]), fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel(r"$\sigma^2$")
    ax.set_ylabel(r"Dirichlet decay rate $\alpha$ (with self-loops)")
    ax.set_title("Over-smoothing decay vs spectral variance")
    fig.colorbar(sc, ax=ax, label="edge homophily")

    ax = axes[1]
    sc = ax.scatter(s2, e16, c=homo, cmap="coolwarm", vmin=0, vmax=1, s=70, edgecolors="k", linewidths=0.4)
    for i, nm in enumerate(names):
        ax.annotate(nm, (s2[i], e16[i]), fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel(r"$\sigma^2$")
    ax.set_ylabel(r"Dirichlet energy at $k=16$ (with self-loops)")
    ax.set_title("Remaining energy vs spectral variance")
    fig.colorbar(sc, ax=ax, label="edge homophily")
    fig.tight_layout()
    p1 = FIG / "sigma2_vs_oversmoothing.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)

    # Figure 2: intervention curves (with self-loops) for the 3 datasets
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.8), sharey=False)
    colors = {"original": "#222222", "up": "#c0392b", "down": "#1a73e8", "random": "#7f8c8d"}
    ls = {"original": "-", "up": "--", "down": "-.", "random": ":"}
    for ax, name in zip(axes, ["Cora", "Chameleon", "Texas"]):
        pack = interventions.get(name)
        if not pack:
            ax.set_title(f"{name} (missing)")
            continue
        for variant in ["original", "up", "down", "random"]:
            ev = pack["curves"].get(variant)
            if not ev:
                continue
            recs = ev["with_self_loops"]["records"]
            ks = [r["k"] for r in recs if r["k"] >= 1]
            es = [r["dirichlet"] for r in recs if r["k"] >= 1]
            ax.plot(ks, es, color=colors[variant], linestyle=ls[variant], label=variant, lw=1.8)
        ax.set_title(name)
        ax.set_xlabel("propagation steps k")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Dirichlet energy (with self-loops)")
    axes[0].legend(fontsize=8)
    fig.suptitle(r"Intervention: raise / lower $\sigma^2$ (edge count fixed)", y=1.03, fontsize=11)
    fig.tight_layout()
    p2 = FIG / "intervention_oversmoothing_curves.png"
    fig.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return [str(p1.resolve().relative_to(ROOT)), str(p2.resolve().relative_to(ROOT))]


def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    t_start = time.time()
    def remaining():
        return TIME_BUDGET_SEC - (time.time() - t_start)

    rng = np.random.default_rng(0)
    print("=== Phase 1: load + stats ===", flush=True)
    graphs = load_all()
    by_name = {g.name: g for g in graphs}
    resume = (RES / "dataset_stats.json").exists() and (RES / "oversmooth.json").exists() and (RES / "correlations.json").exists()
    if resume:
        stats = json.loads((RES / "dataset_stats.json").read_text())
        oversmooth = json.loads((RES / "oversmooth.json").read_text())
        corr_rows = json.loads((RES / "correlations.json").read_text())
        print("  resume: loaded existing stats / oversmooth / correlations", flush=True)
    else:
        stats = []
        for g in graphs:
            st = dataset_stats(g)
            stats.append(st)
            print(
                f"  {st['name']}: n={st['n']} m={st['n_edges']} sigma2={st['sigma2']:.5f} "
                f"<d>={st['mean_degree']:.3f} s2*<d>={st['sigma2_times_mean_degree']:.3f} "
                f"h={st['edge_homophily']:.3f} gap={st['spectral_gap']:.4f} n0={st['n0_lcc']}",
                flush=True,
            )
        (RES / "dataset_stats.json").write_text(json.dumps(_jsonify(stats), indent=2))

        print("=== Phase 2: over-smoothing curves ===", flush=True)
        oversmooth = []
        for g in graphs:
            ev = run_oversmooth(g, rng)
            oversmooth.append(ev)
            a = ev["with_self_loops"]
            print(
                f"  {g.name} loops: alpha={a['alpha']:.4f} (R2={a['alpha_r2']:.3f}) "
                f"E16={a['dirichlet_k16']:.4e} half={a['half_life_k']} ({a['seconds']:.1f}s)",
                flush=True,
            )
        (RES / "oversmooth.json").write_text(json.dumps(_jsonify(oversmooth), indent=2))

        print("=== Phase 3: across-dataset correlations ===", flush=True)
    corr_rows = []
    for loops, key in [(False, "no_self_loops"), (True, "with_self_loops")]:
        alpha = [next(o for o in oversmooth if o["name"] == s["name"])[key]["alpha"] for s in stats]
        e16 = [next(o for o in oversmooth if o["name"] == s["name"])[key]["dirichlet_k16"] for s in stats]
        mad16 = [next(o for o in oversmooth if o["name"] == s["name"])[key]["mad_k16"] for s in stats]
        predictors = {
            "sigma2": [s["sigma2"] for s in stats],
            "inv_mean_degree": [s["inv_mean_degree"] for s in stats],
            "sigma2_times_mean_degree": [s["sigma2_times_mean_degree"] for s in stats],
            "edge_homophily": [s["edge_homophily"] for s in stats],
            "spectral_gap_matching": [
                s["spectral_gap_self_loops"] if loops else s["spectral_gap"] for s in stats
            ],
        }
        block = {"propagator": key, "targets": {}}
        for tname, tvals in {"alpha": alpha, "dirichlet_k16": e16, "mad_k16": mad16}.items():
            block["targets"][tname] = {
                "vs": {p: correlate(predictors[p], tvals) for p in predictors},
                "sigma2_beyond_d_and_h": residual_corr(
                    tvals, predictors, predictors["sigma2"], ["inv_mean_degree", "edge_homophily"]
                ),
            }
        corr_rows.append(block)
    (RES / "correlations.json").write_text(json.dumps(_jsonify(corr_rows), indent=2))
    for block in corr_rows:
        print(" ", block["propagator"], flush=True)
        for tname, pack in block["targets"].items():
            bits = [f"{p}: r={pack['vs'][p]['spearman']}" for p in pack["vs"]]
            part = pack["sigma2_beyond_d_and_h"]
            print(f"    {tname}: " + ", ".join(bits) + f" | partial s2={part}", flush=True)

    print("=== Phase 4: interventions ===", flush=True)
    interventions = {}
    for name in ["Cora", "Chameleon", "Texas"]:
        if remaining() < 120:
            print("  time budget: stop before further interventions", flush=True)
            break
        if name not in by_name:
            print(f"  SKIP intervention {name}: not loaded", flush=True)
            continue
        g = by_name[name]
        print(f"  rewiring {name} (n={g.n}, m={g.n_edges}) ...", flush=True)
        t0 = time.time()
        # More swaps on larger graphs; Texas is tiny so cap lower.
        max_swaps = 80 if g.n < 400 else 500
        sample = 25 if g.n < 400 else 40
        up = rewire_target(g.A, "up", rel_target=0.20, max_swaps=max_swaps, sample_del=sample, sample_add=sample, seed=0)
        down = rewire_target(g.A, "down", rel_target=0.20, max_swaps=max_swaps, sample_del=sample, sample_add=sample, seed=1)
        n_ctrl = int(np.median([up["n_swaps"], down["n_swaps"]]))
        n_ctrl = max(n_ctrl, 1)
        rnd = random_rewire(g.A, n_swaps=n_ctrl, seed=2)
        print(
            f"    up: {up['sigma2_before']:.5f}->{up['sigma2_after']:.5f} ({up['rel_change']:+.1%}, {up['n_swaps']} swaps) "
            f"down: {down['rel_change']:+.1%} ({down['n_swaps']}) random: {rnd['rel_change']:+.1%} ({rnd['n_swaps']}) "
            f"[{time.time()-t0:.1f}s]",
            flush=True,
        )
        variants = {"original": g.A, "up": up["A"], "down": down["A"], "random": rnd["A"]}
        curves = {}
        variant_stats = {}
        for vn, Av in variants.items():
            s2, mu, n0 = sigma2_and_mu(Av)
            variant_stats[vn] = {
                "sigma2": s2,
                "mu": mu,
                "n0": n0,
                "mean_degree": mean_degree(Av),
                "edge_homophily": edge_homophily(Av, g.y),
                "spectral_gap": spectral_gap(Av, self_loops=False),
                "spectral_gap_self_loops": spectral_gap(Av, self_loops=True),
                "n_edges": int(Av.nnz // 2),
            }
            pack = {}
            for loops, key in [(False, "no_self_loops"), (True, "with_self_loops")]:
                pack[key] = propagate_curve(Av, g.X, KS, self_loops=loops, rng=rng)
            curves[vn] = pack
        # drop the sparse matrices from saved json
        def slim(d):
            return {k: v for k, v in d.items() if k != "A"}
        interventions[name] = {
            "meta": {"up": slim(up), "down": slim(down), "random": slim(rnd)},
            "variant_stats": variant_stats,
            "curves": curves,
            "variants_A": variants,  # kept in memory for training; stripped before save
        }

    print("=== Phase 5: GNN training ===", flush=True)
    train_rows = []
    partial = RES / "train_rows_partial.json"
    if partial.exists():
        train_rows = json.loads(partial.read_text())
        print(f"  loaded {len(train_rows)} cached training rows", flush=True)
    done_orig = {r["dataset"] for r in train_rows if r.get("variant") == "original"}
    # Original-graph accuracy for the smaller datasets
    for g in graphs:
        if g.name not in TRAIN_ORIGINAL:
            print(f"  skip original-graph training on {g.name}", flush=True)
            continue
        if g.name in done_orig:
            print(f"  skip original-graph training on {g.name} (cached)", flush=True)
            continue
        if remaining() < 90:
            print("  time budget: skip remaining original-graph training", flush=True)
            break
        print(f"  train original {g.name}", flush=True)
        train_rows.extend(train_suite(g.name, g.A, g, "original", N_SEEDS, GCN_LAYERS, remaining))
        (RES / "train_rows_partial.json").write_text(json.dumps(_jsonify(train_rows), indent=2))

    # Intervention variants (include original again only if not already trained, to keep one block)
    already_orig = {r["dataset"] for r in train_rows if r["variant"] == "original"}
    for name, pack in interventions.items():
        g = by_name[name]
        for vn, Av in pack["variants_A"].items():
            if vn == "original" and name in already_orig:
                continue
            if remaining() < 60:
                print("  time budget: skip remaining intervention training", flush=True)
                break
            print(f"  train {name}/{vn}", flush=True)
            seeds = 3 if (g.n > 1000 and g.n_features > 1500) else N_SEEDS
            train_rows.extend(train_suite(name, Av, g, vn, seeds, GCN_LAYERS, remaining))
            (RES / "train_rows_partial.json").write_text(json.dumps(_jsonify(train_rows), indent=2))

    # optional GCN-16 on tiny graphs if time left
    if remaining() > 90:
        for name in ["Texas", "Cornell", "Wisconsin"]:
            if name not in by_name or remaining() < 40:
                break
            g = by_name[name]
            print(f"  extra GCN-16 original {name}", flush=True)
            train_rows.extend(train_suite(name, g.A, g, "original-gcn16", N_SEEDS, (16,), remaining))

    (RES / "train_rows.json").write_text(json.dumps(_jsonify(train_rows), indent=2))

    # Strip heavy matrices then save interventions
    saved_int = {}
    for name, pack in interventions.items():
        saved_int[name] = {
            "meta": pack["meta"],
            "variant_stats": pack["variant_stats"],
            "curves": pack["curves"],
        }
    (RES / "interventions.json").write_text(json.dumps(_jsonify(saved_int), indent=2))

    print("=== Phase 6: plots ===", flush=True)
    paths = make_plots(stats, oversmooth, interventions, train_rows)
    print("  wrote", paths, flush=True)

    meta = {
        "elapsed_seconds": time.time() - t_start,
        "time_budget_sec": TIME_BUDGET_SEC,
        "n_datasets": len(graphs),
        "dataset_names": [g.name for g in graphs],
        "n_train_rows": len(train_rows),
        "choices": {
            "graph": "largest connected component of the undirected simple graph (self-loops removed)",
            "sigma2": "T1 closed form; isolates L_ii=0. After LCC, n0 is typically 0.",
            "gcn": "Kipf-Welling renormalisation (A+I), hidden=32, Adam lr=0.01, wd=5e-4, dropout=0.5, early stopping on val acc",
            "sgc": "same P=(A+I) propagator, then StandardScaler + logistic regression on train nodes",
            "splits": "Planetoid official split with random inits; geom-GCN / Platonov official splits otherwise",
            "intervention": "greedy delete+add scored by exact T2 composition, reject disconnecting or isolate-creating swaps; random control uses the median swap count",
        },
        "plot_paths": paths,
    }
    (RES / "meta.json").write_text(json.dumps(_jsonify(meta), indent=2))
    print(f"DONE in {meta['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
