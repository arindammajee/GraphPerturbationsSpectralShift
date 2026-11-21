#!/usr/bin/env python3
"""Build RESULTS.md and figures from saved JSON, without retraining."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)


def loadj(name):
    p = RES / name
    if not p.exists():
        return None
    return json.loads(p.read_text())


def mean_std(vals):
    vals = [float(v) for v in vals if v is not None and np.isfinite(v)]
    if not vals:
        return None, None, 0
    if len(vals) == 1:
        return vals[0], 0.0, 1
    return float(np.mean(vals)), float(np.std(vals, ddof=1)), len(vals)


def fmt_pm(mean, std, n=None, digits=3):
    if mean is None:
        return "—"
    if n == 1 or std is None or std == 0:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f}±{std:.{digits}f}"


def r_fmt(pack):
    if not pack or pack.get("spearman") is None:
        return "—"
    r = pack["spearman"]
    p = pack.get("pvalue")
    n = pack.get("n")
    return f"r={r:.2f} (p={p:.3f}, n={n})"


def corr(xs, ys):
    a = np.array(xs, dtype=float)
    b = np.array(ys, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if len(a) < 3:
        return {"n": int(len(a)), "spearman": None, "pvalue": None}
    r, p = spearmanr(a, b)
    return {"n": int(len(a)), "spearman": float(r), "pvalue": float(p)}


def make_plots(stats, oversmooth, interventions):
    names = [s["name"] for s in stats]
    s2 = np.array([s["sigma2"] for s in stats], dtype=float)
    homo = np.array([s["edge_homophily"] for s in stats], dtype=float)
    ov_by = {o["name"]: o for o in oversmooth}
    alpha, e16 = [], []
    for s in stats:
        ev = ov_by[s["name"]]["with_self_loops"]
        alpha.append(ev["alpha"])
        e16.append(ev["dirichlet_k16"])
    alpha = np.array(alpha, dtype=float)
    e16 = np.array(e16, dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.5))
    ax = axes[0]
    sc = ax.scatter(s2, alpha, c=homo, cmap="coolwarm", vmin=0, vmax=1, s=72, edgecolors="k", linewidths=0.4, zorder=3)
    for i, nm in enumerate(names):
        ax.annotate(nm, (s2[i], alpha[i]), fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel(r"$\sigma^2$")
    ax.set_ylabel(r"Dirichlet decay rate $\alpha$ (with self-loops)")
    ax.set_title("Over-smoothing decay vs spectral variance")
    fig.colorbar(sc, ax=ax, label="edge homophily")

    ax = axes[1]
    sc = ax.scatter(s2, e16, c=homo, cmap="coolwarm", vmin=0, vmax=1, s=72, edgecolors="k", linewidths=0.4, zorder=3)
    for i, nm in enumerate(names):
        ax.annotate(nm, (s2[i], e16[i]), fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel(r"$\sigma^2$")
    ax.set_ylabel(r"Dirichlet energy at $k=16$ (with self-loops)")
    ax.set_yscale("log")
    ax.set_title("Remaining energy vs spectral variance")
    fig.colorbar(sc, ax=ax, label="edge homophily")
    fig.tight_layout()
    p1 = FIG / "sigma2_vs_oversmoothing.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)

    paths = [p1]
    if interventions:
        fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.9), sharey=False)
        colors = {"original": "#222222", "up": "#c0392b", "down": "#1a73e8", "random": "#7f8c8d"}
        ls = {"original": "-", "up": "--", "down": "-.", "random": ":"}
        order = ["Cora", "Chameleon", "Texas"]
        for ax, name in zip(axes, order):
            pack = interventions.get(name)
            if not pack:
                ax.set_title(f"{name} (not finished)")
                ax.set_xlabel("propagation steps k")
                continue
            for variant in ["original", "up", "down", "random"]:
                ev = pack.get("curves", {}).get(variant)
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
        paths.append(p2)
    return paths


def train_table(rows):
    agg = defaultdict(list)
    for r in rows:
        if r.get("test_acc") is None:
            continue
        agg[(r["dataset"], r["variant"], r["model"])].append(r["test_acc"])
    return {k: mean_std(v) for k, v in agg.items()}


def write_md(stats, oversmooth, corr_rows, interventions, rows, plot_paths, notes):
    ov_by = {o["name"]: o for o in oversmooth}
    acc = train_table(rows)

    lines = []
    a = lines.append
    a("# Pilot: does normalised-Laplacian eigenvalue variance predict GNN over-smoothing?")
    a("")
    a("Quick CPU-only pilot, not a paper. The question is whether")
    a(r"\(\sigma^2=\frac1n\|\mathcal{L}\|_F^2-\mu^2\)")
    a(r"(which equals \(\frac2n\sum_{(i,j)\in E}1/(d_i d_j)\) when there are no isolates)")
    a("tracks over-smoothing and node-classification accuracy, especially on heterophilic graphs.")
    a("")
    a("## What was run")
    a("")
    a("- **Graphs.** Largest connected component of the undirected simple graph, with self-loops removed for statistics and for \(\sigma^2\). GCN and SGC still use the Kipf–Welling renormalisation \(P=\tilde D^{-1/2}(A+I)\tilde D^{-1/2}\). Over-smoothing curves were computed both with and without self-loops in the propagator.")
    a("- **Datasets that loaded.** Cora, CiteSeer, PubMed (Planetoid); Texas, Wisconsin, Cornell (WebKB / geom-GCN); Chameleon, Squirrel (Wikipedia / geom-GCN, known duplicate-node caveat); Roman-empire and Amazon-ratings (Platonov et al.). All ten loaded successfully.")
    a("- **Splits.** Official Planetoid split (random inits) on Cora/CiteSeer/PubMed, restricted to the LCC; official geom-GCN or Platonov splits otherwise, also restricted to the LCC.")
    a("- **Models (original graphs).** GCN-2 and GCN-8 (hidden 32, Adam lr=0.01, weight decay \(5\\times10^{-4}\), dropout 0.5, early stopping on validation accuracy, 5 seeds) and SGC with \(k=2\) and \(k=10\) (logistic regression on the propagated features). Trained on Cora, CiteSeer, Texas, Wisconsin, Cornell, and Chameleon. **Not trained:** PubMed, Roman-empire, and Amazon-ratings (too large for this budget); Squirrel (started but stopped — SGC on 5k nodes × 2k features was too slow on CPU).")
    a("- **Intervention.** On Cora, Chameleon, and Texas: greedy delete+add rewires scored by the exact T2 \(\Delta\sigma^2\) formula, rejecting swaps that disconnect the graph or create isolates. Target ±20% in \(\sigma^2\) at fixed edge count (so \(\langle d\\rangle\) is invariant). A random-rewire control uses the median number of swaps. Then the same models were retrained on the rewired graphs (5 seeds on Cora and Texas; 3 official splits on Chameleon interventions to save time).")
    a("")
    if notes:
        a("**Completion notes.** " + notes)
        a("")

    a("## Dataset statistics")
    a("")
    a("Computed on the LCC. \(\sigma^2\langle d\\rangle=1\) exactly on a regular graph; deviations reflect degree heterogeneity. Edge homophily is the fraction of edges with the same label on both ends. The spectral gap is \(\lambda_2\) of the normalised Laplacian (no self-loops). Roman-empire's gap did not converge with ARPACK and is left as missing, not imputed.")
    a("")
    a("| Dataset | n (LCC / full) | edges | \(\\sigma^2\) | \(\\langle d\\rangle\) | \(\\sigma^2\\langle d\\rangle\) | edge h | gap |")
    a("|---|---:|---:|---:|---:|---:|---:|---:|")
    for s in stats:
        gap = s.get("spectral_gap")
        gap_s = "—" if gap is None or (isinstance(gap, float) and (math.isnan(gap) or math.isinf(gap))) else f"{gap:.4f}"
        a(
            f"| {s['name']} | {s['n']} / {s['n_full']} | {s['n_edges']} | "
            f"{s['sigma2']:.4f} | {s['mean_degree']:.2f} | {s['sigma2_times_mean_degree']:.3f} | "
            f"{s['edge_homophily']:.3f} | {gap_s} |"
        )
    a("")
    a("CiteSeer drops from 3327 to 2120 nodes on the LCC (many isolates); Cora drops 2708→2485. Accuracies below are on these LCCs, so they are not directly comparable to full-graph Planetoid numbers. Chameleon and Squirrel are the geom-GCN dumps (commit `f1fc0d1`) and are known to contain duplicate/overlapping nodes.")
    a("")
    a("Two observations already cut against a simple “high \(\\sigma^2\) = heterophily” story: WebKB graphs have \(\\sigma^2\\approx 0.22\), similar to Cora, but much lower homophily; Chameleon and Squirrel are heterophilic yet have *low* \(\\sigma^2\) because they are dense (\(\\langle d\\rangle=28\) and \(76\)). Roman-empire combines very low homophily (0.047) with the *highest* \(\\sigma^2\) (0.342).")
    a("")

    a("## Over-smoothing curves and cross-dataset correlations")
    a("")
    a("Features were row-normalised and propagated as \(X^{(k)}=S^k X\), with Dirichlet energy \(\\frac1{nf}\\mathrm{Tr}(X^\\top \\mathcal L X)\). The decay rate \(\\alpha\) is from a log-linear fit of energy versus \(k\\ge 1\). Mean cosine distance (MAD) used up to 8000 random node pairs. Half-life relative to \(k=1\) was 2 on *every* dataset with self-loops, so that particular summary is not discriminative.")
    a("")
    a("| Dataset | \(\\alpha\) (loops) | \(R^2\) | energy \(k=16\) | \(\\alpha\) (no loops) | energy \(k=16\) (no loops) |")
    a("|---|---:|---:|---:|---:|---:|")
    for s in stats:
        ev = ov_by[s["name"]]
        a_loops = ev["with_self_loops"]
        a_nl = ev["no_self_loops"]
        a(
            f"| {s['name']} | {a_loops['alpha']:.3f} | {a_loops['alpha_r2']:.2f} | "
            f"{a_loops['dirichlet_k16']:.2e} | {a_nl['alpha']:.3f} | {a_nl['dirichlet_k16']:.2e} |"
        )
    a("")
    a("Spearman correlations across the ten datasets (nine for the gap, because Roman-empire is missing). These are correlations, not proof, and \(n\\approx 10\) is small; p-values are reported only as a rough flag.")
    a("")
    # with self loops is the GCN-relevant one
    def find_block(key):
        for b in corr_rows:
            if b["propagator"] == key:
                return b
        return None

    def corr_table(block, title):
        a(f"**{title}**")
        a("")
        a("| Target | vs \(\\sigma^2\) | vs \(1/\\langle d\\rangle\) | vs \(\\sigma^2\\langle d\\rangle\) | vs homophily | vs spectral gap | \(\\sigma^2\) after \(1/\\langle d\\rangle\) + h |")
        a("|---|---|---|---|---|---|---|")
        for tname, label in [
            ("alpha", r"$\alpha$"),
            ("dirichlet_k16", "energy at k=16"),
            ("mad_k16", "MAD at k=16"),
        ]:
            pack = block["targets"][tname]
            part = pack["sigma2_beyond_d_and_h"]
            part_s = (
                f"r={part['partial_spearman']:.2f} (p={part['pvalue']:.3f})"
                if part and part.get("partial_spearman") is not None
                else "—"
            )
            vs = pack["vs"]
            a(
                f"| {label} | {r_fmt(vs['sigma2'])} | {r_fmt(vs['inv_mean_degree'])} | "
                f"{r_fmt(vs['sigma2_times_mean_degree'])} | {r_fmt(vs['edge_homophily'])} | "
                f"{r_fmt(vs['spectral_gap_matching'])} | {part_s} |"
            )
        a("")

    if corr_rows:
        corr_table(find_block("with_self_loops"), "With self-loops (closer to GCN / SGC)")
        corr_table(find_block("no_self_loops"), "Without self-loops")

    a("The only *strong* predictor of the Dirichlet decay rate is the spectral gap (Spearman \(r=0.93\) with self-loops, \(p=2\\times10^{-4}\); \(r=0.95\) without). That is the expected linear-algebra fact: the decay of \(S^k\) is governed by the second-largest eigenvalue of \(S\). \(\\sigma^2\) is only weakly related to \(\\alpha\) (\(r=-0.18\), \(p=0.63\) with self-loops). Energy remaining at \(k=16\) has a moderate association with \(\\sigma^2\) (\(r=0.55\), \(p=0.10\)) that shrinks after controlling for \(1/\\langle d\\rangle\) and homophily (partial \(r=0.16\), \(p=0.65\)). The residual association of \(\\sigma^2\) with \(\\alpha\) after those controls is larger but still inconclusive (partial \(r=-0.55\), \(p=0.098\)).")
    a("")
    a("Homophily itself correlates with MAD at \(k=16\) (\(r=0.65\), \(p=0.04\)): more homophilic graphs keep more feature similarity under propagation, which is expected and is *not* a \(\\sigma^2\) effect.")
    a("")

    a("## Original-graph accuracy")
    a("")
    a("Test accuracy mean±std over 5 seeds (Planetoid: one official split, varying initialisation; WebKB/Wikipedia: first 5 official splits). SGC on Planetoid is deterministic given the split, so the std is zero.")
    a("")
    models = ["GCN-2", "GCN-8", "SGC-2", "SGC-10"]
    ds_order = ["Cora", "CiteSeer", "Texas", "Wisconsin", "Cornell", "Chameleon"]
    a("| Dataset | GCN-2 | GCN-8 | SGC-2 | SGC-10 | \(\\sigma^2\) | h |")
    a("|---|---:|---:|---:|---:|---:|---:|")
    gcn2_for_corr = []
    drop_for_corr = []
    s2_for_corr = []
    h_for_corr = []
    for name in ds_order:
        cells = []
        for m in models:
            mu, sd, n = acc.get((name, "original", m), (None, None, 0))
            cells.append(fmt_pm(mu, sd, n))
        st = next(s for s in stats if s["name"] == name)
        a(f"| {name} | " + " | ".join(cells) + f" | {st['sigma2']:.3f} | {st['edge_homophily']:.3f} |")
        mu2, _, n2 = acc.get((name, "original", "GCN-2"), (None, None, 0))
        mu8, _, n8 = acc.get((name, "original", "GCN-8"), (None, None, 0))
        if mu2 is not None:
            gcn2_for_corr.append(mu2)
            s2_for_corr.append(st["sigma2"])
            h_for_corr.append(st["edge_homophily"])
            if mu8 is not None:
                drop_for_corr.append(mu2 - mu8)
    a("")
    c_acc = corr(s2_for_corr, gcn2_for_corr)
    c_h = corr(h_for_corr, gcn2_for_corr)
    c_drop = corr(s2_for_corr, drop_for_corr) if len(drop_for_corr) == len(s2_for_corr) else None
    a(
        f"Across these {c_acc['n']} trained datasets, Spearman of GCN-2 test accuracy with \(\\sigma^2\) is "
        f"{r_fmt(c_acc)}; with edge homophily it is {r_fmt(c_h)}. "
        + (
            f"The GCN-2 minus GCN-8 drop (a crude over-smoothing cost) versus \(\\sigma^2\) is {r_fmt(c_drop)}."
            if c_drop
            else ""
        )
    )
    a(" Deep GCN-8 is unstable and usually worse than GCN-2 on both homophilic and heterophilic graphs in this setup (no residuals, no pair-norm). That drop is real, but it is not larger specifically where \(\\sigma^2\) is large: Chameleon has *low* \(\\sigma^2\) and still a large deep-GCN drop (0.619→0.389).")
    a("")

    a("## Intervention (the causal test)")
    a("")
    if not interventions:
        a("Rewiring finished at least in part (see logs), but `interventions.json` was not written before interruption, so variant-level over-smoothing curves and rewired-graph accuracies are **not available to report**. They should not be guessed.")
        a("")
    else:
        a("Edge count, and therefore \(\\langle d\\rangle\), is fixed by construction. Homophily and the spectral gap can still move; both are reported below so that a change in over-smoothing or accuracy is not automatically attributed to \(\\sigma^2\).")
        a("")
        a("| Graph | variant | \(\\sigma^2\) | rel. change | \(\\langle d\\rangle\) | edge h | gap (loops) | \(\\alpha\) (loops) | energy \(k=16\) | swaps |")
        a("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for name in ["Cora", "Chameleon", "Texas"]:
            pack = interventions.get(name)
            if not pack:
                a(f"| {name} | — |  |  |  |  |  |  |  | not finished |")
                continue
            vs = pack["variant_stats"]
            curves = pack.get("curves", {})
            meta = pack.get("meta", {})
            s0 = vs["original"]["sigma2"]
            for vn in ["original", "up", "down", "random"]:
                st = vs[vn]
                ev = curves.get(vn, {}).get("with_self_loops", {})
                rel = (st["sigma2"] - s0) / s0 if s0 else float("nan")
                nsw = "—" if vn == "original" else str(meta.get(vn, {}).get("n_swaps", "—"))
                gap = st.get("spectral_gap_self_loops")
                gap_s = "—" if gap is None else f"{gap:.4f}"
                a(
                    f"| {name} | {vn} | {st['sigma2']:.4f} | {rel:+.1%} | {st['mean_degree']:.3f} | "
                    f"{st['edge_homophily']:.3f} | {gap_s} | {ev.get('alpha', float('nan')):.3f} | "
                    f"{ev.get('dirichlet_k16', float('nan')):.2e} | {nsw} |"
                )
        a("")
        # accuracy table for interventions
        any_int_acc = any(v != "original" and not str(v).startswith("original") for _, v, _ in acc)
        if any_int_acc:
            a("Test accuracy after rewiring (mean±std). Cora and Texas use 5 seeds; Chameleon interventions used 3 official splits when present.")
            a("")
            a("| Dataset | model | original | \(\\sigma^2\\uparrow\) | \(\\sigma^2\\downarrow\) | random rewire |")
            a("|---|---|---:|---:|---:|---:|")
            for name in ["Cora", "Chameleon", "Texas"]:
                for m in models:
                    cells = []
                    present = False
                    for vn in ["original", "up", "down", "random"]:
                        mu, sd, n = acc.get((name, vn, m), (None, None, 0))
                        if n:
                            present = True
                        cells.append(fmt_pm(mu, sd, n))
                    if present:
                        a(f"| {name} | {m} | " + " | ".join(cells) + " |")
            a("")
            a("If \(\\sigma^2\) were a causal driver of over-smoothing, raising it should *accelerate or deepen* the deep-model drop relative to a random rewire with the same number of edits, and lowering it should help deep GCN/SGC, with \(\\langle d\\rangle\) held fixed. Homophily shifts must be checked before interpreting any accuracy move.")
            a("")
        else:
            a("Rewired-graph training results were not yet available when this report was written.")
            a("")

    a("## Verdict")
    a("")
    # Verdict is written honestly from what we actually have; write_report will include a conservative
    # default that the caller can leave as-is. We'll fill more precisely in the file after seeing interventions.
    a(notes_verdict(interventions, acc, c_acc, c_h, c_drop))
    a("")
    a("## Caveats")
    a("")
    a("- Ten datasets (nine for the gap) is too few for a stable ranking correlation; a single extra point can move Spearman substantially. p-values here are descriptive.")
    a("- Cora and especially CiteSeer are evaluated on the LCC, not the full Planetoid graph.")
    a("- Chameleon and Squirrel have a documented data-quality issue (duplicate nodes). Classification numbers on those two should be treated cautiously.")
    a("- GCN-8 has no residual, identity mapping, or PairNorm; it is *meant* to over-smooth, and it is a noisy estimator of the “deep” regime (large seed-to-seed variance on Cora and CiteSeer).")
    a("- The Dirichlet half-life was 2 on every dataset with self-loops, so decay-*speed* summaries are more informative than that threshold.")
    a("- Squirrel, PubMed, Roman-empire, and Amazon-ratings were not used in the trained-GNN tables. Roman-empire's spectral gap is missing.")
    a("- Intervention rewires can change homophily and the gap even when \(\\langle d\\rangle\) is fixed; those confounds are in the intervention table.")
    a("- No result here uses ogbn-arxiv or a GAT, and the pilot is CPU-only with a small hidden size (32).")
    a("")
    def _rel(p) -> str:
        path = Path(p)
        if not path.is_absolute():
            return path.as_posix()
        try:
            return path.resolve().relative_to(ROOT).as_posix()
        except ValueError:
            return path.name

    a("## Artefacts")
    a("")
    for p in plot_paths:
        rel = _rel(p)
        a(f"- [{rel}]({rel})")
    a("- [results/dataset_stats.json](results/dataset_stats.json)")
    a("- [results/oversmooth.json](results/oversmooth.json)")
    a("- [results/correlations.json](results/correlations.json)")
    a("- [results/train_rows.json](results/train_rows.json) (or `train_rows_partial.json` if the full file is absent)")
    if interventions:
        a("- [results/interventions.json](results/interventions.json)")
    a("")

    (ROOT / "RESULTS.md").write_text("\n".join(lines) + "\n")
    return ROOT / "RESULTS.md"


def notes_verdict(interventions, acc, c_acc, c_h, c_drop):
    # Built from evidence, not optimism.
    parts = []
    parts.append(
        "**Weak, and on the current evidence closer to negative than promising, as a predictor of over-smoothing or of heterophily.** "
        "The quantity that actually tracks Dirichlet decay across datasets is the spectral gap, not \(\\sigma^2\). "
        "\(\\sigma^2\) is only weakly correlated with the fitted decay rate, does not line up with edge homophily "
        "(dense heterophilic graphs can have *small* \(\\sigma^2\); sparse homophilic graphs can have *large* \(\\sigma^2\)), "
        "and GCN-2 accuracy across the six trained datasets tracks homophily more than \(\\sigma^2\)."
    )
    if interventions:
        # inspect whether alpha/energy moved in a consistent way
        comments = []
        for name in ["Cora", "Chameleon", "Texas"]:
            pack = interventions.get(name)
            if not pack or "curves" not in pack:
                continue
            ev = {vn: pack["curves"][vn]["with_self_loops"] for vn in pack["curves"]}
            if "up" in ev and "down" in ev and "original" in ev and "random" in ev:
                comments.append(
                    f"{name}: original α={ev['original']['alpha']:.3f}, "
                    f"up {ev['up']['alpha']:.3f}, down {ev['down']['alpha']:.3f}, "
                    f"random {ev['random']['alpha']:.3f}"
                )
        if comments:
            parts.append(" Intervention over-smoothing (with self-loops): " + "; ".join(comments) + ".")
        # accuracy shifts if present
        acc_notes = []
        for name in ["Cora", "Chameleon", "Texas"]:
            for m in ["GCN-2", "GCN-8"]:
                o = acc.get((name, "original", m))
                u = acc.get((name, "up", m))
                d = acc.get((name, "down", m))
                r = acc.get((name, "random", m))
                if o and o[0] is not None and u and u[0] is not None and d and d[0] is not None and r and r[0] is not None:
                    acc_notes.append(
                        f"{name} {m}: orig {o[0]:.3f}, up {u[0]:.3f}, down {d[0]:.3f}, random {r[0]:.3f}"
                    )
        if acc_notes:
            parts.append(" Rewired test accuracies: " + "; ".join(acc_notes) + ".")
            parts.append(
                " Those accuracy moves should be read against the random-rewire column and against any homophily shift in the table above; "
                "a one-sided pattern that survives the random control would have been needed to call the link promising."
            )
        else:
            parts.append(" Rewired-graph accuracies were incomplete; the cross-dataset pattern already weighs against a strong claim.")
    else:
        parts.append(
            " The key causal test (rewire \(\\sigma^2\) up/down at fixed \(\\langle d\\rangle\), then re-measure decay and accuracy) "
            "had not been fully written to disk when the first parts of this report were compiled."
        )
    parts.append(
        " A narrower follow-up that remains conceivable is whether \(\\sigma^2\\langle d\\rangle\) (the deviation from the regular-graph law) "
        "or a gap-conditioned residual of \(\\sigma^2\) describes anything about the *shape* of the spectrum beyond \(\\lambda_2\); "
        "this pilot does not support using \(\\sigma^2\) itself as a cheap over-smoothing or heterophily score."
    )
    return "".join(parts)


def main():
    stats = loadj("dataset_stats.json")
    oversmooth = loadj("oversmooth.json")
    corr_rows = loadj("correlations.json")
    interventions = loadj("interventions.json")
    rows = loadj("train_rows.json") or loadj("train_rows_partial.json") or []
    if not stats or not oversmooth:
        raise SystemExit("missing dataset_stats.json or oversmooth.json")
    # If interventions missing, still plot fig 1
    paths = make_plots(stats, oversmooth, interventions or {})
    # completion note from files
    notes = []
    if not interventions:
        notes.append("Intervention curves and rewired-graph training were not saved (`interventions.json` missing).")
    variants = {(r.get("dataset"), r.get("variant")) for r in rows}
    int_done = any(v in {"up", "down", "random"} for _, v in variants)
    if interventions and not int_done:
        notes.append("Rewired graphs and over-smoothing curves are on disk, but rewired-graph GNN training was not finished.")
    if not any(r.get("dataset") == "Squirrel" for r in rows):
        notes.append("Squirrel, PubMed, Roman-empire, and Amazon-ratings have statistics and over-smoothing curves but no trained GNN numbers.")
    note = " ".join(notes)
    md = write_md(stats, oversmooth, corr_rows or [], interventions or {}, rows, [str(p) for p in paths], note)
    print("wrote", md)
    for p in paths:
        print("plot", p)


if __name__ == "__main__":
    main()
