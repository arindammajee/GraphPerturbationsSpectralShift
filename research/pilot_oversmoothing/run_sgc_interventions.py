#!/usr/bin/env python3
"""Re-create the seeded rewires and run SGC (lbfgs) on them, saving incrementally."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from datasets import load_all
from gnn import sgc_accuracy
from rewire import random_rewire, rewire_target

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
ROWS = RES / "train_rows.json"


def load_rows():
    return json.loads(ROWS.read_text()) if ROWS.exists() else []


def save_rows(rows):
    tmp = ROWS.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rows, indent=2))
    tmp.replace(ROWS)


def have(rows, dataset, variant, model, seed):
    return any(
        r.get("dataset") == dataset
        and r.get("variant") == variant
        and r.get("model") == model
        and r.get("seed") == seed
        for r in rows
    )


def main():
    rows = load_rows()
    graphs = {g.name: g for g in load_all(["cora", "texas", "chameleon"])}
    jobs = [
        ("Cora", 500, 40, 5, True),       # planetoid: SGC deterministic, still do seed 0 only below
        ("Texas", 80, 25, 5, False),
        ("Chameleon", 500, 40, 3, False),
    ]
    for name, max_swaps, sample, n_splits, planetoid in jobs:
        g = graphs[name]
        print(f"=== rewire {name} ===", flush=True)
        t0 = time.time()
        up = rewire_target(g.A, "up", rel_target=0.20, max_swaps=max_swaps, sample_del=sample, sample_add=sample, seed=0)
        down = rewire_target(g.A, "down", rel_target=0.20, max_swaps=max_swaps, sample_del=sample, sample_add=sample, seed=1)
        n_ctrl = max(int(np.median([up["n_swaps"], down["n_swaps"]])), 1)
        rnd = random_rewire(g.A, n_swaps=n_ctrl, seed=2)
        print(
            f"  up {up['rel_change']:+.1%} ({up['n_swaps']}) down {down['rel_change']:+.1%} ({down['n_swaps']}) "
            f"random {rnd['rel_change']:+.1%} ({rnd['n_swaps']}) [{time.time()-t0:.1f}s]",
            flush=True,
        )
        # sanity vs previous intervention file
        variants = {"up": up["A"], "down": down["A"], "random": rnd["A"]}
        splits = g.splits
        if planetoid:
            jobs_s = [(splits[0], 0)]  # one official split; SGC is deterministic
        else:
            jobs_s = [(splits[i], i) for i in range(min(n_splits, len(splits)))]
        for vn, Av in variants.items():
            for k in (2, 10):
                for split, seed in jobs_s:
                    if have(rows, name, vn, f"SGC-{k}", seed):
                        print(f"  skip {name}/{vn} SGC-{k} seed={seed}", flush=True)
                        continue
                    t1 = time.time()
                    rec = sgc_accuracy(Av, g.X, g.y, split, k=k, seed=seed)
                    rec.update({"dataset": name, "variant": vn, "seconds": time.time() - t1})
                    rows.append(rec)
                    save_rows(rows)
                    print(
                        f"  {name}/{vn} SGC-{k} seed={seed} test={rec['test_acc']:.3f} ({rec['seconds']:.1f}s)",
                        flush=True,
                    )
    print("SGC interventions DONE", flush=True)


if __name__ == "__main__":
    main()
