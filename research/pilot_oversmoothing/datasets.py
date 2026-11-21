"""Load Planetoid, geom-GCN WebKB/Wikipedia, and heterophilous graphs."""

from __future__ import annotations

import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from graph_utils import largest_connected_component, subgraph, to_symmetric_simple

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


@dataclass
class GraphData:
    name: str
    family: str  # planetoid / webkb / wikipedia / heterophilous
    A: sp.csr_matrix
    X: np.ndarray
    y: np.ndarray
    splits: list[dict]  # each is {train, val, test} boolean masks
    n_full: int
    n_lcc: int
    n0_full: int
    notes: str = ""
    extras: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.A.shape[0])

    @property
    def n_edges(self) -> int:
        return int(self.A.nnz // 2)

    @property
    def n_classes(self) -> int:
        return int(self.y.max() + 1) if self.y.size else 0

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])


def _row_normalize(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    return X / nrm


def _parse_index_file(path: Path) -> list[int]:
    return [int(line.strip()) for line in path.read_text().splitlines() if line.strip()]


def load_planetoid(name: str) -> GraphData:
    """Yang et al. Planetoid files; CiteSeer isolated-test-node fix as in PyG/Kipf."""
    name = name.lower()
    path = DATA / "planetoid"
    names = ["x", "y", "tx", "ty", "allx", "ally", "graph"]
    objects = []
    for n in names:
        with open(path / f"ind.{name}.{n}", "rb") as f:
            if sys.version_info > (3, 0):
                objects.append(pickle.load(f, encoding="latin1"))
            else:
                objects.append(pickle.load(f))
    x, y, tx, ty, allx, ally, graph = objects
    test_idx = np.array(_parse_index_file(path / f"ind.{name}.test.index"), dtype=np.int64)
    test_idx_sorted = np.sort(test_idx)

    if name == "citeseer":
        # Isolated test nodes are not in tx; extend the range (PyG convention).
        lo, hi = int(test_idx.min()), int(test_idx.max())
        test_range = np.arange(lo, hi + 1)
        tx_ext = sp.lil_matrix((len(test_range), x.shape[1]))
        ty_ext = np.zeros((len(test_range), y.shape[1]))
        tx_ext[test_idx_sorted - lo, :] = tx
        ty_ext[test_idx_sorted - lo, :] = ty
        tx, ty = tx_ext, ty_ext

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx, :] = features[test_idx_sorted, :]
    labels = np.vstack((ally, ty))
    labels[test_idx, :] = labels[test_idx_sorted, :]
    y_int = np.argmax(labels, axis=1).astype(np.int64)

    n = features.shape[0]
    # Build adj from graph dict (0..n-1).
    rows, cols = [], []
    for u, nbrs in graph.items():
        u = int(u)
        for v in nbrs:
            rows.append(u)
            cols.append(int(v))
    A = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    A = to_symmetric_simple(A)

    train_idx = np.arange(len(y))  # 20 per class in the official split
    val_idx = np.arange(len(y), len(y) + 500)
    test_mask = np.zeros(n, dtype=bool)
    train_mask = np.zeros(n, dtype=bool)
    val_mask = np.zeros(n, dtype=bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    n0_full = int((np.asarray(A.sum(axis=1)).ravel() == 0).sum())
    keep = largest_connected_component(A)
    notes = (
        "Official Planetoid split; statistics and models use the largest "
        "connected component (undirected, no self-loops). GCN training still "
        "adds self-loops in the renormalisation trick."
    )
    A2, X2, y2, splits = _restrict_to_idx(
        A, np.asarray(features.todense()), y_int,
        [{"train": train_mask, "val": val_mask, "test": test_mask}],
        keep,
    )
    return GraphData(
        name={"cora": "Cora", "citeseer": "CiteSeer", "pubmed": "PubMed"}[name],
        family="planetoid",
        A=A2,
        X=_row_normalize(X2),
        y=y2,
        splits=splits,
        n_full=n,
        n_lcc=int(len(keep)),
        n0_full=n0_full,
        notes=notes,
    )


def _load_geom_txt(name: str) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    feat_path = DATA / "geom_gcn" / "new_data" / name / "out1_node_feature_label.txt"
    edge_path = DATA / "geom_gcn" / "new_data" / name / "out1_graph_edges.txt"
    xs, ys, ids = [], [], []
    lines = feat_path.read_text().splitlines()
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.split("\t")
        ids.append(int(parts[0]))
        xs.append([float(v) for v in parts[1].split(",")])
        ys.append(int(parts[2]))
    ids = np.array(ids)
    # assume node ids are 0..n-1 (true for geom-gcn files)
    n = int(ids.max()) + 1
    X = np.zeros((n, len(xs[0])), dtype=np.float64)
    y = np.zeros(n, dtype=np.int64)
    for i, feat, lab in zip(ids, xs, ys):
        X[i] = feat
        y[i] = lab
    rows, cols = [], []
    for line in edge_path.read_text().splitlines()[1:]:
        if not line.strip():
            continue
        u, v = line.split("\t")
        rows.append(int(u))
        cols.append(int(v))
    A = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    A = to_symmetric_simple(A)
    return A, X, y


def _load_geom_splits(name: str, n: int) -> list[dict]:
    splits = []
    for i in range(10):
        p = DATA / "geom_gcn" / "splits" / f"{name}_split_0.6_0.2_{i}.npz"
        if not p.exists():
            break
        z = np.load(p)
        def _mask(key):
            a = np.array(z[key])
            if a.dtype == bool or a.dtype == np.uint8:
                m = a.astype(bool)
                if m.shape[0] != n:
                    raise ValueError(f"split {name} {i} {key} length {m.shape[0]} != {n}")
                return m
            m = np.zeros(n, dtype=bool)
            m[a.astype(int)] = True
            return m
        splits.append({"train": _mask("train_mask"), "val": _mask("val_mask"), "test": _mask("test_mask")})
    return splits


def _random_splits(y: np.ndarray, n_splits: int, seed: int = 0, train=0.6, val=0.2) -> list[dict]:
    rng = np.random.default_rng(seed)
    n = len(y)
    splits = []
    for _ in range(n_splits):
        idx = rng.permutation(n)
        n_train = int(n * train)
        n_val = int(n * val)
        tr, va, te = idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]
        train_m = np.zeros(n, dtype=bool); train_m[tr] = True
        val_m = np.zeros(n, dtype=bool); val_m[va] = True
        test_m = np.zeros(n, dtype=bool); test_m[te] = True
        splits.append({"train": train_m, "val": val_m, "test": test_m})
    return splits


def _restrict_to_idx(A, X, y, splits, keep):
    keep = np.asarray(keep)
    A2 = subgraph(A, keep)
    X2 = X[keep]
    y2 = y[keep]
    mapping = {int(old): i for i, old in enumerate(keep)}
    new_splits = []
    for sp_ in splits:
        out = {}
        for k, mask in sp_.items():
            m2 = np.zeros(len(keep), dtype=bool)
            for old in np.where(mask)[0]:
                if old in mapping:
                    m2[mapping[old]] = True
            out[k] = m2
        new_splits.append(out)
    return A2, X2, y2, new_splits


def load_geom(name: str, family: str) -> GraphData:
    A, X, y = _load_geom_txt(name)
    n = A.shape[0]
    n0_full = int((np.asarray(A.sum(axis=1)).ravel() == 0).sum())
    splits = _load_geom_splits(name, n)
    if not splits:
        splits = _random_splits(y, 10, seed=0)
        note_split = "random 60/20/20 splits (official files missing)"
    else:
        note_split = "official geom-GCN 60/20/20 splits"
    keep = largest_connected_component(A)
    A2, X2, y2, splits = _restrict_to_idx(A, X, y, splits, keep)
    notes = (
        f"{note_split}. LCC of the undirected simple graph (no self-loops). "
    )
    if name in {"chameleon", "squirrel"}:
        notes += (
            "These Wikipedia networks are the geom-GCN preprocessed dumps "
            "(commit f1fc0d1); they are known to contain duplicate/overlapping "
            "nodes (Platonov et al.), so treat classification numbers with caution."
        )
    title = name.capitalize()
    return GraphData(
        name=title,
        family=family,
        A=A2,
        X=_row_normalize(X2),
        y=y2,
        splits=splits,
        n_full=n,
        n_lcc=int(len(keep)),
        n0_full=n0_full,
        notes=notes.strip(),
    )


def load_heterophilous(name: str) -> GraphData:
    z = np.load(DATA / "heterophilous" / f"{name}.npz", allow_pickle=True)
    X = np.array(z["node_features"], dtype=np.float64)
    y = np.array(z["node_labels"], dtype=np.int64)
    edges = np.array(z["edges"])
    n = X.shape[0]
    A = sp.csr_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n))
    A = to_symmetric_simple(A)
    train_masks = np.array(z["train_masks"])
    val_masks = np.array(z["val_masks"])
    test_masks = np.array(z["test_masks"])
    # shapes (10, n)
    splits = [
        {"train": train_masks[i].astype(bool), "val": val_masks[i].astype(bool), "test": test_masks[i].astype(bool)}
        for i in range(train_masks.shape[0])
    ]
    n0_full = int((np.asarray(A.sum(axis=1)).ravel() == 0).sum())
    keep = largest_connected_component(A)
    A2, X2, y2, splits = _restrict_to_idx(A, X, y, splits, keep)
    title = {"roman_empire": "Roman-empire", "amazon_ratings": "Amazon-ratings"}[name]
    return GraphData(
        name=title,
        family="heterophilous",
        A=A2,
        X=_row_normalize(X2),
        y=y2,
        splits=splits,
        n_full=n,
        n_lcc=int(len(keep)),
        n0_full=n0_full,
        notes=(
            "Platonov et al. heterophilous graphs; official 10 splits. "
            "LCC of the undirected simple graph (no self-loops)."
        ),
    )


def load_all(wanted: list[str] | None = None) -> list[GraphData]:
    catalog = [
        ("cora", lambda: load_planetoid("cora")),
        ("citeseer", lambda: load_planetoid("citeseer")),
        ("pubmed", lambda: load_planetoid("pubmed")),
        ("texas", lambda: load_geom("texas", "webkb")),
        ("wisconsin", lambda: load_geom("wisconsin", "webkb")),
        ("cornell", lambda: load_geom("cornell", "webkb")),
        ("chameleon", lambda: load_geom("chameleon", "wikipedia")),
        ("squirrel", lambda: load_geom("squirrel", "wikipedia")),
        ("roman_empire", lambda: load_heterophilous("roman_empire")),
        ("amazon_ratings", lambda: load_heterophilous("amazon_ratings")),
    ]
    out = []
    for key, fn in catalog:
        if wanted and key not in wanted:
            continue
        try:
            g = fn()
            out.append(g)
            print(f"loaded {g.name}: n={g.n} (full {g.n_full}) m={g.n_edges} f={g.n_features} c={g.n_classes} splits={len(g.splits)}", flush=True)
        except Exception as e:
            print(f"SKIP {key}: {type(e).__name__}: {e}", flush=True)
    return out
