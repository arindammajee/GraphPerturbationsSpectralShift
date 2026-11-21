"""Sparse GCN (2 / 8 / 16 layers) and SGC on CPU."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from graph_utils import normalised_adjacency


def sparse_to_torch(S: sp.spmatrix) -> torch.Tensor:
    S = S.tocoo()
    idx = torch.tensor(np.vstack([S.row, S.col]), dtype=torch.long)
    val = torch.tensor(S.data, dtype=torch.float32)
    with torch.sparse.check_sparse_tensor_invariants(False):
        return torch.sparse_coo_tensor(idx, val, S.shape).coalesce()


def build_renorm(A: sp.spmatrix) -> torch.Tensor:
    """Kipf–Welling P = Dtilde^{-1/2} (A+I) Dtilde^{-1/2} as a torch sparse tensor."""
    return sparse_to_torch(normalised_adjacency(A, self_loops=True))


class GCN(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, hidden: int = 32, n_layers: int = 2, dropout: float = 0.5):
        super().__init__()
        assert n_layers >= 2
        self.dropout = dropout
        dims = [in_dim] + [hidden] * (n_layers - 1) + [n_classes]
        self.ws = nn.ParameterList([
            nn.Parameter(torch.empty(dims[i], dims[i + 1])) for i in range(n_layers)
        ])
        self.bs = nn.ParameterList([
            nn.Parameter(torch.zeros(dims[i + 1])) for i in range(n_layers)
        ])
        for w in self.ws:
            nn.init.xavier_uniform_(w)

    def forward(self, x: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
        h = x
        last = len(self.ws) - 1
        for i, (w, b) in enumerate(zip(self.ws, self.bs)):
            h = torch.sparse.mm(P, h)
            h = h @ w + b
            if i < last:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h


@torch.no_grad()
def _acc(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> float:
    if int(mask.sum()) == 0:
        return float("nan")
    pred = logits[mask].argmax(dim=1)
    return float((pred == y[mask]).float().mean().item())


def train_gcn(
    A: sp.spmatrix,
    X: np.ndarray,
    y: np.ndarray,
    split: dict,
    n_layers: int,
    seed: int,
    hidden: int = 32,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    dropout: float = 0.5,
    epochs: int = 200,
    patience: int = 25,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    P = build_renorm(A)
    x = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    tr = torch.tensor(split["train"], dtype=torch.bool)
    va = torch.tensor(split["val"], dtype=torch.bool)
    te = torch.tensor(split["test"], dtype=torch.bool)
    model = GCN(X.shape[1], int(y.max()) + 1, hidden=hidden, n_layers=n_layers, dropout=dropout)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = -1.0
    best_state = None
    best_epoch = 0
    stall = 0
    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        logits = model(x, P)
        loss = F.cross_entropy(logits[tr], y_t[tr])
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            logits = model(x, P)
            val = _acc(logits, y_t, va)
        if val > best_val + 1e-6:
            best_val = val
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            stall = 0
        else:
            stall += 1
            if stall >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(x, P)
        train_a = _acc(logits, y_t, tr)
        val_a = _acc(logits, y_t, va)
        test_a = _acc(logits, y_t, te)
    return {
        "model": f"GCN-{n_layers}",
        "seed": seed,
        "train_acc": train_a,
        "val_acc": val_a,
        "test_acc": test_a,
        "best_epoch": best_epoch,
        "stopped_epoch": epoch,
        "hidden": hidden,
    }


def sgc_accuracy(
    A: sp.spmatrix,
    X: np.ndarray,
    y: np.ndarray,
    split: dict,
    k: int,
    seed: int = 0,
) -> dict:
    S = normalised_adjacency(A, self_loops=True)
    H = np.asarray(X, dtype=np.float64)
    for _ in range(k):
        H = S @ H
    tr, va, te = split["train"], split["val"], split["test"]
    # Fit on train only (no val tuning besides default C).
    clf = LogisticRegression(
        max_iter=400,
        solver="lbfgs",
        C=1.0,
        random_state=seed,
    )
    clf.fit(H[tr], y[tr])
    def acc(mask):
        if int(mask.sum()) == 0:
            return float("nan")
        return float(accuracy_score(y[mask], clf.predict(H[mask])))
    return {
        "model": f"SGC-{k}",
        "seed": seed,
        "train_acc": acc(tr),
        "val_acc": acc(va),
        "test_acc": acc(te),
        "best_epoch": 0,
        "stopped_epoch": 0,
        "hidden": 0,
    }
