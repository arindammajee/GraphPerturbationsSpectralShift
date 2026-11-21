# Research log

Follow-up work on the preprint [*Spectral Impact of Graph Topology Perturbations*](../Spectral_Impact_of_Graph_Topology_Perturbations.pdf) (Majee). This directory holds the review of that preprint, related literature, numerical checks of the spectral-variance claims, and a pilot on whether eigenvalue variance predicts GNN over-smoothing. The numbers and verdicts below are the ones recorded on 20–21 November 2026.

## Current status

- Closed form (no isolates): \(\sigma^2 = (2/n) \sum_{(i,j)\in E} 1/(d_i d_j)\). Exact local updates for edge addition, deletion, and rewiring follow, and the numerical checks support them. The preprint's edge-deletion formula is wrong; the corrected formula matched the spectrum on every trial in the review and in check T2.
- Sign laws hold on the synthetic checks: every addition on a \(d\)-regular graph lowers \(\sigma^2\); a star leaf–leaf addition raises \(\sigma^2\) exactly when the hub degree is greater than 4; both signs occur on Barabási–Albert graphs. On ER (and Watts–Strogatz) graphs, \(\sigma^2 \langle d\rangle \approx 1\); BA graphs fall 10–14% below that line.
- A change in \(\sigma^2\) is a weak proxy for spectral distance (Spearman 0.14 on the quick check) and a weak edge-fragility score.
- The over-smoothing pilot's verdict is **weak or negative**. Across 10 datasets the feature-collapse rate tracks the spectral gap (\(r = 0.93\)), not \(\sigma^2\) (\(r = -0.18\)). \(\sigma^2\) does not track heterophily, nor GCN accuracy after controlling for homophily. Lowering \(\sigma^2\) at a fixed edge count widened the spectral gap on Cora, Chameleon, and Texas; deep SGC accuracy on Cora tracked the gap.
- Proposed next step, not started: test \(\sigma^2\)-guided rewiring as a cheap, eigendecomposition-free way to widen the spectral gap, compared with FoSR and other gap-maximising rewiring.

Detail is in [notes/03-status-and-next-steps.md](notes/03-status-and-next-steps.md) and [pilot_oversmoothing/RESULTS.md](pilot_oversmoothing/RESULTS.md).

## What is where

| Path | Contents |
| --- | --- |
| [notes/01-repo-review.md](notes/01-repo-review.md) | Review of the preprint and the repo as of commit `5af62fc`, including the edge-deletion bug and the correction that matched 30/30 graphs. |
| [notes/02-literature-review.md](notes/02-literature-review.md) | Related work with links (checked 2026-11-20) and the three gaps this project sits in. |
| [notes/03-status-and-next-steps.md](notes/03-status-and-next-steps.md) | Results so far and the proposed next direction. |
| [verification/spectral-variance-verification-plan.md](verification/spectral-variance-verification-plan.md) | Fourteen testable claims (T1–T6, N1–N4, A1–A4), each marked supported, mixed, or untested. |
| [verification/checks.py](verification/checks.py) | Seed-0 numerical checks for the claims that were run. |
| [verification/check_results.json](verification/check_results.json) | Output of that run. The beyond-ER scaling check (plan section N4) is stored as item `N5`. A2 and A4 were left untested. |
| [pilot_oversmoothing/](pilot_oversmoothing/) | CPU pilot: code, [requirements.txt](pilot_oversmoothing/requirements.txt), [RESULTS.md](pilot_oversmoothing/RESULTS.md), figures, and saved results. |

Pilot modules: `datasets.py` (loaders), `graph_utils.py` (spectrum and \(\sigma^2\)), `oversmooth.py` (Dirichlet curves), `gnn.py` (GCN and SGC), `rewire.py` (T2-scored rewires), `run_pilot.py` (full pilot), `run_sgc_interventions.py` (intervention SGC rows), `write_report.py` (draft report from the JSON).

Figures: [sigma2_vs_oversmoothing.png](pilot_oversmoothing/figures/sigma2_vs_oversmoothing.png), [intervention_oversmoothing_curves.png](pilot_oversmoothing/figures/intervention_oversmoothing_curves.png).

Saved tables and the small run logs (about 5–37 KB) are in [pilot_oversmoothing/results/](pilot_oversmoothing/results/). `train_rows.json` is the complete training table. The partial checkpoint `train_rows_partial.json` and the earlier `train_rows_gcn_backup.json` were left out. Graph datasets are not in the repo.

## Rerun the verification checks

From the repository root, install [pilot_oversmoothing/requirements.txt](pilot_oversmoothing/requirements.txt) and run the script. It writes `research/verification/check_results.json` next to itself. The recorded run took about 0.3 seconds (seed 0). A fresh run overwrites that file; wall-clock times will differ.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r research/pilot_oversmoothing/requirements.txt
python research/verification/checks.py
```

The checks import only NumPy and NetworkX. PyTorch, SciPy, scikit-learn, and Matplotlib are there for the pilot.

## Rerun the over-smoothing pilot

The recorded pilot used a 38-minute CPU budget. The saved JSON, logs, and figures are already in this tree, so reading [RESULTS.md](pilot_oversmoothing/RESULTS.md) does not require a rerun. To run it again, use the same virtualenv and put the datasets under `research/pilot_oversmoothing/data/` (gitignored):

- `data/planetoid/` — Planetoid files `ind.{cora,citeseer,pubmed}.{x,y,tx,ty,allx,ally,graph,test.index}`
- `data/geom_gcn/new_data/{texas,wisconsin,cornell,chameleon,squirrel}/` — `out1_node_feature_label.txt` and `out1_graph_edges.txt` (geom-GCN dumps; Chameleon and Squirrel from commit `f1fc0d1`)
- `data/geom_gcn/splits/` — `{name}_split_0.6_0.2_{0..9}.npz`
- `data/heterophilous/` — `roman_empire.npz` and `amazon_ratings.npz` (Platonov et al.)

```bash
python research/pilot_oversmoothing/run_pilot.py
```

`run_sgc_interventions.py` rebuilds the intervention SGC rows and updates `results/train_rows.json`. `write_report.py` rebuilds the two figures and a draft `RESULTS.md` from the JSON, with no training. The `RESULTS.md` committed here is the curated write-up of the 2026-11-20 run; running `write_report.py` replaces that file with the generator's draft.
