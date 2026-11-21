# Status and next steps (as of 2026-11-21)

## Key results so far
- Closed form (no isolates): sigma^2 = (2/n) * sum_{(i,j) in E} 1/(d_i d_j). Exact local updates for add/delete/rewire follow. Verified numerically.
- Sign laws: on d-regular graphs every addition lowers sigma^2; star leaf-leaf addition raises sigma^2 iff hub degree > 4; both signs occur on BA graphs.
- ER scaling: sigma^2 * <d> ~= 1 (also on WS); BA graphs fall 10-14% below.
- sigma^2 change is a weak proxy for spectral distance (Spearman 0.14) and a weak edge-fragility score.

## Over-smoothing pilot ([pilot_oversmoothing/RESULTS.md](../pilot_oversmoothing/RESULTS.md)): weak or negative
- Feature-collapse rate tracks the spectral gap (r=0.93), not sigma^2 (r=-0.18), across 10 datasets.
- sigma^2 does not track heterophily, nor GCN accuracy after controlling for homophily.
- Lowering sigma^2 at a fixed edge count widened the spectral gap on Cora, Chameleon and Texas; deep SGC accuracy on Cora tracked the gap.

## Proposed next step (not yet started)
Test sigma^2-guided rewiring as a cheap, eigendecomposition-free way to widen the spectral gap, compared with FoSR and other gap-maximising rewiring (gap gain per edit, runtime, downstream accuracy on heterophilic benchmarks), framed against the over-squashing literature.

## Path to a NeurIPS-level paper (discussed)
1. Theory: exact local update, sign laws, scaling and why BA deviates (cite the Randic-index literature for sum 1/(d_i d_j)).
2. Link to learning (the over-smoothing link was not supported; the gap-widening angle is the current candidate).
3. A practical rewiring method benchmarked against existing rewiring and augmentation.
