# Review of the preprint and repo (2026-11-20)

Repo state at review time: commit 5af62fc (16 Dec 2024) contained only README.md, LICENSE (GPL-3.0), the 9-page PDF preprint and two figures. No code.

## Findings
- **Edge addition formula (PDF pp. 4-5): correct**, verified to machine precision on 30 random ER graphs (n=20), assuming a simple unweighted graph, both endpoints degree >= 1, edge not already present.
- **Edge deletion formula (PDF pp. 7-8): wrong** (failed 30/30, abs error ~3e-3 to 1e-2). Three issues:
  1. Original entry is |l_uv|^2 = 1/(d_u d_v), not 1/((d_u-1)(d_v-1)).
  2. A minus sign is dropped when rewriting -2/((d_u-1)(d_v-1)).
  3. The c_i sums must exclude the uv terms for deletion (a_uv=1); missing terms h_u/d_v + h_v/d_u.
  Corrected version matched 30/30. The deletion prose also says "after the addition of an edge", and d_i=1 (vertex becomes isolated) is not handled.
- Normalised Laplacian definition puts 1 on the diagonal for isolated vertices; under the usual convention it is 0, so the mean eigenvalue changes when isolates are created or destroyed.
- README figures appear to show the combinatorial Laplacian (K_25 spectrum {0, 25x24}), not the normalised one (eigenvalues 0 and 25/24). Axis labels have float artifacts (0.30000000000000004).
- README claims rewiring and comparative experiments that are not in the PDF; typo "a ongoing project"; empty heading.
- PDF metadata has template leftovers (authors "David S. Hippocampus, Elias D. Striatum", subject q-bio).
- No references or related-work section; the introduction promises a Section 4 that doesn't exist; no LaTeX source.

## Prioritised fixes
1. Fix the deletion proof, state the assumptions, handle d_i = 1.
2. Release code and seeds that reproduce the figures; use the normalised Laplacian in the figures.
3. Commit the LaTeX source, add references and related work, clean the PDF metadata.
4. Align the README with the paper.
5. Reconsider the GPL-3.0 licence for a paper-only repo.
