# Things to verify: variance of the normalised Laplacian spectrum under edge perturbations

This plan turns the gaps in *Spectral Impact of Graph Topology Perturbations* (Majee, 2024, repo `GraphPerturbationsSpectralShift`) into concrete, falsifiable claims. The preprint studies \(\Delta\sigma^2\) for the symmetric normalised Laplacian under edge addition and deletion, but the deletion formula is incorrect, isolates and degree-1 vertices are not handled, there is no rewiring analysis, and there are no reproducible experiments.

Nearby literature bounds *individual* eigenvalues or eigenspaces (Weyl, Davis–Kahan, Tran–Vu relative-strength bounds), treats the *combinatorial* Laplacian under edge addition (Bueno et al., 2026), proves CLTs for proximity-graph Laplacian eigenvalues (Li et al., 2025), studies GCNN stability under random edge changes (Testa et al., 2024; Zhang et al., NeurIPS 2025), maximises spectral distance in attacks (Lin et al., KDD 2022), and perturbs the normalised Laplacian for spectral augmentation (GASSER, AS-GCL, Qin et al., ICML 2025). That leaves three gaps:

- **(a)** no systematic theory of the *variance of the full normalised spectrum* under discrete add/delete/rewire, including degree dependence;
- **(b)** eigenvalue variance is unused as a cheap proxy or regulariser for GNN stability, or as an edge-fragility score;
- **(c)** no tracking of spectral variance in dynamic networks.

## Notation

Throughout, \(G=(V,E)\) is a simple undirected unweighted graph on \(n=|V|\) vertices, with adjacency matrix \(A\), degree matrix \(D\), and degrees \(d_i\). The symmetric normalised Laplacian is

\[
\mathcal{L}=I-D^{-1/2}AD^{-1/2},
\]

with the standard convention \(\mathcal{L}_{ii}=0\) when \(d_i=0\) (the off-diagonal formula is otherwise undefined). Write \(\lambda_1\le\cdots\le\lambda_n\) for its eigenvalues, \(\mu=\frac1n\sum_i\lambda_i=\frac1n\mathrm{Tr}(\mathcal{L})\), and

\[
\sigma^2=\frac1n\|\mathcal{L}\|_F^2-\mu^2=\frac1n\sum_{i=1}^n(\lambda_i-\mu)^2.
\]

Let \(n_0=\#\{i:d_i=0\}\). After a perturbation, primes denote the new graph. \(\Delta\sigma^2=\sigma^2(\mathcal{L}')-\sigma^2(\mathcal{L})\). Neighbour-degree sums are \(c_i=\sum_{j\sim i}1/d_j\).

Quick numerical checks used NumPy and NetworkX (seed 0) on small synthetic graphs. They are sanity checks, not proofs, and not results on Cora, CiteSeer, or ogbn-arxiv. Each item is labelled **supported**, **refuted**, **mixed**, or **untested**. Run them with:

```bash
python research/verification/checks.py
```

Results are written to [`check_results.json`](check_results.json) in this directory. The run that this document records took about 0.3 seconds.

---

## Theory

### T1. Closed form for \(\sigma^2\)

**Statement.** With the isolate convention above,

\[
\mu=1-\frac{n_0}{n},\qquad
\|\mathcal{L}\|_F^2=(n-n_0)+2\sum_{\{i,j\}\in E}\frac{1}{d_i d_j},\qquad
\sigma^2=\frac1n\|\mathcal{L}\|_F^2-\mu^2.
\]

If \(n_0=0\), this collapses to \(\mu=1\) and

\[
\sigma^2=\frac{2}{n}\sum_{\{i,j\}\in E}\frac{1}{d_i d_j}.
\]

On a \(d\)-regular graph this is exactly \(\sigma^2=1/d\). Forcing \(\mathcal{L}_{ii}=1\) on isolates (as in the preprint) yields the wrong mean: \(\mu=1\) instead of \(1-n_0/n\).

**Why.** This is the identity that makes every later first-order formula a statement about a sum of \(1/(d_i d_j)\) over edges, and it is the object missing from gap (a).

**How to verify.** Prove it from \(\|\mathcal{L}\|_F^2=\mathrm{Tr}(\mathcal{L}^2)\) and the definition of \(\mathcal{L}\). Numerically, compare the closed form to the population variance of `numpy.linalg.eigvalsh` on ER, BA, WS, a star, a cycle, a complete graph, and a path plus isolates.

**Expected / falsification.** The two computations must agree to numerical precision on every simple graph. Falsified if any connected isolate-free graph has \(|\sigma^2_{\mathrm{eig}}-\frac2n\sum_e 1/(d_i d_j)|\) well above \(10^{-10}\), or if a graph with isolates has \(\mu=1\) under the standard convention.

**Quick check: supported.** Max absolute error \(3.3\times 10^{-16}\) across those graphs. A 6-vertex path plus 3 isolates has standard \(\mu=2/3\) and \(\sigma^2=0.611\); forcing diagonal 1 (preprint style) wrongly gives \(\mu=1\) and \(\sigma^2=0.389\). A 20-cycle has \(\sigma^2=1/2\); \(K_{10}\) has \(\sigma^2=1/9\), both matching \(1/d\).

### T2. Corrected \(\Delta\sigma^2\) for addition and deletion

**Statement.** If \(uv\notin E\) and \(d_u,d_v\ge 1\), then

\[
\Delta\sigma^2_{\mathrm{add}}(uv;G)
=\frac2n\Bigl(c_u h_u^++c_v h_v^++d_u d_v h_u^+ h_v^+\Bigr),
\qquad
h_i^+=\frac{1}{d_i+1}-\frac{1}{d_i}.
\]

If \(uv\in E\) and \(d_u,d_v\ge 2\) (so the deletion creates no isolate), then

\[
\Delta\sigma^2_{\mathrm{del}}(uv;G)
=\frac2n\Bigl(c_u^{\setminus v}h_u^-+c_v^{\setminus u}h_v^--\frac{1}{d_u d_v}\Bigr),
\]

where \(c_u^{\setminus v}=\sum_{j\sim u,\,j\neq v}1/d_j\) and \(h_i^-=\frac{1}{d_i-1}-\frac{1}{d_i}\). Consequently \(\Delta\sigma^2_{\mathrm{add}}(uv;G)=-\Delta\sigma^2_{\mathrm{del}}(uv;G+uv)\). The preprint's deletion formula is incorrect: it uses the denominator \((d_u-1)(d_v-1)\) for the *original* off-diagonal entry, drops a minus sign, and includes the \(j=v\) term in \(c_u\).

**Why.** This repairs the bug found in the review and gives a usable calculus for gap (a). The addition formula in the preprint is already of this form (under the no-isolate hypothesis) and should be kept; the deletion formula should not.

**How to verify.** Algebra from T1, comparing edge terms before and after the degree change. Numerically, on random connected ER graphs, compare both formulae and the preprint deletion formula to the change in the empirical spectral variance.

**Expected / falsification.** Corrected formulae match the eigendecomposition to \(\sim 10^{-15}\); the preprint deletion formula does not. Falsified if the corrected deletion formula systematically misses the empirical \(\Delta\sigma^2\) on graphs with minimum degree at least 2.

**Quick check: supported.** On 25 connected ER graphs (\(n=24\), \(p=0.35\)): addition 25/25, max error \(2.4\times 10^{-16}\); corrected deletion 25/25, max error \(2.2\times 10^{-16}\); preprint deletion 0/25, max error \(1.2\times 10^{-2}\); add-then-delete negation 25/25.

### T3. Rewiring as a composition, and a direct formula

**Statement.** The rewire “delete \(uv\), add \(uw\)” (with \(w\neq v\), \(uw\notin E\), \(d_v\ge 2\)) satisfies

\[
\Delta\sigma^2_{\mathrm{rewire}}(uv\to uw;G)
=\Delta\sigma^2_{\mathrm{del}}(uv;G)+\Delta\sigma^2_{\mathrm{add}}(uw;G-uv).
\]

The second term *must* be evaluated on \(G-uv\), not on \(G\): after the deletion, \(d_u\) and \(d_v\) have already changed. Because this rewire leaves \(d_u\) unchanged and only shifts one incident edge from \(v\) to \(w\), there is also a direct formula from T1:

\[
\begin{aligned}
\frac n2\Delta\sigma^2_{\mathrm{rewire}}
&= -\frac{1}{d_u d_v}+\frac{1}{d_u(d_w+1)}
+\sum_{\substack{x\sim v\\ x\neq u,w}}\frac{1}{d_x\,d_v(d_v-1)}
-\sum_{\substack{y\sim w\\ y\neq u,v}}\frac{1}{d_y\,d_w(d_w+1)}\\
&\quad+\mathbf{1}_{vw\in E}\left(\frac{1}{(d_v-1)(d_w+1)}-\frac{1}{d_v d_w}\right).
\end{aligned}
\]

**Why.** Rewiring is the natural move for spectral augmentation and for “which edge to swap” in gap (a); the preprint never treats it.

**How to verify.** Prove the composition by telescoping and the direct formula by comparing \(\sum_e 1/(d_i d_j)\) termwise (degrees change only at \(v\) and \(w\)). Numerically, draw random legal triples \((u,v,w)\) on BA graphs and compare composition, direct formula, and the naive “sum of add and delete formulae on the original \(G\)” to the eigendecomposition.

**Expected / falsification.** Composition and the direct formula agree with the spectrum; the naive sum on \(G\) does not, except by accident. Falsified if either identity fails on a legal rewire that creates no isolates.

**Quick check: supported.** 20 random rewires on BA (\(n=22\), \(m=2\)): composition 20/20 and direct formula 20/20, max error \(3.1\times 10^{-16}\). Naive sum on the original graph: 0/20, max error \(1.4\times 10^{-3}\). Example: on degrees \((d_u,d_v,d_w)=(4,3,3)\), actual \(\Delta\sigma^2=-2.23\times 10^{-3}\), naive sum \(-1.26\times 10^{-3}\).

### T4. Sign of \(\Delta\sigma^2\) and degree dependence

**Statement.** For an addition with \(d_u,d_v\ge 1\),

\[
\mathrm{sign}\bigl(\Delta\sigma^2_{\mathrm{add}}(uv;G)\bigr)
=\mathrm{sign}\left(1-\frac{d_v+1}{d_u}c_u-\frac{d_u+1}{d_v}c_v\right).
\]

Consequences:

1. On a \(d\)-regular graph, \(c_u=c_v=1\), so the quantity equals \(1-2(d+1)/d=-1-2/d<0\): *every* legal addition strictly decreases \(\sigma^2\). Every deletion strictly increases it, with the constant value \(\Delta\sigma^2_{\mathrm{del}}=(2/n)d^{-2}\).
2. Adding an edge between two leaves of a star with hub degree \(d_h\) has sign equal to the sign of \(1-4/d_h\): negative for \(d_h<4\), zero at \(d_h=4\), positive for \(d_h>4\).
3. On heterogeneous graphs both signs occur. In particular \(\Delta\sigma^2_{\mathrm{add}}\) can be positive when both endpoints have small degree and high-degree neighbours (small \(c_u,c_v\)).

**Why.** Gap (a) is not just a formula for \(\Delta\sigma^2\) but its sign and degree dependence, which later items use for fragility scores and augmentation.

**How to verify.** Substitute \(c_i=d\cdot(1/d)=1\) on regular graphs; compute the star case in closed form; scan all non-edges of a BA graph. Falsify (1) by finding a regular graph and a non-edge with \(\Delta\sigma^2\ge 0\); falsify (2) if the star sign change is not at hub degree 4.

**Quick check: supported.** All 240 additions on random regular graphs (\(n,d)\in\{(16,3),(20,4),(18,5)\}\) were negative, and the sign agreed with the criterion in every case. All 40 deletions on a 4-regular graph with \(n=20\) had \(\Delta\sigma^2=0.00625=(2/20)\cdot 4^{-2}\) exactly. Star leaf–leaf additions: \(\Delta\sigma^2=-0.0417,0,0.0167,0.0238,\ldots\) for hub degrees \(3,4,5,6,\ldots\), changing sign exactly as predicted. On one BA graph (\(n=40\), \(m=2\)), 146 of 704 legal non-edges increased \(\sigma^2\) and 557 decreased it (mean \(\Delta\sigma^2=-1.52\times 10^{-3}\)).

### T5. Which edge maximises or minimises \(\Delta\sigma^2\)

**Statement (conjecture, qualitative).** On a degree-heterogeneous graph, the non-edges with the *largest* (most positive) \(\Delta\sigma^2_{\mathrm{add}}\) join two low-degree vertices whose neighbours have high degree. The most *negative* additions join mid-to-high degree vertices with relatively low-degree neighbours. For deletions, the largest *increase* in \(\sigma^2\) tends to involve at least one high-degree endpoint, but this correlation is weaker. A complete characterisation of the argmax/argmin (for example, that a leaf–leaf edge at a common hub is always maximal among additions) is left as a lemma to prove or refute.

**Why.** An edge-fragility score and a spectral-augmentation policy both need to know which move is extremal (gap (b)).

**How to verify.** On BA, SBM, and real graphs (Cora, CiteSeer), enumerate legal adds/deletes (or a large sample on bigger graphs), record \((d_u,d_v,\overline{d}_{N(u)\cup N(v)},\Delta\sigma^2)\), and report Spearman correlations and the top-\(k\) extremal edges. Try to prove a comparison lemma from the sign criterion in T4. Falsify the qualitative claim if, on typical BA samples, the non-edges with maximal \(\Delta\sigma^2\) are *not* low-degree pairs with high-degree neighbours.

**Expected / falsification.** A stable covariate pattern across models; or a clean proof that the sign quantity in T4 is monotone in the neighbour-harmonic means. Falsified by a counterexample class where hub–hub additions are the most positive, or where the neighbour-degree correlation flips sign.

**Quick check: mixed.** On one BA graph (\(n=36\), \(m=2\), 562 non-edges), \(\Delta\sigma^2_{\mathrm{add}}\) ranged from \(-6.05\times 10^{-3}\) to \(+2.44\times 10^{-3}\). Spearman correlation of \(\Delta\sigma^2\) with mean neighbour degree was \(+0.82\); with \(d_u+d_v\) it was \(-0.52\). The eight most positive additions were all between two degree-2 vertices with mean neighbour degree \(8\)–\(10.3\). The most negative additions had mid degrees (2–5) and *low* mean neighbour degree (\(\approx 3.1\)–\(4.3\)). Deletion \(\Delta\sigma^2\) versus \(d_u+d_v\) had only a weak Pearson correlation of \(0.27\). This supports the qualitative addition claim on *one* BA sample; it is not a theorem, and the deletion side is unsettled.

### T6. Isolates and degree-1 endpoints

**Statement.** The no-isolate formulae of T2 are not valid when an addition touches an isolated vertex, or when a deletion has \(\min(d_u,d_v)=1\). In those cases \(n_0\) changes, so \(\mu\) changes: adding an edge to an isolate increases the trace by 1 and \(\mu\) by \(1/n\); deleting a leaf edge decreases them by the same amount and makes \(h_i^-\) undefined at the leaf. The *general* closed form of T1 still holds and should be used instead. A separate (elementary) lemma should give \(\Delta\sigma^2\) for “connect an isolate to \(u\)” and “detach a leaf from \(u\)”.

**Why.** This is the domain error identified in the review; any algorithm that scores every possible edge must handle leaves.

**How to verify.** Derive the isolate/leaf cases from T1 (the new vertex contributes a diagonal 1 and a single off-diagonal term, and \(d_u\) increases by 1). Numerically, start from a path plus an isolate and from a star, and check that \(\mu\) moves by \(1/n\) and that the T2 deletion formula refuses (or would be wrong on) a leaf edge.

**Expected / falsification.** \(\mu=1-n_0/n\) always; T2's deletion formula is inapplicable at degree 1. Falsified if attaching an isolate leaves \(\mu\) unchanged under the standard convention.

**Quick check: supported.** A 5-path plus one isolate has \(\mu=5/6\). Connecting the isolate to a leaf, or to an internal vertex, both lift \(\mu\) to \(1\) (trace \(5\to 6\)), with \(\Delta\sigma^2=-0.0556\) and \(-0.0833\) respectively. Deleting a leaf from a 6-leaf star drops \(\mu\) from \(1\) to \(6/7\), raises \(\sigma^2\) from \(0.286\) to \(0.408\), creates one isolate, and the corrected T2 formula raises `ValueError` as intended. The general T1 closed form still matches the spectrum (error \(2.8\times 10^{-16}\)).

---

## Numerical checks

### N1. Asymptotic \(\sigma^2\) versus density in ER graphs

**Statement.** For \(G\sim G(n,p)\) in the connected, isolate-free regime (\(np\gg\log n\)),

\[
\sigma^2(G)\;\sim\;\frac{1}{np}\;\approx\;\frac{1}{\langle d\rangle}.
\]

A slightly sharper first-order statement is \(\sigma^2=1/\langle d\rangle\cdot(1+o(1))\), because given that \(uv\) is an edge, \(d_u=1+\mathrm{Bin}(n-2,p)\). The crude \(1/(np)\) form is already the right scale.

**Why.** This is the baseline law for gap (a) and the null trajectory for dynamic tracking (gap (c)). If \(\sigma^2\) is just \(1/\langle d\rangle\), it carries no extra structural information on ER.

**How to verify.** For several \(n\) (e.g. 80, 200, 500) and a grid of \(p\) with no isolates, average \(\sigma^2\) over many ER draws and plot \(\sigma^2\cdot np\) and \(\sigma^2\cdot\langle d\rangle\) against \(p\). Fit a slope in log-log scale. Falsify \(1/(np)\) scaling if the product systematically drifts away from 1 as \(n\) grows at fixed \(np\) in the connected regime, or if a different exponent is clearly better.

**Expected / falsification.** \(\sigma^2\cdot\langle d\rangle\to 1\). At moderate \(n\), \(1/\langle d\rangle\) should already be closer than \(1/(np)\) because of degree variability.

**Quick check: supported** (at \(n=80\), 6 replicates, \(p\in[0.12,0.75]\)). The product \(\sigma^2\cdot np\) lay in \([0.996,\,1.025]\); \(\sigma^2\cdot\langle d\rangle\) was even tighter, in \([0.9985,\,1.0000]\). Example: at \(p=0.2\), mean \(\sigma^2=0.0627\) versus \(1/(np)=0.0625\). This is a finite-\(n\) check, not an asymptotic proof.

### N2. Expected \(\Delta\sigma^2\) and concentration under random addition, ER versus BA

**Statement.** Let \(uv\) be a uniform random legal non-edge and consider \(\Delta\sigma^2_{\mathrm{add}}(uv;G)\). On a typical ER graph, this random variable is concentrated (all mass on a small negative interval, coefficient of variation \(\ll 1\)). On a BA graph of similar \(n\), the distribution is much more spread, can take both signs, and the mean depends on the degree class of the endpoints (hub–hub and hub–mid additions more negative; leaf–mid more positive). In ER one also expects \(\mathbb{E}[\Delta\sigma^2]\sim -C/(n^2 p^2)\) or a nearby scale, to be derived from T2 and degree concentration.

**Why.** Gap (a) asks for more than a pointwise formula: an average-case and concentration statement, and a comparison of homogeneous versus heavy-tailed degree sequences.

**How to verify.** On many ER and BA draws (and later WS, SBM), sample hundreds of non-edges, report mean, standard deviation, CV, and 5th/95th percentiles, and bucket BA pairs as leaf/mid/hub. Derive \(\mathbb{E}[\Delta\sigma^2]\) for \(G(n,p)\) from T2. Falsify the concentration comparison if, at matched \(n\) and \(\langle d\rangle\), ER does not show a clearly smaller CV than BA, or if BA additions never change sign.

**Expected / falsification.** \(\mathrm{CV}_{\mathrm{BA}}>\mathrm{CV}_{\mathrm{ER}}\); ER samples sit almost entirely on \(\Delta\sigma^2<0\).

**Quick check: supported** (one ER \(n=60,p=0.2\) and one BA \(n=60,m=3\), 250 random non-edges each). ER: mean \(\Delta\sigma^2=-2.74\times 10^{-4}\), std \(4.48\times 10^{-5}\), CV \(0.16\), *all 250 additions negative*. BA: mean \(-6.34\times 10^{-4}\), std \(5.38\times 10^{-4}\), CV \(0.85\), range \([-1.69,\,+0.97]\times 10^{-3}\). BA buckets: hub–mid mean \(-9.8\times 10^{-4}\) (\(n=41\)); mid–mid \(-6.0\times 10^{-4}\) (\(n=200\)); leaf–mid \(+3.9\times 10^{-4}\) (\(n=7\), small cell). The ER scaling law for the mean was not fitted here.

### N3. Relation of \(\Delta\sigma^2\) to \(\sum_i(\lambda_i'-\lambda_i)^2\)

**Statement.** Let \(\lambda,\lambda'\) be the *sorted* spectra. Then in general

\[
\Delta\sigma^2\;\neq\;\frac1n\sum_{i=1}^n(\lambda_i'-\lambda_i)^2.
\]

The right-hand side is a spectral distance (the quantity a Lin-style attack can target). When \(\mu\) is unchanged (\(n_0\) fixed), the exact identity is

\[
n\Delta\sigma^2=\sum_{i=1}^n\bigl(\lambda_i'^2-\lambda_i^2\bigr).
\]

Hoffman–Wielandt still gives \(\sum_i(\lambda_i'-\lambda_i)^2\le\|\mathcal{L}'-\mathcal{L}\|_F^2\). There is no reason to expect \(|\Delta\sigma^2|\) to be a monotone proxy for the spectral distance, so their rank correlation may be weak.

**Why.** This is the bridge — or the separation — between gap (a) and gap (b). If the correlation is weak, \(\Delta\sigma^2\) cannot be advertised as a drop-in cheap surrogate for spectral-distance attacks without further work.

**How to verify.** On BA/ER and later on Cora, for many single-edge deletions compute \(\Delta\sigma^2\), \(\sum_i(\lambda_i'-\lambda_i)^2\), and \(\|\mathcal{L}'-\mathcal{L}\|_F^2\). Check the identity and Hoffman–Wielandt, and report Spearman correlations. Falsify the identity if it fails at fixed \(\mu\); falsify Hoffman–Wielandt if the inequality is violated for Hermitian \(\mathcal{L}\). The proxy claim is falsified by consistently low rank correlation.

**Expected / falsification.** Identity and Hoffman–Wielandt hold; the Spearman correlation between \(|\Delta\sigma^2|\) and \(\sum_i(\Delta\lambda_i)^2\) may be modest.

**Quick check: supported** for the identities, and already evidence *against* a tight proxy. On 80 deletions in a BA graph (\(n=40\), \(m=3\)): Hoffman–Wielandt held; \(n\Delta\sigma^2=\sum_i(\lambda_i'^2-\lambda_i^2)\) to max error \(2.4\times 10^{-14}\). Spearman of \(|\Delta\sigma^2|\) versus \(\sum_i(\lambda_i'-\lambda_i)^2\) was only \(0.14\); of signed \(\Delta\sigma^2\) versus that distance, \(0.07\). The ratio \(\sum_i(\Delta\lambda_i)^2\big/\big(n|\Delta\sigma^2|\big)\) ranged from \(0.01\) to \(0.95\) (median \(0.032\)). The two quantities are related but far from interchangeable.

### N4. Does \(\sigma^2\approx 1/\langle d\rangle\) hold beyond ER?

**Statement.** The regular-graph law \(\sigma^2=1/d\) suggests the approximation \(\sigma^2\approx 1/\langle d\rangle\) on nearly regular models (WS with small rewiring probability). It need *not* hold under heavy-tailed degrees: on BA, edges are biased toward hubs, so \(\sum_e 1/(d_i d_j)\) is smaller than \(|E|/\langle d\rangle^2\) and \(\sigma^2\) should be *strictly below* \(1/\langle d\rangle\). Two-block SBMs with comparable intra-block densities may still sit near \(1/\langle d\rangle\), because they are locally ER-like.

**Why.** If \(\sigma^2\) is a function of mean degree alone, gap (c) tracking is uninteresting on any densifying network. A deviation on BA would show that \(\sigma^2\) carries an extra, degree-assortative signal.

**How to verify.** Average \(\sigma^2\cdot\langle d\rangle\) over WS, BA, and SBM ensembles (and, later, Cora, CiteSeer, ogbn-arxiv). Falsify “always \(\approx 1/\langle d\rangle\)” if BA (or a real citation graph) sits well away from 1; falsify the BA-direction claim if BA ratios are not systematically below 1.

**Expected / falsification.** WS ratio \(\approx 1\); BA ratio \(<1\) and not approaching 1 at these \(n\); SBM with homogeneous blocks close to 1.

**Quick check: mixed** (as a universal scaling), and in line with the refined claim. Five replicates, \(n=60\): WS ratios in \([0.9990,\,1.0001]\) for \(k\in\{4,6,8\}\); two-block SBM also \(\approx 1.000\); BA \(m=2\) ratio \(0.863\), BA \(m=3\) ratio \(0.897\). So \(1/\langle d\rangle\) is excellent on WS and these SBMs, and a 10–14% overestimate on BA.

---

## Applied and GNN experiments

### A1. \(\Delta\sigma^2\) as an edge-fragility score versus a spectral-distance attack

**Statement.** Rank candidate deletions (or additions) by \(\Delta\sigma^2\), by \(|\Delta\sigma^2|\), or by \(-\Delta\sigma^2\), and compare that ranking to the ranking by the sorted-spectrum distance \(\|\lambda'-\lambda\|_2=\bigl(\sum_i(\lambda_i'-\lambda_i)^2\bigr)^{1/2}\) (the objective of a Lin et al. style spectral attack). The T2 formula makes the \(\Delta\sigma^2\) score \(O(\mathrm{deg})\) after a degree pass, whereas the spectral distance needs an eigendecomposition per candidate edge. The claim to test is that a \(\Delta\sigma^2\)-based ranking recovers a useful fraction of the top-\(k\) spectral-distance edges, cheaply enough to be used as a fragility score.

**Why.** This is the first half of gap (b). N3 already suggests the two scores are not the same functional, so this item is a ranking experiment, not a corollary.

**How to verify.** (i) Small synthetic graphs: enumerate edges, report Spearman correlation and top-\(k\) overlap. (ii) Real graphs: Cora, CiteSeer, and a subsample of ogbn-arxiv; compare full or sampled rankings; optionally measure downstream node-classification accuracy after deleting the top-\(k\) edges of each score (GCN/GAT, standard splits). Falsify usefulness if top-\(k\) overlap stays near chance and a GNN attacked by the \(\Delta\sigma^2\) ranking loses little accuracy compared with a spectral-distance or random attack.

**Expected / falsification.** A usable score would show substantial top-\(k\) overlap (say \(\ge 0.5\) at small \(k\)) and a clear accuracy drop relative to random deletions. Given N3, a weak correlation on some graphs would not be surprising.

**Quick check: mixed**, and already cautious. On one BA graph (\(n=32\), \(m=2\), 60 scored edges), Spearman of \(|\Delta\sigma^2|\) versus spectral distance was only \(0.24\) (Pearson \(0.16\)). Top-\(k\) overlap when ranking by \(|\Delta\sigma^2|\) was \(0/3\), \(1/5\), \(2/8\). Ranking by \(-\Delta\sigma^2\) did no better at \(k=5,8\). This is not a reproduction of Lin et al., and it is not a Cora experiment; it already suggests that \(\Delta\sigma^2\) is a *poor* stand-in for spectral-distance ranking on this graph.

### A2. A variance regulariser for GNN robustness — *untested*

**Statement (experiment, not a theorem).** Adding a penalty \(\gamma\sigma^2(\mathcal{L})\) (or \(\gamma\sigma^2(\mathcal{L}_{\hat{A}})\) on a learned or attention-weighted adjacency) to a GCN/GAT node-classification loss, or rejecting training-time edge perturbations that would increase \(\sigma^2\), improves robustness to random edge additions/deletions (the Testa et al. / Zhang et al. setting) and possibly to spectral-distance attacks (Lin et al.), at a modest cost in clean accuracy. Because T4 says that most random additions on homogeneous graphs *decrease* \(\sigma^2\), a regulariser that only shrinks \(\sigma^2\) may be too crude; a more refined version would penalise *increase* in \(\sigma^2\) under a neighbourhood of rewires, or would use the T2 score as a differentiable surrogate on soft adjacencies.

**Why.** Gap (b): variance is unused as a regulariser. This is also the place to connect, rather than duplicate, Qin et al. / GASSER / AS-GCL, which already perturb the normalised Laplacian for augmentation.

**How to verify.** Datasets: Cora, CiteSeer, ogbn-arxiv (standard splits). Models: GCN and GAT, with and without the regulariser, swept over \(\gamma\). Noise: random edge drop/add at several rates; plus a spectral-distance targeted perturbation of a given budget. Metrics: clean accuracy, attacked accuracy, and the realised \(\sigma^2\) of the graph (or of the learned adjacency). Baselines: vanilla training, DropEdge, and at least one published spectral augmentation. Report means and standard deviations over seeds. Falsify if, at matched clean accuracy, robust accuracy is no better than DropEdge / published spectral augmentation, or if the regulariser does not actually move \(\sigma^2\).

**Expected / falsification.** Untested — no GNN was trained in the quick checks. A negative outcome is plausible: N3 and A1 suggest \(\sigma^2\) is only weakly aligned with spectral distance, so a \(\sigma^2\) penalty may not defend against a Lin-style attack.

### A3. Tracking \(\sigma^2\) in dynamic networks

**Statement.** On a graph that densifies by random edge arrival (an evolving ER), \(\sigma^2(t)\) closely tracks \(1/\langle d(t)\rangle\) and therefore falls as \(\Theta(1/(np(t)))\). On a dynamic network with changing degree heterogeneity or community structure, \(\sigma^2(t)\cdot\langle d(t)\rangle\) should *move*, and those moves are the interesting signal (see N4). The quantity is cheap to maintain incrementally via T2 as edges arrive or leave.

**Why.** Gap (c). The ER case is the null; real contact, message, or co-authorship graphs are the test.

**How to verify.** (i) Synthetic: start from a sparse connected ER graph and add random edges, recording \(\sigma^2\), \(\langle d\rangle\), and algebraic connectivity. (ii) Real temporal networks (e.g. SNAP email or co-authorship snapshots): plot \(\sigma^2(t)\), \(1/\langle d(t)\rangle\), their ratio, and event markers; test whether the residual \(\sigma^2-1/\langle d\rangle\) forecasts a downstream change (a newly forming community, a burst of activity). Falsify interest in (ii) if the residual is noise and \(\sigma^2\) is interchangeable with mean degree. Falsify the ER null if a growing ER trajectory does *not* hug \(1/\langle d\rangle\).

**Expected / falsification.** Synthetic ER: Pearson correlation of \(\sigma^2\) with \(1/\langle d\rangle\) near 1, and a falling trajectory. Real data: untested.

**Quick check: supported** for the synthetic null only. Growing an ER graph on \(n=50\) vertices from 99 edges (\(\langle d\rangle=3.96\), \(\sigma^2=0.252\)) to 428 edges (\(\langle d\rangle=17.12\), \(\sigma^2=0.058\)) over 23 snapshots, \(\sigma^2\cdot\langle d\rangle\) stayed at \(0.999\), and the Pearson correlation of \(\sigma^2\) with \(1/\langle d\rangle\) was \(0.9999\). No real dynamic network was examined.

### A4. \(\Delta\sigma^2\)-guided spectral augmentation — *untested*

**Statement (experiment).** A training-time augmentation that rewires (or adds/deletes) edges to *decrease* \(\sigma^2\), or that mixes decreases with a few T4-style leaf–leaf increases, is a controllable alternative to generic normalised-Laplacian perturbations (GASSER, AS-GCL, Qin et al., ICML 2025). Because a rewire can be scored in closed form (T3), the policy can be greedy or sampled proportional to \(-\Delta\sigma^2\) without an eigendecomposition.

**Why.** Complements A2 and situates the calculus of gap (a) against existing spectral augmentation (gap (b)), rather than ignoring that line of work.

**How to verify.** On Cora, CiteSeer, and ogbn-arxiv, compare node-classification accuracy and robustness (same attack suite as A2) for: no augmentation; random rewires; greedy \(\Delta\sigma^2\)-reducing rewires; a mixed sign policy; and a published spectral-augmentation baseline, at a matched perturbation budget. Falsify if \(\Delta\sigma^2\)-guided rewires do not beat random rewires on clean or robust accuracy.

**Expected / falsification.** Untested. No augmentation loop was run.

---

## What the quick checks do not show

- No result above uses Cora, CiteSeer, ogbn-arxiv, or any trained GNN. A2 and A4 are experiment designs only.
- “Supported” means the claim held on the small synthetic instances in `checks.py` (seed 0). It is not a proof of an asymptotic statement (N1, N2) and not a licence to treat T5 as a theorem.
- A1’s weak top-\(k\) overlap is a warning, not a final evaluation of fragility scores.
- Incremental complexity, directed or weighted graphs, and multiplicity of eigenvalues are out of scope for the checks.

## Suggested order of work

1. Write T1–T4 and T6 as short lemmas, with the preprint deletion formula discussed only as a correction.
2. Prove or abandon a comparison lemma for T5; if it resists a clean statement, keep the covariate analysis.
3. Upgrade N1–N4 to larger \(n\) and add the three citation graphs as observational points (no training yet).
4. Run A1 on Cora/CiteSeer before investing in A2 or A4; if the ranking overlap remains near chance, treat \(\Delta\sigma^2\) as a *different* design objective, not as a cheap spectral-distance proxy.
5. Only then train the A2 regulariser and the A4 augmentation, and finally test A3 on a real temporal network, using \(\sigma^2\cdot\langle d\rangle\) as the residual of interest.
