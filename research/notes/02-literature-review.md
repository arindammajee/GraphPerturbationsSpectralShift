# Recent related work (compiled 2026-11-20)

All links were opened and verified at compile time.

## Spectral theory: eigenvalue and eigenspace shifts under perturbation
- Bueno, Furtado, Perez. *Eigenspaces of Preserved Graph Laplacian Eigenvalues under Positive Edge-Weight Addition.* arXiv 2026. https://arxiv.org/abs/2610.14310 (closest recent theory to edge addition, but for the combinatorial Laplacian)
- Van Mieghem, Ke. *Some Laplacian eigenvalues can be computed by matrix perturbation.* arXiv 2025. https://arxiv.org/abs/2504.19610
- Cardoen et al. *Spectral Effects of Heavy-Tailed Vertex Noise in Geometric Graphs.* arXiv 2026. https://arxiv.org/abs/2604.15123
- Tran, Vu. *New matrix perturbation bounds with relative strength: Perturbation of eigenspaces.* arXiv. https://arxiv.org/abs/2409.20207
- Tran, Vu. *Davis-Kahan Theorem under a moderate gap condition.* arXiv 2025. https://arxiv.org/abs/2510.22393

## Variance of Laplacian eigenvalues
- Li, Garcia Trillos, Li, Suchan. *Central limit theorems for the eigenvalues of graph Laplacians on data clouds.* arXiv 2025. https://arxiv.org/abs/2507.18803

## Stability of graph filters and GCNNs
- Testa et al. *Stability of GCNNs through the lens of small perturbation analysis.* ICASSP 2024. https://arxiv.org/abs/2312.12934
- Testa et al. *Robust Filter Design for Graph Signals.* EUSIPCO 2024. https://arxiv.org/abs/2403.16983
- Zhang, Zhao. *Improved Stability Bounds for GCNNs under Graph Perturbations.* ITW 2024. https://doi.org/10.1109/ITW61385.2024.10806975
- Zhang, Kenlay, Zhang, Cucuringu, Dong. *On the Stability of GCNNs: A Probabilistic Perspective.* NeurIPS 2025. https://arxiv.org/abs/2506.01213

## Spectral GNNs
- Qin, Chen, Liu. *Enhancing Spectral GNNs: From Topology and Perturbation Perspectives.* ICML 2025. https://proceedings.mlr.press/v267/qin25a.html
- Hordan et al. *Spectral GNNs are Incomplete on Graphs with a Simple Spectrum.* NeurIPS 2025. https://proceedings.neurips.cc/paper_files/paper/2025/hash/9cbdc427ca38524c9c96d256c3a2ec2c-Abstract-Conference.html

## Spectral adversarial attacks
- Lin, Blaser, Wang. *Graph Structural Attack by Perturbing Spectral Distance.* KDD 2022. https://arxiv.org/abs/2111.00684
- Wang. *SEA: Spectral Edge Attack on GNNs.* arXiv (cite v3). https://arxiv.org/abs/2512.08964v3

## Spectral augmentation and contrastive learning
- Ko, Choi, Kim. *Universal Graph Contrastive Learning with a Novel Laplacian Perturbation.* UAI 2023. https://proceedings.mlr.press/v216/ko23a.html
- Yang, Han, Jin, Liu. *Spectral-Aware Augmentation for Enhanced Graph Representation Learning (GASSER).* CIKM 2024. https://arxiv.org/abs/2310.13845
- Yang et al. *Graph Contrastive Learning with Joint Spectral Augmentation of Attribute and Topology.* AAAI 2025. https://doi.org/10.1609/aaai.v39i20.35506
- Liu et al. *AS-GCL: Asymmetric Spectral Augmentation on Graph Contrastive Learning.* 2025. https://arxiv.org/abs/2502.13525
- Deng et al. *SPGCL: Graph Contrastive Learning via SVD-Guided Structural Perturbation.* arXiv 2026. https://arxiv.org/abs/2602.00064

## Gaps identified
1. No systematic theory of the variance of the full normalised Laplacian spectrum under discrete edge add/delete/rewire, including degree dependence (most work bounds single eigenvalues or uses the combinatorial Laplacian).
2. Eigenvalue variance has not been tested as a cheap proxy or regulariser for GNN stability, or as an edge-fragility score.
3. No tracking of spectral variance in dynamic networks.
