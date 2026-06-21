# Stage 3: Embedding EDA Findings

**Date:** 2026-06-22 04:07
**Status:** Complete

---

## Dataset Summary

| Metric | Value |
|:---|---:|
| Embedding Model | Qwen/Qwen3-Embedding-0.6B |
| Embedding Dimension | 1024 |
| Total Embeddings | 233,863 |
| Person Names | 107,563 |
| Merchant Names | 126,300 |

## Separability Metrics

| Metric | Value | Interpretation |
|:---|---:|:---|
| Silhouette Score | 0.0607 | Weak separation |
| Calinski-Harabasz | 1768.1 | Higher = better |
| Fisher Discriminant | 73.5996 | Higher = more separable |
| **LR Accuracy (5-fold)** | **0.9275** | +/- 0.0035 |

## Figures

| # | Figure | Description |
|:---|:---|:---|
| 1 | `01_umap_person_vs_merchant.png` | UMAP 2D projection colored by class |
| 2 | `02_umap_augmentation_impact.png` | Original vs augmented names in UMAP space |
| 3 | `03_umap_merchant_categories.png` | Merchant sub-categories in UMAP space |
| 4 | `04_tsne_person_vs_merchant.png` | t-SNE 2D projection |
| 5 | `05_pca_analysis.png` | PCA variance curve + 2D projection |
| 6 | `06_cosine_similarity_distributions.png` | Intra vs inter-class similarity |
| 7 | `07_separability_metrics.png` | Separability metric bar charts |
| 8 | `08_nearest_neighbor_analysis.png` | Cross-class nearest neighbor distances |
| 9 | `09_embedding_norms.png` | Embedding norm distributions |
| 10 | `10_fisher_per_dimension.png` | Per-dimension Fisher discriminant scores |
| 11 | `11_ttest_per_dimension.png` | Per-dimension t-test significance |
| 12 | `12_lr_coefficients.png` | Logistic Regression coefficient weights |
| 13 | `13_dimension_ablation.png` | Top-K vs Bottom-K vs Random-K accuracy |
| 14 | `14_dimension_distributions.png` | Value distributions of best vs worst dims |
| 15 | `15_method_agreement.png` | Cross-method ranking agreement |

## Dimension Importance Analysis

| Metric | Value |
|:---|---:|
| Statistically significant dims (Bonferroni) | 938/1024 |
| Non-significant dims | 86 |
| Top 5 Fisher dims | dim_32, dim_416, dim_6, dim_24, dim_201 |
| Top 5 LR coef dims | dim_350, dim_7, dim_534, dim_500, dim_64 |
| Consensus top-50 (all 3 methods) | 4 dims |
| 80% Fisher discriminability | 320 dims |
| 90% Fisher discriminability | 454 dims |
| 95% Fisher discriminability | 562 dims |

### Ablation Study (Accuracy by Dimension Count)

| K dims | Top-K (Fisher) | Top-K (LR) | Random-K | Bottom-K |
|---:|---:|---:|---:|---:|
| 10 | 0.8594 | 0.7192 | 0.7286 | 0.5000 |
| 50 | 0.8767 | 0.8541 | 0.8158 | 0.5211 |
| 100 | 0.8882 | 0.8845 | 0.8553 | 0.5651 |
| 200 | 0.9037 | 0.9066 | 0.8880 | 0.6920 |
| 512 | 0.9195 | 0.9241 | 0.9207 | 0.9003 |
| 1024 | 0.9311 | 0.9314 | 0.9315 | 0.9311 |

## Next Steps

- [ ] Train production classifier (Logistic Regression / MLP)
- [ ] Evaluate on held-out test set
- [ ] Test edge cases and hard examples
- [ ] Package for inference
