# Merchant Classification Research

This directory contains research notes, methodology documentation, and stage-by-stage findings for the UPI Merchant Classification project.

## Stages

| Stage | File | Description |
|:---|:---|:---|
| 1 | `01_problem_definition.md` | Problem statement, goals, and approach selection |
| 2 | `02_dataset_generation.md` | Dataset creation methodology, statistics, and quality analysis |
| 3 | `03_embedding_eda_findings.md` | Full-dataset EDA: UMAP, t-SNE, PCA, cosine similarity, separability |
| 4 | `04_dimension_importance.md` | Dimension importance: Fisher, t-test, LR coefficients, ablation study |
| 5 | `05_model_selection.md` | Model selection: 18 experiments, 8 classifiers, stacking analysis |
| 6 | `06_final_model_training.md` | Final model training, error analysis, and production export |
| 7 | `07_pytorch_deep_learning.md` | PyTorch architectures: Attention, ResNet, Wide — beating 97.5% |
| 8 | `08_onnx_edge_cases_validation.md` | ONNX export (9.5x faster), edge cases, adversarial tests, unseen data |
| 9 | `09_final_summary.md` | **Complete project summary, architecture, deployment, limitations** |

## Figures

All research figures are in `figures/` (19 plots covering EDA, dimension analysis, model evaluation, PyTorch tournament).

## Project Goal

Build a binary classifier that distinguishes **Person Names** from **Merchant/Business Names** in UPI transaction narration strings, using Qwen3 embeddings + MLP classifier.

## Best Result

**PyTorch AttentionMLP (self-attention + MLP) on Qwen3 embeddings: 98.36% test accuracy** (F1=0.985, AUC=0.997) — 33% fewer errors than sklearn MLP baseline.

## Project Status: COMPLETE ✓
