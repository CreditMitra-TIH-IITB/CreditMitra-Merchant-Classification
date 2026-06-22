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

## Figures

All research figures are in `figures/` (18 plots covering EDA, dimension analysis, model evaluation).

## Project Goal

Build a binary classifier that distinguishes **Person Names** from **Merchant/Business Names** in UPI transaction narration strings, using Qwen3 embeddings + MLP classifier.

## Best Result

**MLP (512-256-128) on StandardScaler-transformed Qwen3 embeddings: 97.59% test accuracy** (F1=0.978, AUC=0.997) on 233,863 samples.
