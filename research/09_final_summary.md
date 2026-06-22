# Stage 9: Final Summary — Project Complete

**Date:** 2026-06-22
**Status:** Complete ✓

---

## Project Overview

**Goal:** Build a binary classifier to distinguish **Person Names** from **Merchant/Business Names** in UPI transaction narration strings for CreditMitra's credit assessment pipeline.

**Why it matters:** In UPI transactions, the counterparty name field contains either a person (P2P transfer — e.g., "Rajesh Kumar") or a merchant (P2M payment — e.g., "Swiggy Instamart"). Correctly classifying these enables CreditMitra to understand spending patterns, identify income sources, and assess creditworthiness from transaction data alone.

---

## Journey: From Problem to Production

### Stage 1: Problem Definition
- Defined the task as binary classification: person (0) vs merchant (1)
- Chose embedding-based approach over rule-based (handles ambiguity better)
- Selected Qwen3-Embedding-0.6B for its strong multilingual + semantic capabilities

### Stage 2: Dataset Generation (233,478 names)
- Generated 114,279 person names (Indian: Hindu, Muslim, Sikh, South Indian, regional)
- Generated 119,199 merchant names (real Indian brands + synthetic business names)
- Applied augmentation: case variation, abbreviation, prefix/suffix patterns
- 80/20 stratified train/test split

### Stage 3: Embedding EDA (16 visualizations)
- UMAP/t-SNE show clear person vs merchant clusters with some overlap at boundaries
- Cosine similarity distributions confirm distinct embedding spaces
- PCA: first 50 components capture 70% variance, meaningful structure in embedding space
- Fisher discriminant ratio identifies dimensions most useful for classification

### Stage 4: Dimension Importance Analysis
- Fisher scores, t-tests, logistic regression coefficients — all agree on important dims
- Top 200 of 1024 dimensions carry most discriminative power
- Ablation study: can achieve 95%+ accuracy with just top-300 dimensions
- Full 1024 dimensions still optimal for maximum performance

### Stage 5: Model Selection (18 experiments, 8 classifiers)
- **Best sklearn model:** MLP_large (512-256-128) → 97.59% accuracy
- Random Forest, XGBoost, SVM, Logistic Regression all compared
- Stacking ensemble provided no improvement over single MLP
- StandardScaler critical for MLP/SVM performance

### Stage 6: Final sklearn Model
- Trained on full training set: **97.54% test accuracy**
- Exported: `sklearn_mlp_large.joblib` + `scaler.joblib`
- Error analysis: most errors are genuinely ambiguous names

### Stage 7: PyTorch Deep Learning (4-model tournament on GPU)
- **AttentionMLP:** 98.36% ← WINNER 🏆
- ResidualMLP: 97.70%
- WideMLP: 97.93%
- DeepMLP: 97.38%
- AttentionMLP treats 1024-dim embedding as 64 tokens × 16 dims, applies 2-layer multi-head self-attention (8 heads), then classifies
- Training techniques: OneCycleLR, label smoothing, mixup augmentation, BatchNorm

### Stage 8: ONNX Export + Validation
- AttentionMLP classifier → ONNX: **0.01 MB** (205x smaller than PyTorch), **9.5x faster** single-sample on CPU
- Edge case testing: 100% on clear names, 50% on genuinely ambiguous brands (Tanishq, Lakme)
- Unseen data: **94.1%** (97.1% persons, 91.2% merchants) — errors are person-derived brand names
- Qwen3 embedding model → ONNX: full PyTorch-free pipeline

---

## Final Architecture

```
Input: "Swiggy Instamart"
  │
  ▼
┌─────────────────────────────┐
│  Qwen3-Embedding-0.6B       │  595M params, 1024-dim output
│  (sentence-transformers)     │  Last-token pooling + L2 norm
└─────────────┬───────────────┘
              │ 1024-dim embedding
              ▼
┌─────────────────────────────┐
│  StandardScaler              │  Fitted on training embeddings
└─────────────┬───────────────┘
              │ scaled embedding
              ▼
┌─────────────────────────────┐
│  AttentionMLP (661K params)  │
│  ┌────────────────────────┐ │
│  │ Reshape: 1024 → 64×16  │ │
│  │ + Positional Embedding  │ │
│  ├────────────────────────┤ │
│  │ Self-Attention (8 heads)│ │
│  │ + LayerNorm + Residual  │ │
│  ├────────────────────────┤ │
│  │ Self-Attention (8 heads)│ │
│  │ + LayerNorm + Residual  │ │
│  ├────────────────────────┤ │
│  │ Flatten → 512 → 256 → 1│ │
│  │ BatchNorm + GELU + Drop │ │
│  └────────────────────────┘ │
└─────────────┬───────────────┘
              │ sigmoid
              ▼
Output: merchant (0.9774)
```

---

## Performance Summary

| Metric | Value |
|:---|:---|
| **Test Accuracy** | 98.36% |
| **F1 Score** | 0.9848 |
| **AUC-ROC** | 0.9974 |
| **Unseen Data Accuracy** | 94.1% |
| **Edge Case Accuracy** | 85.4% (includes deliberately ambiguous cases) |
| **Inference Latency (PyTorch GPU)** | ~50ms/name (embedding) + 1.5ms (classifier) |
| **Inference Latency (ONNX CPU)** | ~153ms/name (embedding) + 0.16ms (classifier) |
| **Model Size (classifier only)** | 0.01 MB (ONNX) |
| **Total Pipeline Size** | ~2.3 GB (dominated by embedding model) |

---

## Deployment Options

### Option A: PyTorch + GPU (fastest)
```python
from classifier import MerchantClassifier
clf = MerchantClassifier()  # Auto-detects GPU
clf.classify("Swiggy")      # 50ms latency
```

### Option B: ONNX + CPU (portable, no PyTorch)
```python
from onnx_classifier import ONNXMerchantClassifier
clf = ONNXMerchantClassifier()  # CPU only, no GPU needed
clf.classify("Swiggy")          # 153ms latency
```

### Option C: CLI Tool
```bash
python classify.py "Swiggy" "Rajesh Kumar" "HDFC Bank"
python classify.py --json "Swiggy"
python classify.py --input names.csv --output results.csv
python classify.py --interactive
```

---

## Known Limitations

1. **Person-derived brand names** (Haldiram, Tanishq, Lakme) are sometimes misclassified as persons — these are genuine linguistic ambiguities where even humans need context.

2. **Single-word brand names** without business suffixes (e.g., "Bata", "Allen") tend to be classified as persons. Adding "Store", "Pvt Ltd", etc. fixes this.

3. **Extreme case randomization** (e.g., "sWiGgY") breaks the model. Normal variations (lowercase, UPPERCASE) work fine.

4. **Training data is synthetic** — real-world UPI narrations may contain additional noise (truncation, special characters, UPI reference codes). The model handles UPI prefixes (e.g., "UPI-Swiggy") gracefully but hasn't been validated on actual transaction data.

---

## Repository Structure

```
CreditMitra-Merchant-Classification/
├── classifier.py              # PyTorch inference (MerchantClassifier)
├── classify.py                # CLI tool
├── onnx_classifier.py         # ONNX inference (no PyTorch)
├── data/
│   ├── generate_indian_names.py
│   ├── generate_merchant_names.py
│   ├── augment_names.py
│   ├── generate_embeddings.py
│   ├── dimension_importance.py
│   ├── train_models.py
│   ├── train_pytorch_model.py
│   ├── export_and_test.py     # ONNX classifier export + edge cases
│   └── export_qwen3_onnx.py   # Qwen3 ONNX export
├── models/
│   ├── sklearn_mlp_large.joblib
│   ├── scaler.joblib
│   ├── pytorch_attentionmlp.pt
│   ├── attentionmlp.onnx
│   ├── model_config.json
│   └── onnx_pipeline/
│       ├── qwen3_embedding.onnx
│       ├── qwen3_embedding.onnx.data
│       ├── tokenizer/
│       └── requirements_onnx.txt
└── research/
    ├── README.md
    ├── 01_problem_definition.md
    ├── 02_dataset_generation.md
    ├── 03_embedding_eda_findings.md
    ├── 04_dimension_importance.md
    ├── 05_model_selection.md
    ├── 06_final_model_training.md
    ├── 07_pytorch_deep_learning.md
    ├── 08_onnx_edge_cases_validation.md
    ├── 09_final_summary.md          ← This document
    └── figures/ (19 plots)
```

---

## What Would Push Accuracy Higher

If further improvement is needed, the primary bottleneck is the **embedding model**, not the classifier. Directions:

1. **Fine-tune Qwen3** on Indian person/merchant names (could push to 99%+)
2. **Real transaction data** — train on actual UPI narrations instead of synthetic names
3. **Multi-class** — classify merchant *type* (food, finance, retail, etc.)
4. **Ensemble** — combine Qwen3 embeddings with rule-based features (presence of "Pvt Ltd", "Store", etc.)

---

*Project completed 2026-06-22 by CreditMitra × TIH-IITB.*
