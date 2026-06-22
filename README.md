# 🏪 UPI Merchant Classification

> **Binary classifier to distinguish Person Names from Merchant/Business Names in UPI transaction strings.**

Built for [CreditMitra](https://creditmitra.in) × TIH-IITB — classifies UPI counterparty names to understand spending patterns and assess creditworthiness.

```
"Swiggy Instamart"  →  merchant (0.98)
"Rajesh Kumar"      →  person   (0.92)
"HDFC Bank"         →  merchant (0.95)
"Priya Sharma"      →  person   (0.95)
```

---

## ⚡ Quick Start

### Option 1: PyTorch (GPU, fastest)

```python
from classifier import MerchantClassifier

clf = MerchantClassifier()
result = clf.classify("Swiggy Instamart")
# → {"label": "merchant", "confidence": 0.9774, "p_merchant": 0.9774}

results = clf.classify_batch(["Rajesh Kumar", "HDFC Bank", "Priya Sharma"])
```

### Option 2: ONNX (CPU, no PyTorch needed)

```python
from onnx_classifier import ONNXMerchantClassifier

clf = ONNXMerchantClassifier()
clf.classify("Swiggy Instamart")
# → {"label": "merchant", "confidence": 0.9797}
```

### Option 3: CLI

```bash
# Single or multiple names
python classify.py "Swiggy" "Rajesh Kumar" "HDFC Bank"

# JSON output
python classify.py --json "Swiggy"

# CSV batch processing
python classify.py --input names.csv --output results.csv

# Interactive mode
python classify.py --interactive
```

---

## 📊 Performance

| Metric | Value |
|:---|---:|
| **Test Accuracy** | **98.36%** |
| F1 Score | 0.9848 |
| AUC-ROC | 0.9974 |
| Unseen Data Accuracy | 94.1% |
| Inference (PyTorch GPU) | ~52ms/name |
| Inference (ONNX CPU) | ~153ms/name |
| Classifier ONNX size | 0.01 MB |

---

## 🏗️ Architecture

```
Input: "Swiggy Instamart"
         │
         ▼
┌───────────────────────┐
│  Qwen3-Embedding-0.6B │  595M params → 1024-dim embedding
│  (last-token pooling)  │  L2 normalized
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│  StandardScaler        │  Feature normalization
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│  AttentionMLP          │  661K params
│  ├─ Reshape 1024→64×16│
│  ├─ 2× Self-Attention  │  8 heads each
│  ├─ LayerNorm+Residual │
│  └─ MLP: 512→256→1    │  BatchNorm + GELU
└───────────┬───────────┘
            │ sigmoid
            ▼
Output: merchant (0.9774)
```

---

## 🛠️ Installation

### Full Setup (training + inference)

```bash
git clone https://github.com/CreditMitra-TIH-IITB/CreditMitra-Merchant-Classification.git
cd CreditMitra-Merchant-Classification

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

pip install -r requirements.txt
```

### Lightweight Setup (ONNX inference only)

```bash
pip install onnxruntime transformers numpy joblib
```

---

## 📁 Project Structure

```
├── classifier.py              # PyTorch inference module
├── classify.py                # CLI tool (single/batch/interactive/JSON)
├── onnx_classifier.py         # ONNX inference (no PyTorch needed)
├── requirements.txt
│
├── data/
│   ├── generate_indian_names.py     # Person name generation
│   ├── generate_merchant_names.py   # Merchant name generation
│   ├── augment_names.py             # Data augmentation
│   ├── generate_embeddings.py       # Qwen3 embedding generation
│   ├── dimension_importance.py      # Feature analysis
│   ├── train_models.py              # sklearn model training
│   ├── train_pytorch_model.py       # PyTorch model tournament
│   ├── export_and_test.py           # ONNX export + edge case testing
│   └── export_qwen3_onnx.py        # Qwen3 ONNX export
│
├── models/                          # (gitignored — regenerate via scripts)
│   ├── pytorch_attentionmlp.pt      # Best PyTorch model
│   ├── sklearn_mlp_large.joblib     # sklearn baseline
│   ├── scaler.joblib                # StandardScaler
│   ├── attentionmlp.onnx            # Classifier ONNX (0.01 MB)
│   └── onnx_pipeline/
│       ├── qwen3_embedding.onnx     # Full embedding model ONNX
│       └── tokenizer/               # Qwen3 tokenizer files
│
├── embeddings/                      # (gitignored — regenerate)
│   └── qwen3_embeddings.npz        # Pre-computed embeddings
│
└── research/                        # 9 research documents + 19 figures
    ├── 01_problem_definition.md
    ├── 02_dataset_generation.md
    ├── 03_embedding_eda_findings.md
    ├── 04_dimension_importance.md
    ├── 05_model_selection.md
    ├── 06_final_model_training.md
    ├── 07_pytorch_deep_learning.md
    ├── 08_onnx_edge_cases_validation.md
    ├── 09_final_summary.md
    └── figures/
```

---

## 🔬 Research Pipeline

The full pipeline can be reproduced from scratch:

```bash
# 1. Generate dataset (233K names)
python data/generate_indian_names.py
python data/generate_merchant_names.py
python data/augment_names.py

# 2. Generate embeddings (needs GPU, ~30 min)
python data/generate_embeddings.py

# 3. Train models
python data/train_models.py              # sklearn (18 experiments)
python data/train_pytorch_model.py       # PyTorch (4-model tournament)

# 4. Export to ONNX
python data/export_and_test.py           # Classifier ONNX + edge cases
python data/export_qwen3_onnx.py         # Full embedding model ONNX

# 5. Run inference
python classify.py "Swiggy Instamart" "Rajesh Kumar"
```

---

## 📈 Model Comparison

| Model | Accuracy | F1 | AUC |
|:---|---:|---:|---:|
| Logistic Regression | 96.59% | 0.968 | 0.994 |
| Random Forest | 95.21% | 0.957 | 0.990 |
| XGBoost | 96.70% | 0.970 | 0.994 |
| SVM (RBF) | 97.16% | 0.974 | 0.996 |
| sklearn MLP (512-256-128) | 97.54% | 0.978 | 0.997 |
| PyTorch DeepMLP | 97.38% | 0.976 | 0.996 |
| PyTorch ResidualMLP | 97.70% | 0.979 | 0.997 |
| PyTorch WideMLP | 97.93% | 0.981 | 0.997 |
| **PyTorch AttentionMLP** | **98.36%** | **0.985** | **0.997** |

---

## ⚠️ Known Limitations

- **Person-derived brand names** (Tanishq, Haldiram, Lakme) may be misclassified — genuine linguistic ambiguity
- **Single-word brands** without business suffixes ("Bata", "Allen") tend toward person classification
- **Extreme case randomization** ("sWiGgY") breaks the model; normal case variations work fine
- **Synthetic training data** — not validated on actual UPI transaction narrations yet

---

## 📄 License

This project is developed as part of the CreditMitra × TIH-IITB collaboration.

---

## 👥 Team

- **CreditMitra** — Fintech credit assessment platform
- **TIH-IITB** — Technology Innovation Hub, IIT Bombay
