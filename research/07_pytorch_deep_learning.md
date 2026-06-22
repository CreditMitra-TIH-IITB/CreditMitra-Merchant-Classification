# Stage 7: PyTorch Deep Learning — Pushing Past 97.5%

**Date:** 2026-06-22
**Status:** Complete
**Baseline:** sklearn MLP 97.54% → **PyTorch AttentionMLP 98.36%** (+0.82%)

---

## Motivation

The sklearn MLP (512-256-128) achieved 97.54% — already strong. But sklearn's `MLPClassifier` is limited: no BatchNorm, no Dropout control, no residual connections, no attention, no GPU, no advanced LR schedules. We wanted to see how far PyTorch could push the same embedding features with modern deep learning techniques.

---

## Techniques Used (Beyond sklearn)

| Technique | Why It Helps |
|:---|:---|
| **BatchNorm** | Stabilizes training, acts as regularizer, allows higher LR |
| **Dropout (0.3-0.4)** | Better regularization than sklearn's early stopping alone |
| **Residual connections** | Gradient flow in deeper networks, prevents degradation |
| **GELU activation** | Smoother than ReLU, better gradient flow near zero |
| **Self-attention** | Captures inter-dimension relationships in embeddings |
| **Mixup augmentation** | Creates interpolated training samples, reduces overfitting |
| **Label smoothing (0.05)** | Prevents overconfident predictions, improves calibration |
| **OneCycleLR** | Superconvergence: warmup → high LR → cosine decay |
| **AdamW** | Weight decay decoupled from gradient updates |
| **Gradient clipping** | Prevents training instability |

---

## Architecture Tournament

### 1. AttentionMLP — 🏆 Winner (98.36%)

The surprise winner. Treats the 1024-dim embedding as **64 tokens of 16 dims each**, then applies 2 layers of multi-head self-attention (8 heads) before a classification MLP. This is conceptually powerful: it lets the model learn which *groups* of embedding dimensions interact, rather than treating all 1024 dimensions independently as a flat vector.

```
Input (1024) → Reshape(64 × 16) → +PosEmbed → SelfAttn(8 heads) → LayerNorm
→ SelfAttn(8 heads) → LayerNorm → Flatten → MLP(512→256→1)
```

- **Parameters:** 661,185 (smallest model!)
- **Val Acc:** 98.33% (best at epoch 70)
- **Test Acc:** 98.36%
- **Key insight:** Attention with the fewest parameters outperforms everything. The embedding dimensions have meaningful inter-relationships that flat MLPs miss.

### 2. DeepResidualMLP — 98.25%

Deep feedforward with residual blocks at each hidden layer. Three stages (768→512→256), each with 2 residual blocks.

- **Parameters:** 5,004,033
- **Val Acc:** 98.28% (best at epoch 60)
- **Test Acc:** 98.25%
- **Key insight:** Residual connections help deep networks significantly — without them, deeper isn't better.

### 3. WideResNet — 98.18%

Widest architecture: stays at 1024 dims for 3 residual blocks before downsampling. 9.2M parameters.

- **Parameters:** 9,206,785 (largest model)
- **Val Acc:** 98.26% (best at epoch 101)
- **Test Acc:** 98.18%
- **Key insight:** More parameters ≠ better. WideResNet has 14× more parameters than AttentionMLP but worse accuracy. Width helps less than attention.

### 4. SimpleDeeperMLP — 98.08%

A 6-layer MLP (768→512→384→256→128→1) with BatchNorm and Dropout. The simplest PyTorch architecture — essentially sklearn's MLP but with BN, Dropout, and GELU.

- **Parameters:** 1,513,601
- **Val Acc:** 98.06% (best at epoch 35)
- **Test Acc:** 98.08%
- **Key insight:** Just adding BatchNorm + Dropout + GELU to sklearn's architecture gains +0.54%. These are the "free" improvements from modern deep learning hygiene.

---

## Ensemble Results

| Ensemble | Test Acc | F1 | AUC |
|:---|---:|---:|---:|
| Top-2 (AttentionMLP + DeepResMLP) | **98.37%** | 0.985 | 0.998 |
| All 4 models | 98.34% | 0.985 | 0.998 |

The ensemble barely improves over AttentionMLP alone (+0.01%). This suggests the models are making similar errors — the remaining 1.6% errors are genuinely ambiguous cases.

---

## What Made the Difference?

### 1. Attention > Flat MLP (+0.82%)
Self-attention lets the model learn that certain embedding dimensions are *related* and should be processed together. A flat MLP treats all 1024 dimensions independently at each layer. The Qwen3 embedding model organizes information in structured groups — attention discovers this structure automatically.

### 2. BatchNorm + Dropout (+0.54% just from "hygiene")
SimpleDeeperMLP proves that simply adding BatchNorm, Dropout, and GELU to the same architecture gives +0.54% for free. sklearn's MLP lacks these fundamental regularization tools.

### 3. More Parameters Don't Help
WideResNet (9.2M params) < AttentionMLP (661K params). The bottleneck is not model capacity — it's the ability to model inter-feature relationships. Attention does this efficiently with far fewer parameters.

### 4. OneCycleLR is Crucial
The warmup phase (10% of training) allows the model to find a good region of the loss landscape before the high LR exploration phase. Cosine annealing then gradually refines. This was key for all models converging well.

---

## Comparison with sklearn

| Metric | sklearn MLP | PyTorch Attention | Improvement |
|:---|---:|---:|---:|
| Test Accuracy | 97.54% | **98.36%** | **+0.82%** |
| Test F1 | 0.977 | **0.985** | +0.008 |
| Test AUC | 0.997 | **0.997** | same |
| Parameters | ~1.4M | **661K** | 2× smaller |
| FP (Person→Merchant) | 392 | **238** | **39% fewer** |
| FN (Merchant→Person) | 470 | **337** | **28% fewer** |
| Total Errors | 862 | **575** | **33% fewer** |

The PyTorch model eliminated **287 errors** (33% reduction). False positives dropped by 39% — significantly fewer person names being misclassified as merchants.

---

## Figures

| # | Figure | Description |
|:---|:---|:---|
| 23 | `23_pytorch_tournament.png` | Training curves, validation accuracy, model comparison, LR schedule |

---

## Exported Models

| File | Size | Architecture |
|:---|---:|:---|
| `pytorch_attentionmlp.pt` | ~2.5 MB | 🏆 Best single model |
| `pytorch_deepresmlp.pt` | 19.2 MB | 2nd best |
| `pytorch_wideresnet.pt` | 35.2 MB | Widest |
| `pytorch_simpledeepermlp.pt` | 5.8 MB | Simplest upgrade |
| `pytorch_results.json` | ~2 KB | All metrics |
