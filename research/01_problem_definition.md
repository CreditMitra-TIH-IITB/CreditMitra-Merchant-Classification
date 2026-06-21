# Stage 1: Problem Definition & Approach Selection

**Date:** 2026-06-22  
**Status:** ✅ Complete

---

## Problem Statement

In UPI (Unified Payments Interface) transactions, the narration string contains a **payee name** extracted from the transaction. This payee name can be either:

- A **Person Name** (e.g., "Ramesh Kumar Sharma", "Neha P", "AMIT GUPTA")
- A **Merchant/Business Name** (e.g., "Swiggy", "Sharma Electronics", "Raju Ki Dukan")

We need a system that can **classify** the extracted payee name into one of these two categories (PERSON vs MERCHANT) with high accuracy, even when:

- Names are corrupted, truncated, or noisy (real-world UPI narrations are messy)
- Business names contain person names as prefixes (e.g., "Sharma Traders")
- The classifier must handle India's extreme linguistic diversity (17+ regional languages)

## Why This Is Hard

1. **Ambiguity**: "Sharma" alone is a person, but "Sharma Electronics" is a merchant. "Zara" is both a name and a brand.
2. **Noise**: UPI narrations have truncation (`RAMESH KUM...`), typos (`Flipkrat`), case noise (`rAmEsH`), and prefixes (`UPI-Dominos`).
3. **Diversity**: Indian names span Hindi, Tamil, Bengali, Malayalam, Punjabi, Kannada, Telugu, Odia, and more.
4. **Scale**: Must be fast enough for production (real-time transaction processing).

## Approaches Considered

### 1. Rule-Based / Keyword Matching
- **Pros**: Fast, interpretable
- **Cons**: Brittle, can't handle noise, requires constant maintenance of keyword lists
- **Verdict**: ❌ Not scalable

### 2. TF-IDF + Classical ML
- **Pros**: Simple, fast inference
- **Cons**: No semantic understanding, character-level features only
- **Verdict**: ⚠️ Baseline only

### 3. Fine-tuned Transformer (e.g., BERT)
- **Pros**: Best accuracy potential
- **Cons**: Expensive inference, overkill for a binary classification
- **Verdict**: ⚠️ Too heavy for production

### 4. Embedding + Logistic Regression (Hybrid) ✅ SELECTED
- **Pros**: Semantic embeddings capture meaning, LR is fast at inference, best accuracy/speed tradeoff
- **Cons**: Requires good embeddings model
- **Verdict**: ✅ Best fit

## Selected Architecture

```
Input Name String
    |
    v
[Qwen/Qwen3-Embedding-0.6B]  -->  768-dim embedding vector
    |
    v
[Logistic Regression / Light MLP]  -->  PERSON | MERCHANT
    |
    v
Output: Classification + Confidence Score
```

### Why Qwen3-Embedding-0.6B?

- **Small & fast**: 0.6B parameters, suitable for production
- **Multilingual**: Trained on 30+ languages including Hindi, Bengali, Tamil
- **High quality**: Strong performance on MTEB benchmarks
- **Local**: Can run entirely on-premise (no API costs)

## Next Steps

→ Proceed to **Stage 2: Dataset Generation**
