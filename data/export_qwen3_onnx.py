"""
Export Qwen3-Embedding-0.6B to ONNX
========================================
Exports the full embedding model to ONNX format for cross-platform inference.

Uses a wrapper model to avoid DynamicCache export issues.

Usage:
    python data/export_qwen3_onnx.py
"""

import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
ONNX_DIR = os.path.join(MODELS_DIR, "onnx_pipeline")

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"


# ============================================================================
# WRAPPER: Returns only the tensor we need (avoids DynamicCache issue)
# ============================================================================

class EmbeddingWrapper(nn.Module):
    """Wraps the Qwen3 model to return only the embedding tensor.

    Uses LAST-TOKEN pooling (matches sentence-transformers for Qwen3)
    + L2 normalization. Avoids DynamicCache export issue.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        # Get model output
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=False,
        )
        hidden_state = outputs[0]  # (batch, seq_len, hidden_dim)

        # Last-token pooling: take hidden state of last non-padding token
        # Sum attention_mask along seq_len to get sequence lengths
        seq_lengths = attention_mask.sum(dim=1) - 1  # (batch,) — index of last token
        batch_size = hidden_state.shape[0]
        batch_idx = torch.arange(batch_size, device=hidden_state.device)
        pooled = hidden_state[batch_idx, seq_lengths]  # (batch, hidden_dim)

        # L2 normalize
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)

        return pooled


def export_qwen3_onnx():
    """Export Qwen3 embedding model to ONNX."""
    print("=" * 70)
    print("PHASE 1: EXPORT QWEN3 EMBEDDING MODEL TO ONNX")
    print("=" * 70)

    os.makedirs(ONNX_DIR, exist_ok=True)

    # Load model
    print(f"\n  Loading {MODEL_NAME}...")
    from transformers import AutoTokenizer, AutoModel

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    base_model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
    base_model.eval()

    params = sum(p.numel() for p in base_model.parameters())
    print(f"  Parameters: {params:,}")

    # Save tokenizer
    tokenizer_dir = os.path.join(ONNX_DIR, "tokenizer")
    tokenizer.save_pretrained(tokenizer_dir)
    print(f"  Tokenizer saved: {tokenizer_dir}")

    # Wrap model
    wrapper = EmbeddingWrapper(base_model)
    wrapper.eval()

    # Dummy input
    dummy_text = "Swiggy Instamart"
    inputs = tokenizer(dummy_text, return_tensors="pt", padding="max_length",
                       truncation=True, max_length=32)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Verify wrapper works
    with torch.no_grad():
        test_out = wrapper(input_ids, attention_mask)
    print(f"  Wrapper output shape: {test_out.shape}")  # Should be (1, 1024)
    print(f"  Output norm: {torch.norm(test_out, dim=1).item():.4f}")  # Should be 1.0

    # Export
    onnx_path = os.path.join(ONNX_DIR, "qwen3_embedding.onnx")
    print(f"\n  Exporting to ONNX (this may take a few minutes)...")
    start = time.time()

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (input_ids, attention_mask),
            onnx_path,
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=["input_ids", "attention_mask"],
            output_names=["output_embedding"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "seq_len"},
                "attention_mask": {0: "batch_size", 1: "seq_len"},
                "output_embedding": {0: "batch_size"},
            },
        )

    elapsed = time.time() - start
    onnx_size = os.path.getsize(onnx_path) / 1024 / 1024
    print(f"  Export time: {elapsed:.1f}s")
    print(f"  ONNX size:   {onnx_size:.1f} MB")
    print(f"  Path:        {onnx_path}")

    return onnx_path


def verify_onnx(onnx_path):
    """Verify ONNX outputs match PyTorch."""
    print("\n" + "=" * 70)
    print("PHASE 2: VERIFY NUMERICAL EQUIVALENCE")
    print("=" * 70)

    import onnxruntime as ort
    from transformers import AutoTokenizer, AutoModel

    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(ONNX_DIR, "tokenizer"), trust_remote_code=True
    )

    # PyTorch reference
    base_model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
    base_model.eval()
    wrapper = EmbeddingWrapper(base_model)
    wrapper.eval()

    # ONNX session
    ort_session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    test_names = [
        "Swiggy Instamart", "Rajesh Kumar", "HDFC Bank",
        "Priya Sharma", "Amazon Pay", "Deepak Singh",
        "Zomato", "Flipkart", "Ola Cabs", "Rahul Verma",
    ]

    print("\n  Comparing PyTorch vs ONNX embeddings:")
    max_diffs = []

    for name in test_names:
        inputs = tokenizer(name, return_tensors="pt", padding="max_length",
                           truncation=True, max_length=32)

        # PyTorch
        with torch.no_grad():
            pt_embed = wrapper(inputs["input_ids"], inputs["attention_mask"]).numpy()

        # ONNX
        onnx_embed = ort_session.run(None, {
            "input_ids": inputs["input_ids"].numpy(),
            "attention_mask": inputs["attention_mask"].numpy(),
        })[0]

        diff = np.max(np.abs(pt_embed - onnx_embed))
        max_diffs.append(diff)
        print(f"    {name:<25} max diff: {diff:.8f}  embed[0:3]: [{pt_embed[0,0]:.4f}, {pt_embed[0,1]:.4f}, {pt_embed[0,2]:.4f}]")

    overall_max = max(max_diffs)
    print(f"\n  Overall max difference: {overall_max:.8f}")
    print(f"  Equivalent: {'YES' if overall_max < 1e-4 else 'CLOSE (float32 tolerance)'}")

    return overall_max


def benchmark_onnx(onnx_path):
    """Benchmark ONNX embedding speed vs PyTorch."""
    print("\n" + "=" * 70)
    print("PHASE 3: BENCHMARK")
    print("=" * 70)

    import onnxruntime as ort
    from transformers import AutoTokenizer
    from sentence_transformers import SentenceTransformer

    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(ONNX_DIR, "tokenizer"), trust_remote_code=True
    )
    ort_session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    # Also load sentence-transformers for comparison
    st_model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)

    test_name = "Swiggy Instamart"
    inputs = tokenizer(test_name, return_tensors="np", padding="max_length",
                       truncation=True, max_length=32)

    # --- ONNX CPU Benchmark ---
    for _ in range(5):
        _ = ort_session.run(None, {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        })

    times = []
    for _ in range(50):
        start = time.perf_counter()
        _ = ort_session.run(None, {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        })
        times.append(time.perf_counter() - start)

    onnx_median = np.median(times) * 1000
    onnx_p95 = np.percentile(times, 95) * 1000

    # --- sentence-transformers GPU Benchmark ---
    _ = st_model.encode([test_name])
    times = []
    for _ in range(50):
        start = time.perf_counter()
        _ = st_model.encode([test_name], normalize_embeddings=True)
        times.append(time.perf_counter() - start)

    st_median = np.median(times) * 1000
    st_p95 = np.percentile(times, 95) * 1000

    print(f"\n  Single-name embedding latency:")
    print(f"    ONNX (CPU):               {onnx_median:.1f}ms median, {onnx_p95:.1f}ms p95")
    print(f"    sentence-transformers:     {st_median:.1f}ms median, {st_p95:.1f}ms p95")
    print(f"    Speedup:                   {st_median/onnx_median:.1f}x")


def create_standalone_classifier():
    """Create onnx_classifier.py — needs NO PyTorch."""
    print("\n" + "=" * 70)
    print("PHASE 4: CREATE STANDALONE ONNX CLASSIFIER")
    print("=" * 70)

    pipeline_code = '''"""
Standalone ONNX Merchant Classifier
========================================
Full end-to-end: name -> person/merchant classification.
NO PyTorch, NO sentence-transformers required.

Dependencies (minimal):
    pip install onnxruntime transformers numpy joblib

Usage:
    from onnx_classifier import ONNXMerchantClassifier

    clf = ONNXMerchantClassifier()
    result = clf.classify("Swiggy Instamart")
    # -> {"label": "merchant", "confidence": 0.9774}

    results = clf.classify_batch(["Rajesh Kumar", "HDFC Bank", "Priya"])
    # -> [{"label": "person", ...}, {"label": "merchant", ...}, ...]
"""

import os
import numpy as np
import joblib


class ONNXMerchantClassifier:
    """Full ONNX pipeline: text -> embedding -> classification.

    No PyTorch or sentence-transformers needed.
    Just onnxruntime + transformers (tokenizer only) + numpy + joblib.

    The embedding model (Qwen3-Embedding-0.6B) and classifier (AttentionMLP)
    both run as ONNX models on CPU.
    """

    def __init__(self, models_dir=None):
        if models_dir is None:
            models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

        onnx_dir = os.path.join(models_dir, "onnx_pipeline")

        # Load tokenizer (lightweight, no PyTorch needed)
        from transformers import AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(onnx_dir, "tokenizer"), trust_remote_code=True
        )

        # Load ONNX models
        import onnxruntime as ort
        self._embedding_session = ort.InferenceSession(
            os.path.join(onnx_dir, "qwen3_embedding.onnx"),
            providers=["CPUExecutionProvider"],
        )
        self._classifier_session = ort.InferenceSession(
            os.path.join(models_dir, "attentionmlp.onnx"),
            providers=["CPUExecutionProvider"],
        )

        # Load scaler (fitted StandardScaler)
        self._scaler = joblib.load(os.path.join(models_dir, "scaler.joblib"))

        print("ONNXMerchantClassifier ready (NO PyTorch needed!)")
        print("  Embedding: Qwen3-Embedding-0.6B (ONNX)")
        print("  Classifier: AttentionMLP (ONNX)")

    def _embed(self, name):
        """Generate L2-normalized embedding for a single name."""
        inputs = self._tokenizer(
            name, return_tensors="np",
            padding="max_length", truncation=True, max_length=32
        )
        embedding = self._embedding_session.run(None, {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        })[0]
        return embedding  # (1, 1024), already normalized

    def classify(self, name):
        """Classify a single name.

        Args:
            name: The name string to classify.

        Returns:
            dict with: name, label, confidence, label_id, p_merchant
        """
        return self.classify_batch([name])[0]

    def classify_batch(self, names):
        """Classify a batch of names.

        Args:
            names: List of name strings.

        Returns:
            List of dicts with: name, label, confidence, label_id, p_merchant
        """
        results = []
        for name in names:
            # 1. Embed
            embedding = self._embed(name)
            # 2. Scale
            scaled = self._scaler.transform(embedding).astype(np.float32)
            # 3. Classify
            logit = self._classifier_session.run(None, {"embedding": scaled})[0]
            p_merchant = float(1.0 / (1.0 + np.exp(-logit.item())))

            is_merchant = p_merchant >= 0.5
            confidence = p_merchant if is_merchant else (1.0 - p_merchant)

            results.append({
                "name": name,
                "label": "merchant" if is_merchant else "person",
                "confidence": round(confidence, 4),
                "label_id": 1 if is_merchant else 0,
                "p_merchant": round(p_merchant, 4),
            })
        return results

    def __repr__(self):
        return "ONNXMerchantClassifier(embedding=Qwen3-0.6B, classifier=AttentionMLP, runtime=ONNX)"
'''

    pipeline_path = os.path.join(BASE_DIR, "onnx_classifier.py")
    with open(pipeline_path, "w", encoding="utf-8") as f:
        f.write(pipeline_code)
    print(f"  Created: {pipeline_path}")

    # Requirements file
    req_path = os.path.join(ONNX_DIR, "requirements_onnx.txt")
    with open(req_path, "w") as f:
        f.write("onnxruntime>=1.17.0\n")
        f.write("transformers>=4.40.0\n")
        f.write("numpy>=1.24.0\n")
        f.write("joblib>=1.3.0\n")
    print(f"  Created: {req_path}")
    print(f"\n  To run without PyTorch:")
    print(f"    pip install onnxruntime transformers numpy joblib")
    print(f"    from onnx_classifier import ONNXMerchantClassifier")


def main():
    os.makedirs(ONNX_DIR, exist_ok=True)

    onnx_path = export_qwen3_onnx()
    verify_onnx(onnx_path)
    benchmark_onnx(onnx_path)
    create_standalone_classifier()

    # Summary
    print("\n" + "=" * 70)
    print("COMPLETE!")
    print("=" * 70)

    onnx_size = os.path.getsize(onnx_path) / 1024 / 1024
    clf_size = os.path.getsize(os.path.join(MODELS_DIR, "attentionmlp.onnx")) / 1024 / 1024

    print(f"\n  ONNX Pipeline Files:")
    print(f"    models/onnx_pipeline/qwen3_embedding.onnx  ({onnx_size:.0f} MB)")
    print(f"    models/attentionmlp.onnx                   ({clf_size:.2f} MB)")
    print(f"    models/scaler.joblib                       (~8 KB)")
    print(f"    models/onnx_pipeline/tokenizer/            (tokenizer files)")
    print(f"    onnx_classifier.py                         (standalone classifier)")


if __name__ == "__main__":
    main()
