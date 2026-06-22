"""
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
