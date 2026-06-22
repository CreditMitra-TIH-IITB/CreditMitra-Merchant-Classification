"""
Merchant Classifier — Production Inference Module
=====================================================
End-to-end classification: name string → person/merchant prediction.

Usage:
    from classifier import MerchantClassifier

    clf = MerchantClassifier()
    result = clf.classify("Swiggy Instamart")
    # → {"label": "merchant", "confidence": 0.9987}

    results = clf.classify_batch(["Rajesh Kumar", "HDFC Bank", "Priya"])
    # → [{"label": "person", ...}, {"label": "merchant", ...}, ...]
"""

import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import joblib
from typing import Union

# ============================================================================
# MODEL ARCHITECTURE (must match training exactly)
# ============================================================================

class AttentionMLP(nn.Module):
    """Self-attention over embedding dimensions + MLP classifier.
    
    Treats 1024-dim embedding as 64 tokens of 16 dims each,
    applies 2 layers of multi-head self-attention, then classifies.
    """
    def __init__(self, input_dim=1024, num_heads=8, dropout=0.3):
        super().__init__()
        self.n_tokens = 64
        self.token_dim = input_dim // self.n_tokens  # 16

        self.pos_embed = nn.Parameter(torch.randn(1, self.n_tokens, self.token_dim) * 0.02)

        self.attention = nn.MultiheadAttention(
            embed_dim=self.token_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(self.token_dim)

        self.attention2 = nn.MultiheadAttention(
            embed_dim=self.token_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.attn_norm2 = nn.LayerNorm(self.token_dim)

        self.classifier = nn.Sequential(
            nn.Linear(self.token_dim * self.n_tokens, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        B = x.shape[0]
        tokens = x.view(B, self.n_tokens, self.token_dim)
        tokens = tokens + self.pos_embed

        attn_out, _ = self.attention(tokens, tokens, tokens)
        tokens = self.attn_norm(tokens + attn_out)

        attn_out2, _ = self.attention2(tokens, tokens, tokens)
        tokens = self.attn_norm2(tokens + attn_out2)

        flat = tokens.reshape(B, -1)
        return self.classifier(flat).squeeze(-1)


# ============================================================================
# MERCHANT CLASSIFIER
# ============================================================================

class MerchantClassifier:
    """End-to-end merchant/person name classifier.
    
    Loads the Qwen3 embedding model and AttentionMLP classifier.
    Provides single-name and batch classification methods.
    
    Args:
        models_dir: Path to models/ directory containing weights and scaler.
        device: 'cuda', 'cpu', or 'auto' (default). Auto selects GPU if available.
        embedding_model: HuggingFace model name for embeddings.
        embedding_batch_size: Batch size for embedding generation.
        
    Example:
        >>> clf = MerchantClassifier()
        >>> clf.classify("Swiggy Instamart")
        {'name': 'Swiggy Instamart', 'label': 'merchant', 'confidence': 0.9987, 'label_id': 1}
    """

    EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
    EMBEDDING_DIM = 1024

    def __init__(
        self,
        models_dir: str = None,
        device: str = "auto",
        embedding_model: str = None,
        embedding_batch_size: int = 64,
    ):
        self._embedding_batch_size = embedding_batch_size
        self._embedding_model_name = embedding_model or self.EMBEDDING_MODEL

        # Resolve paths
        if models_dir is None:
            models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        self._models_dir = models_dir

        # Device
        if device == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

        # Load components
        self._load_scaler()
        self._load_classifier()
        self._embedding_model = None  # Lazy load (heavy)

        print(f"MerchantClassifier ready:")
        print(f"  Classifier: AttentionMLP (98.36% accuracy)")
        print(f"  Device: {self._device}")
        print(f"  Embedding model: {self._embedding_model_name} (lazy loaded)")

    def _load_scaler(self):
        """Load the fitted StandardScaler."""
        scaler_path = os.path.join(self._models_dir, "scaler.joblib")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")
        self._scaler = joblib.load(scaler_path)

    def _load_classifier(self):
        """Load the trained AttentionMLP model."""
        model_path = os.path.join(self._models_dir, "pytorch_attentionmlp.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        checkpoint = torch.load(model_path, map_location=self._device, weights_only=True)
        self._classifier = AttentionMLP(input_dim=self.EMBEDDING_DIM, num_heads=8, dropout=0.0)
        self._classifier.load_state_dict(checkpoint['model_state_dict'])
        self._classifier.to(self._device)
        self._classifier.eval()

    def _ensure_embedding_model(self):
        """Lazy-load the embedding model (first call only)."""
        if self._embedding_model is not None:
            return

        print(f"  Loading embedding model: {self._embedding_model_name}...")
        start = time.time()

        from sentence_transformers import SentenceTransformer
        self._embedding_model = SentenceTransformer(
            self._embedding_model_name,
            trust_remote_code=True,
        )

        elapsed = time.time() - start
        print(f"  Embedding model loaded in {elapsed:.1f}s (device: {self._embedding_model.device})")

    def _generate_embeddings(self, names: list[str]) -> np.ndarray:
        """Generate normalized embeddings for a list of names."""
        self._ensure_embedding_model()

        embeddings = self._embedding_model.encode(
            names,
            batch_size=self._embedding_batch_size,
            show_progress_bar=len(names) > 100,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return embeddings

    def classify(self, name: str) -> dict:
        """Classify a single name.
        
        Args:
            name: The name string to classify.
            
        Returns:
            dict with keys: name, label, confidence, label_id
        """
        results = self.classify_batch([name])
        return results[0]

    def classify_batch(self, names: list[str]) -> list[dict]:
        """Classify a batch of names.
        
        Args:
            names: List of name strings to classify.
            
        Returns:
            List of dicts, each with: name, label, confidence, label_id
        """
        if not names:
            return []

        # 1. Generate embeddings
        embeddings = self._generate_embeddings(names)

        # 2. Scale
        scaled = self._scaler.transform(embeddings)

        # 3. Classify
        with torch.no_grad():
            X = torch.FloatTensor(scaled).to(self._device)
            logits = self._classifier(X)
            probas = torch.sigmoid(logits).cpu().numpy()

        # 4. Format results
        results = []
        for i, name in enumerate(names):
            p_merchant = float(probas[i])
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

    def classify_from_embeddings(self, embeddings: np.ndarray) -> list[dict]:
        """Classify from pre-computed embeddings (skip embedding generation).
        
        Args:
            embeddings: numpy array of shape (N, 1024).
            
        Returns:
            List of dicts with: label, confidence, label_id, p_merchant
        """
        scaled = self._scaler.transform(embeddings)

        with torch.no_grad():
            X = torch.FloatTensor(scaled).to(self._device)
            logits = self._classifier(X)
            probas = torch.sigmoid(logits).cpu().numpy()

        results = []
        for i in range(len(probas)):
            p_merchant = float(probas[i])
            is_merchant = p_merchant >= 0.5
            confidence = p_merchant if is_merchant else (1.0 - p_merchant)

            results.append({
                "label": "merchant" if is_merchant else "person",
                "confidence": round(confidence, 4),
                "label_id": 1 if is_merchant else 0,
                "p_merchant": round(p_merchant, 4),
            })

        return results

    def __repr__(self):
        return (
            f"MerchantClassifier("
            f"model=AttentionMLP, "
            f"accuracy=98.36%, "
            f"device={self._device}, "
            f"embedding={self._embedding_model_name})"
        )
