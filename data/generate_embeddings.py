"""
Embedding Generation Pipeline
================================
Generate Qwen3-Embedding-0.6B embeddings for all person + merchant names.

Output:
  embeddings/
    embeddings.npz          - All embeddings + labels + metadata
    embeddings_sample.npz   - Random 30K sample for fast EDA
"""

import csv
import os
import time
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# ============================================================================
# CONFIG
# ============================================================================

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
BATCH_SIZE = 256  # Adjust based on VRAM
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "embeddings")
SAMPLE_SIZE = 30_000  # Subset for fast EDA

DATA_DIR = os.path.join(os.path.dirname(__file__))
PERSON_FILE = os.path.join(DATA_DIR, "indian_person_names_augmented.csv")
MERCHANT_FILE = os.path.join(DATA_DIR, "indian_merchant_names.csv")


# ============================================================================
# DATA LOADING
# ============================================================================

def load_dataset():
    """Load both datasets and combine them."""
    names = []
    labels = []       # 0 = PERSON, 1 = MERCHANT
    metadata = {
        "label_str": [],
        "category": [],
        "source": [],
        "augmentation": [],
    }

    # Person names
    print("Loading person names...")
    with open(PERSON_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            names.append(row["name"].strip())
            labels.append(0)
            metadata["label_str"].append("PERSON")
            metadata["category"].append(row.get("region", "unknown"))
            metadata["source"].append("person_" + row.get("format", "unknown"))
            metadata["augmentation"].append(row.get("augmentation", "unknown"))

    person_count = len(names)
    print(f"  -> {person_count:,} person names")

    # Merchant names
    print("Loading merchant names...")
    with open(MERCHANT_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            names.append(row["name"].strip())
            labels.append(1)
            metadata["label_str"].append("MERCHANT")
            metadata["category"].append(row.get("category", "unknown"))
            metadata["source"].append(row.get("source", "unknown"))
            metadata["augmentation"].append(row.get("augmentation", "unknown"))

    merchant_count = len(names) - person_count
    print(f"  -> {merchant_count:,} merchant names")
    print(f"  Total: {len(names):,}")

    return names, np.array(labels), metadata


# ============================================================================
# EMBEDDING GENERATION
# ============================================================================

def generate_embeddings(names: list[str], model: SentenceTransformer) -> np.ndarray:
    """Generate embeddings in batches with progress bar."""
    print(f"\nGenerating embeddings (batch_size={BATCH_SIZE})...")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Device: {model.device}")
    print(f"  Total names: {len(names):,}")

    start_time = time.time()

    # sentence-transformers handles batching internally
    embeddings = model.encode(
        names,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2 normalize for cosine similarity
    )

    elapsed = time.time() - start_time
    rate = len(names) / elapsed

    print(f"\n  Done in {elapsed:.1f}s ({rate:.0f} names/sec)")
    print(f"  Embedding shape: {embeddings.shape}")
    print(f"  Embedding dtype: {embeddings.dtype}")

    return embeddings


# ============================================================================
# SAVE
# ============================================================================

def save_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    names: list[str],
    metadata: dict,
    output_dir: str,
):
    """Save embeddings and metadata."""
    os.makedirs(output_dir, exist_ok=True)

    # Full dataset
    full_path = os.path.join(output_dir, "embeddings.npz")
    np.savez_compressed(
        full_path,
        embeddings=embeddings,
        labels=labels,
        names=np.array(names, dtype=object),
        label_str=np.array(metadata["label_str"], dtype=object),
        category=np.array(metadata["category"], dtype=object),
        source=np.array(metadata["source"], dtype=object),
        augmentation=np.array(metadata["augmentation"], dtype=object),
    )
    size_mb = os.path.getsize(full_path) / 1024 / 1024
    print(f"\n[OK] Full embeddings saved: {full_path} ({size_mb:.1f} MB)")

    # Sampled subset for fast EDA
    n = len(names)
    if n > SAMPLE_SIZE:
        np.random.seed(42)
        # Stratified sample: equal from PERSON and MERCHANT
        person_idx = np.where(labels == 0)[0]
        merchant_idx = np.where(labels == 1)[0]
        half = SAMPLE_SIZE // 2
        sampled_person = np.random.choice(person_idx, size=min(half, len(person_idx)), replace=False)
        sampled_merchant = np.random.choice(merchant_idx, size=min(half, len(merchant_idx)), replace=False)
        sample_idx = np.concatenate([sampled_person, sampled_merchant])
        np.random.shuffle(sample_idx)

        sample_path = os.path.join(output_dir, "embeddings_sample.npz")
        np.savez_compressed(
            sample_path,
            embeddings=embeddings[sample_idx],
            labels=labels[sample_idx],
            names=np.array(names, dtype=object)[sample_idx],
            label_str=np.array(metadata["label_str"], dtype=object)[sample_idx],
            category=np.array(metadata["category"], dtype=object)[sample_idx],
            source=np.array(metadata["source"], dtype=object)[sample_idx],
            augmentation=np.array(metadata["augmentation"], dtype=object)[sample_idx],
        )
        sample_mb = os.path.getsize(sample_path) / 1024 / 1024
        print(f"[OK] Sample embeddings saved: {sample_path} ({sample_mb:.1f} MB)")
        print(f"     Sample size: {len(sample_idx):,} (stratified)")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("EMBEDDING GENERATION PIPELINE")
    print("=" * 60)

    # Check GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"\nGPU: {gpu_name} ({vram:.1f} GB)")
    else:
        print("\nWARNING: No GPU detected, using CPU (will be slow)")

    # Load data
    names, labels, metadata = load_dataset()

    # Load model
    print(f"\nLoading model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME, device=device)
    print(f"  Model loaded on {device}")
    print(f"  Embedding dimension: {model.get_sentence_embedding_dimension()}")

    # Generate embeddings
    embeddings = generate_embeddings(names, model)

    # Save
    save_embeddings(embeddings, labels, names, metadata, OUTPUT_DIR)

    # Quick sanity check
    print("\n" + "=" * 60)
    print("SANITY CHECK")
    print("=" * 60)
    person_emb = embeddings[labels == 0]
    merchant_emb = embeddings[labels == 1]
    print(f"  Person embeddings:   {person_emb.shape}")
    print(f"  Merchant embeddings: {merchant_emb.shape}")
    print(f"  Person mean norm:    {np.linalg.norm(person_emb, axis=1).mean():.4f}")
    print(f"  Merchant mean norm:  {np.linalg.norm(merchant_emb, axis=1).mean():.4f}")

    # Average cosine similarity within and between classes
    np.random.seed(42)
    sample_n = 1000
    p_idx = np.random.choice(len(person_emb), sample_n, replace=False)
    m_idx = np.random.choice(len(merchant_emb), sample_n, replace=False)

    p_sample = person_emb[p_idx]
    m_sample = merchant_emb[m_idx]

    # Intra-class similarity (person-person)
    pp_sim = (p_sample @ p_sample.T)
    pp_mean = pp_sim[np.triu_indices(sample_n, k=1)].mean()

    # Intra-class similarity (merchant-merchant)
    mm_sim = (m_sample @ m_sample.T)
    mm_mean = mm_sim[np.triu_indices(sample_n, k=1)].mean()

    # Inter-class similarity (person-merchant)
    pm_sim = (p_sample @ m_sample.T)
    pm_mean = pm_sim.mean()

    print(f"\n  Avg cosine similarity (1K sample):")
    print(f"    Person-Person:     {pp_mean:.4f}")
    print(f"    Merchant-Merchant: {mm_mean:.4f}")
    print(f"    Person-Merchant:   {pm_mean:.4f}")
    print(f"    Separation gap:    {((pp_mean + mm_mean) / 2 - pm_mean):.4f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
