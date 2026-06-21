"""
Research-Grade EDA on Embeddings
==================================
Comprehensive analysis of the embedding space for PERSON vs MERCHANT classification.

Analyses:
1. Embedding Space Visualization (UMAP, t-SNE)
2. PCA Variance Analysis
3. Cosine Similarity Distributions
4. Class Separability Metrics
5. Nearest Neighbor Cross-Class Analysis
6. Augmentation Impact Analysis
7. Merchant Sub-Category Clustering
8. Edge Case / Ambiguity Analysis

Output: research/figures/ (all plots)
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("WARNING: umap-learn not installed, skipping UMAP")


# ============================================================================
# CONFIG
# ============================================================================

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
EMBEDDINGS_FILE = os.path.join(BASE_DIR, "embeddings", "embeddings_sample.npz")
FULL_EMBEDDINGS_FILE = os.path.join(BASE_DIR, "embeddings", "embeddings.npz")
FIGURES_DIR = os.path.join(BASE_DIR, "research", "figures")

# Plot styling
plt.rcParams.update({
    'figure.dpi': 150,
    'figure.figsize': (12, 8),
    'font.size': 11,
    'font.family': 'sans-serif',
    'axes.titlesize': 14,
    'axes.labelsize': 12,
})

PERSON_COLOR = '#2196F3'   # Blue
MERCHANT_COLOR = '#FF5722' # Orange-Red
PALETTE = {0: PERSON_COLOR, 1: MERCHANT_COLOR}
LABEL_MAP = {0: 'PERSON', 1: 'MERCHANT'}


# ============================================================================
# DATA LOADING
# ============================================================================

def load_embeddings(use_sample=True):
    """Load embeddings from npz file."""
    path = EMBEDDINGS_FILE if use_sample else FULL_EMBEDDINGS_FILE
    if not os.path.exists(path):
        # Fallback
        path = FULL_EMBEDDINGS_FILE if use_sample else EMBEDDINGS_FILE
    
    print(f"Loading embeddings from: {path}")
    data = np.load(path, allow_pickle=True)
    
    emb = data['embeddings']
    labels = data['labels']
    names = data['names']
    
    meta = {
        'label_str': data['label_str'] if 'label_str' in data else None,
        'category': data['category'] if 'category' in data else None,
        'source': data['source'] if 'source' in data else None,
        'augmentation': data['augmentation'] if 'augmentation' in data else None,
    }
    
    print(f"  Shape: {emb.shape}")
    print(f"  Labels: {Counter(labels)}")
    return emb, labels, names, meta


# ============================================================================
# 1. UMAP / t-SNE VISUALIZATION
# ============================================================================

def plot_umap(emb, labels, names, meta, fig_dir):
    """2D UMAP visualization colored by class."""
    if not HAS_UMAP:
        print("  Skipping UMAP (not installed)")
        return
    
    print("\n[1/8] Computing UMAP projection...")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=30,
        min_dist=0.3,
        metric='cosine',
        random_state=42,
        verbose=False,
    )
    emb_2d = reducer.fit_transform(emb)
    
    # Plot: PERSON vs MERCHANT
    fig, ax = plt.subplots(figsize=(14, 10))
    for label_val, label_name in LABEL_MAP.items():
        mask = labels == label_val
        ax.scatter(
            emb_2d[mask, 0], emb_2d[mask, 1],
            c=PALETTE[label_val], label=label_name,
            alpha=0.15, s=3, rasterized=True,
        )
    ax.set_title('UMAP: Person vs Merchant Embedding Space', fontsize=16, fontweight='bold')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.legend(fontsize=12, markerscale=5)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, '01_umap_person_vs_merchant.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  -> Saved 01_umap_person_vs_merchant.png")
    
    # Plot: Original vs Augmented
    if meta['augmentation'] is not None:
        fig, axes = plt.subplots(1, 2, figsize=(20, 9))
        
        for i, label_val in enumerate([0, 1]):
            ax = axes[i]
            mask = labels == label_val
            aug = meta['augmentation'][mask]
            emb_sub = emb_2d[mask]
            
            is_original = np.array([a == 'original' for a in aug])
            
            ax.scatter(emb_sub[~is_original, 0], emb_sub[~is_original, 1],
                      c='#BDBDBD', alpha=0.1, s=2, label='Augmented', rasterized=True)
            ax.scatter(emb_sub[is_original, 0], emb_sub[is_original, 1],
                      c=PALETTE[label_val], alpha=0.4, s=5, label='Original', rasterized=True)
            
            ax.set_title(f'{LABEL_MAP[label_val]}: Original vs Augmented', fontsize=14, fontweight='bold')
            ax.legend(fontsize=11, markerscale=5)
        
        plt.suptitle('UMAP: Impact of Augmentation on Embedding Space', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, '02_umap_augmentation_impact.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  -> Saved 02_umap_augmentation_impact.png")
    
    # Plot: Merchant sub-categories
    if meta['category'] is not None:
        merchant_mask = labels == 1
        merchant_emb_2d = emb_2d[merchant_mask]
        merchant_cats = meta['category'][merchant_mask]
        
        # Top 10 categories
        cat_counts = Counter(merchant_cats)
        top_cats = [c for c, _ in cat_counts.most_common(10)]
        
        fig, ax = plt.subplots(figsize=(14, 10))
        colors = plt.cm.tab10(np.linspace(0, 1, len(top_cats)))
        
        for i, cat in enumerate(top_cats):
            cat_mask = np.array([c == cat for c in merchant_cats])
            ax.scatter(
                merchant_emb_2d[cat_mask, 0], merchant_emb_2d[cat_mask, 1],
                c=[colors[i]], label=cat, alpha=0.25, s=4, rasterized=True,
            )
        
        ax.set_title('UMAP: Merchant Sub-Categories', fontsize=16, fontweight='bold')
        ax.legend(fontsize=10, markerscale=5, ncol=2)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, '03_umap_merchant_categories.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  -> Saved 03_umap_merchant_categories.png")
    
    return emb_2d


def plot_tsne(emb, labels, fig_dir, n_samples=50000):
    """t-SNE visualization (on subset for speed)."""
    print("\n[2/8] Computing t-SNE projection...")
    
    np.random.seed(42)
    if len(emb) > n_samples:
        idx = np.random.choice(len(emb), n_samples, replace=False)
        emb_sub = emb[idx]
        labels_sub = labels[idx]
    else:
        emb_sub = emb
        labels_sub = labels
    
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000, init='pca')
    emb_2d = tsne.fit_transform(emb_sub)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    for label_val, label_name in LABEL_MAP.items():
        mask = labels_sub == label_val
        ax.scatter(
            emb_2d[mask, 0], emb_2d[mask, 1],
            c=PALETTE[label_val], label=label_name,
            alpha=0.2, s=4, rasterized=True,
        )
    ax.set_title(f't-SNE: Person vs Merchant (n={n_samples:,})', fontsize=16, fontweight='bold')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.legend(fontsize=12, markerscale=5)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, '04_tsne_person_vs_merchant.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  -> Saved 04_tsne_person_vs_merchant.png")


# ============================================================================
# 2. PCA ANALYSIS
# ============================================================================

def plot_pca_analysis(emb, labels, fig_dir):
    """PCA variance analysis and 2D projection."""
    print("\n[3/8] PCA analysis...")
    
    pca_full = PCA(n_components=min(50, emb.shape[1]))
    pca_full.fit(emb)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Explained variance
    ax = axes[0]
    cumvar = np.cumsum(pca_full.explained_variance_ratio_) * 100
    ax.plot(range(1, len(cumvar) + 1), cumvar, 'b-o', markersize=4, linewidth=2)
    ax.axhline(y=90, color='r', linestyle='--', alpha=0.5, label='90% variance')
    ax.axhline(y=95, color='orange', linestyle='--', alpha=0.5, label='95% variance')
    n_90 = np.argmax(cumvar >= 90) + 1
    n_95 = np.argmax(cumvar >= 95) + 1
    ax.axvline(x=n_90, color='r', linestyle=':', alpha=0.5)
    ax.axvline(x=n_95, color='orange', linestyle=':', alpha=0.5)
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('Cumulative Explained Variance (%)')
    ax.set_title(f'PCA Variance (90%@{n_90}, 95%@{n_95} components)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2D PCA projection
    ax = axes[1]
    pca_2d = PCA(n_components=2)
    emb_2d = pca_2d.fit_transform(emb)
    
    for label_val, label_name in LABEL_MAP.items():
        mask = labels == label_val
        ax.scatter(
            emb_2d[mask, 0], emb_2d[mask, 1],
            c=PALETTE[label_val], label=label_name,
            alpha=0.1, s=3, rasterized=True,
        )
    var_explained = pca_2d.explained_variance_ratio_ * 100
    ax.set_xlabel(f'PC1 ({var_explained[0]:.1f}%)')
    ax.set_ylabel(f'PC2 ({var_explained[1]:.1f}%)')
    ax.set_title('PCA 2D Projection')
    ax.legend(fontsize=11, markerscale=5)
    
    plt.suptitle('PCA Analysis of Embedding Space', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, '05_pca_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> 90% variance at {n_90} components, 95% at {n_95}")
    print("  -> Saved 05_pca_analysis.png")
    
    return n_90, n_95


# ============================================================================
# 3. COSINE SIMILARITY DISTRIBUTIONS
# ============================================================================

def plot_similarity_distributions(emb, labels, fig_dir, sample_size=5000):
    """Cosine similarity distributions: intra-class vs inter-class."""
    print("\n[4/8] Cosine similarity distributions...")
    
    np.random.seed(42)
    p_idx = np.random.choice(np.where(labels == 0)[0], min(sample_size, (labels == 0).sum()), replace=False)
    m_idx = np.random.choice(np.where(labels == 1)[0], min(sample_size, (labels == 1).sum()), replace=False)
    
    p_emb = emb[p_idx]
    m_emb = emb[m_idx]
    
    # Compute pairwise similarities
    pp_sim = (p_emb @ p_emb.T)
    mm_sim = (m_emb @ m_emb.T)
    pm_sim = (p_emb @ m_emb.T)
    
    # Extract upper triangles for intra-class
    pp_vals = pp_sim[np.triu_indices(len(p_emb), k=1)]
    mm_vals = mm_sim[np.triu_indices(len(m_emb), k=1)]
    pm_vals = pm_sim.flatten()
    
    # Subsample for plotting
    max_plot = 500_000
    if len(pp_vals) > max_plot:
        pp_vals = np.random.choice(pp_vals, max_plot, replace=False)
    if len(mm_vals) > max_plot:
        mm_vals = np.random.choice(mm_vals, max_plot, replace=False)
    if len(pm_vals) > max_plot:
        pm_vals = np.random.choice(pm_vals, max_plot, replace=False)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    ax.hist(pp_vals, bins=100, alpha=0.5, color=PERSON_COLOR, label=f'Person-Person (mean={pp_vals.mean():.4f})', density=True)
    ax.hist(mm_vals, bins=100, alpha=0.5, color=MERCHANT_COLOR, label=f'Merchant-Merchant (mean={mm_vals.mean():.4f})', density=True)
    ax.hist(pm_vals, bins=100, alpha=0.5, color='#4CAF50', label=f'Person-Merchant (mean={pm_vals.mean():.4f})', density=True)
    
    ax.axvline(pp_vals.mean(), color=PERSON_COLOR, linestyle='--', linewidth=2)
    ax.axvline(mm_vals.mean(), color=MERCHANT_COLOR, linestyle='--', linewidth=2)
    ax.axvline(pm_vals.mean(), color='#4CAF50', linestyle='--', linewidth=2)
    
    ax.set_xlabel('Cosine Similarity', fontsize=13)
    ax.set_ylabel('Density', fontsize=13)
    ax.set_title('Cosine Similarity Distributions: Intra-Class vs Inter-Class', fontsize=16, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, '06_cosine_similarity_distributions.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Person-Person mean:     {pp_vals.mean():.4f} (std={pp_vals.std():.4f})")
    print(f"  Merchant-Merchant mean: {mm_vals.mean():.4f} (std={mm_vals.std():.4f})")
    print(f"  Person-Merchant mean:   {pm_vals.mean():.4f} (std={pm_vals.std():.4f})")
    print(f"  Separation gap:         {((pp_vals.mean() + mm_vals.mean()) / 2 - pm_vals.mean()):.4f}")
    print("  -> Saved 06_cosine_similarity_distributions.png")


# ============================================================================
# 4. SEPARABILITY METRICS
# ============================================================================

def compute_separability_metrics(emb, labels, fig_dir):
    """Compute cluster quality and linear separability metrics."""
    print("\n[5/8] Separability metrics...")
    
    # Use larger subset for robust metrics
    np.random.seed(42)
    n = min(50000, len(emb))
    idx = np.random.choice(len(emb), n, replace=False)
    emb_sub = emb[idx]
    labels_sub = labels[idx]
    
    # Silhouette score
    print("  Computing silhouette score...")
    sil = silhouette_score(emb_sub, labels_sub, metric='cosine', sample_size=5000)
    
    # Calinski-Harabasz
    ch = calinski_harabasz_score(emb_sub, labels_sub)
    
    # Fisher's Linear Discriminant Ratio
    p_emb = emb_sub[labels_sub == 0]
    m_emb = emb_sub[labels_sub == 1]
    p_mean = p_emb.mean(axis=0)
    m_mean = m_emb.mean(axis=0)
    p_var = p_emb.var(axis=0).mean()
    m_var = m_emb.var(axis=0).mean()
    between_class_var = np.sum((p_mean - m_mean) ** 2)
    within_class_var = p_var + m_var
    fisher_ratio = between_class_var / within_class_var if within_class_var > 0 else 0
    
    # Logistic Regression cross-validation
    print("  Computing LR cross-validation accuracy...")
    lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(lr, emb_sub, labels_sub, cv=cv, scoring='accuracy')
    
    # Print results
    print(f"\n  === SEPARABILITY METRICS ===")
    print(f"  Silhouette Score:        {sil:.4f}  (range: -1 to 1, higher = better separation)")
    print(f"  Calinski-Harabasz:       {ch:.1f}  (higher = better defined clusters)")
    print(f"  Fisher Discriminant:     {fisher_ratio:.4f}  (higher = more separable)")
    print(f"  LR 5-Fold Accuracy:      {scores.mean():.4f} +/- {scores.std():.4f}")
    print(f"  LR Per-Fold:             {[f'{s:.4f}' for s in scores]}")
    
    # Save metrics as a bar chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Metric values
    ax = axes[0]
    metrics = ['Silhouette\nScore', 'Fisher\nDiscriminant', 'LR Accuracy\n(5-fold CV)']
    values = [sil, min(fisher_ratio, 1.0), scores.mean()]
    colors_list = ['#2196F3', '#4CAF50', '#FF9800']
    bars = ax.bar(metrics, values, color=colors_list, edgecolor='white', linewidth=2)
    ax.set_ylim(0, 1.05)
    ax.set_title('Separability Metrics', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.4f}', ha='center', fontsize=12, fontweight='bold')
    
    # CV accuracy boxplot
    ax = axes[1]
    ax.bar(range(5), scores, color='#FF9800', edgecolor='white', linewidth=2, alpha=0.8)
    ax.axhline(scores.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {scores.mean():.4f}')
    ax.set_xlabel('Fold')
    ax.set_ylabel('Accuracy')
    ax.set_title('Logistic Regression 5-Fold CV', fontsize=14, fontweight='bold')
    ax.set_xticks(range(5))
    ax.set_xticklabels([f'Fold {i+1}' for i in range(5)])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, '07_separability_metrics.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  -> Saved 07_separability_metrics.png")
    
    return {
        'silhouette': sil,
        'calinski_harabasz': ch,
        'fisher_ratio': fisher_ratio,
        'lr_accuracy_mean': scores.mean(),
        'lr_accuracy_std': scores.std(),
        'lr_fold_scores': scores,
    }


# ============================================================================
# 5. NEAREST NEIGHBOR CROSS-CLASS ANALYSIS
# ============================================================================

def nearest_neighbor_analysis(emb, labels, names, fig_dir, k=5, sample_size=5000):
    """Find names closest to the decision boundary."""
    print("\n[6/8] Nearest neighbor cross-class analysis...")
    
    np.random.seed(42)
    
    # Sample
    p_idx = np.random.choice(np.where(labels == 0)[0], min(sample_size, (labels == 0).sum()), replace=False)
    m_idx = np.random.choice(np.where(labels == 1)[0], min(sample_size, (labels == 1).sum()), replace=False)
    
    p_emb = emb[p_idx]
    m_emb = emb[m_idx]
    p_names = names[p_idx]
    m_names = names[m_idx]
    
    # For each person, find nearest merchant
    pm_sim = p_emb @ m_emb.T  # [P, M]
    
    # Most ambiguous persons (highest similarity to any merchant)
    max_sim_per_person = pm_sim.max(axis=1)
    top_ambiguous_person_idx = np.argsort(max_sim_per_person)[-20:][::-1]
    
    print(f"\n  TOP 20 Most Ambiguous PERSON Names (closest to a merchant):")
    print(f"  {'Person Name':40s} {'Nearest Merchant':40s} {'Cosine Sim':>10s}")
    print(f"  {'-'*40} {'-'*40} {'-'*10}")
    
    ambiguous_persons = []
    for idx in top_ambiguous_person_idx:
        nearest_m_idx = pm_sim[idx].argmax()
        sim = pm_sim[idx, nearest_m_idx]
        p_name = str(p_names[idx])
        m_name = str(m_names[nearest_m_idx])
        print(f"  {p_name:40s} {m_name:40s} {sim:10.4f}")
        ambiguous_persons.append((p_name, m_name, sim))
    
    # Most ambiguous merchants (highest similarity to any person)
    mp_sim = pm_sim.T  # [M, P]
    max_sim_per_merchant = mp_sim.max(axis=1)
    top_ambiguous_merchant_idx = np.argsort(max_sim_per_merchant)[-20:][::-1]
    
    print(f"\n  TOP 20 Most Ambiguous MERCHANT Names (closest to a person):")
    print(f"  {'Merchant Name':40s} {'Nearest Person':40s} {'Cosine Sim':>10s}")
    print(f"  {'-'*40} {'-'*40} {'-'*10}")
    
    ambiguous_merchants = []
    for idx in top_ambiguous_merchant_idx:
        nearest_p_idx = mp_sim[idx].argmax()
        sim = mp_sim[idx, nearest_p_idx]
        m_name = str(m_names[idx])
        p_name = str(p_names[nearest_p_idx])
        print(f"  {m_name:40s} {p_name:40s} {sim:10.4f}")
        ambiguous_merchants.append((m_name, p_name, sim))
    
    # Plot distribution of max cross-class similarity
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    ax = axes[0]
    ax.hist(max_sim_per_person, bins=80, color=PERSON_COLOR, alpha=0.7, edgecolor='white')
    ax.axvline(max_sim_per_person.mean(), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {max_sim_per_person.mean():.4f}')
    ax.set_xlabel('Max Cosine Similarity to Nearest Merchant')
    ax.set_ylabel('Count')
    ax.set_title('Person Names: Distance to Nearest Merchant', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    ax.hist(max_sim_per_merchant, bins=80, color=MERCHANT_COLOR, alpha=0.7, edgecolor='white')
    ax.axvline(max_sim_per_merchant.mean(), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {max_sim_per_merchant.mean():.4f}')
    ax.set_xlabel('Max Cosine Similarity to Nearest Person')
    ax.set_ylabel('Count')
    ax.set_title('Merchant Names: Distance to Nearest Person', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Cross-Class Nearest Neighbor Similarity', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, '08_nearest_neighbor_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  -> Saved 08_nearest_neighbor_analysis.png")


# ============================================================================
# 6. EMBEDDING NORM ANALYSIS
# ============================================================================

def plot_embedding_norms(emb, labels, fig_dir):
    """Analyze embedding vector norms by class."""
    print("\n[7/8] Embedding norm analysis...")
    
    norms = np.linalg.norm(emb, axis=1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Norm distribution by class
    ax = axes[0]
    p_norms = norms[labels == 0]
    m_norms = norms[labels == 1]
    ax.hist(p_norms, bins=80, alpha=0.6, color=PERSON_COLOR, label=f'Person (mean={p_norms.mean():.4f})', density=True)
    ax.hist(m_norms, bins=80, alpha=0.6, color=MERCHANT_COLOR, label=f'Merchant (mean={m_norms.mean():.4f})', density=True)
    ax.set_xlabel('L2 Norm')
    ax.set_ylabel('Density')
    ax.set_title('Embedding Norm Distribution', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Norm vs mean embedding similarity
    ax = axes[1]
    overall_mean = emb.mean(axis=0)
    sims_to_mean = emb @ overall_mean / (np.linalg.norm(overall_mean) * norms + 1e-8)
    
    ax.scatter(sims_to_mean[labels == 0], norms[labels == 0],
               c=PERSON_COLOR, alpha=0.1, s=2, label='Person', rasterized=True)
    ax.scatter(sims_to_mean[labels == 1], norms[labels == 1],
               c=MERCHANT_COLOR, alpha=0.1, s=2, label='Merchant', rasterized=True)
    ax.set_xlabel('Similarity to Global Centroid')
    ax.set_ylabel('L2 Norm')
    ax.set_title('Norm vs Centroid Similarity', fontweight='bold')
    ax.legend(markerscale=5)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, '09_embedding_norms.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  -> Saved 09_embedding_norms.png")


# ============================================================================
# 7. SUMMARY DASHBOARD
# ============================================================================

def create_summary(emb, labels, meta, metrics, fig_dir):
    """Create a summary text file with all findings."""
    print("\n[8/8] Creating summary...")
    
    summary_path = os.path.join(os.path.dirname(fig_dir), '03_embedding_eda_findings.md')
    
    p_count = (labels == 0).sum()
    m_count = (labels == 1).sum()
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("# Stage 3: Embedding EDA Findings\n\n")
        f.write(f"**Date:** {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"**Status:** Complete\n\n---\n\n")
        
        f.write("## Dataset Summary\n\n")
        f.write(f"| Metric | Value |\n|:---|---:|\n")
        f.write(f"| Embedding Model | Qwen/Qwen3-Embedding-0.6B |\n")
        f.write(f"| Embedding Dimension | {emb.shape[1]} |\n")
        f.write(f"| Total Embeddings | {len(emb):,} |\n")
        f.write(f"| Person Names | {p_count:,} |\n")
        f.write(f"| Merchant Names | {m_count:,} |\n\n")
        
        f.write("## Separability Metrics\n\n")
        f.write(f"| Metric | Value | Interpretation |\n|:---|---:|:---|\n")
        f.write(f"| Silhouette Score | {metrics['silhouette']:.4f} | ")
        if metrics['silhouette'] > 0.5:
            f.write("Excellent separation |\n")
        elif metrics['silhouette'] > 0.25:
            f.write("Good separation |\n")
        else:
            f.write("Weak separation |\n")
        f.write(f"| Calinski-Harabasz | {metrics['calinski_harabasz']:.1f} | Higher = better |\n")
        f.write(f"| Fisher Discriminant | {metrics['fisher_ratio']:.4f} | Higher = more separable |\n")
        f.write(f"| **LR Accuracy (5-fold)** | **{metrics['lr_accuracy_mean']:.4f}** | +/- {metrics['lr_accuracy_std']:.4f} |\n\n")
        
        f.write("## Figures\n\n")
        f.write("| # | Figure | Description |\n|:---|:---|:---|\n")
        f.write("| 1 | `01_umap_person_vs_merchant.png` | UMAP 2D projection colored by class |\n")
        f.write("| 2 | `02_umap_augmentation_impact.png` | Original vs augmented names in UMAP space |\n")
        f.write("| 3 | `03_umap_merchant_categories.png` | Merchant sub-categories in UMAP space |\n")
        f.write("| 4 | `04_tsne_person_vs_merchant.png` | t-SNE 2D projection |\n")
        f.write("| 5 | `05_pca_analysis.png` | PCA variance curve + 2D projection |\n")
        f.write("| 6 | `06_cosine_similarity_distributions.png` | Intra vs inter-class similarity |\n")
        f.write("| 7 | `07_separability_metrics.png` | Separability metric bar charts |\n")
        f.write("| 8 | `08_nearest_neighbor_analysis.png` | Cross-class nearest neighbor distances |\n")
        f.write("| 9 | `09_embedding_norms.png` | Embedding norm distributions |\n\n")
        
        f.write("## Next Steps\n\n")
        f.write("- [ ] Train production classifier (Logistic Regression / MLP)\n")
        f.write("- [ ] Evaluate on held-out test set\n")
        f.write("- [ ] Test edge cases and hard examples\n")
        f.write("- [ ] Package for inference\n")
    
    print(f"  -> Saved {summary_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("RESEARCH-GRADE EMBEDDING EDA")
    print("=" * 60)
    
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    # Load data
    emb, labels, names, meta = load_embeddings(use_sample=False)
    
    # Run all analyses
    plot_umap(emb, labels, names, meta, FIGURES_DIR)
    plot_tsne(emb, labels, FIGURES_DIR)
    plot_pca_analysis(emb, labels, FIGURES_DIR)
    plot_similarity_distributions(emb, labels, FIGURES_DIR)
    metrics = compute_separability_metrics(emb, labels, FIGURES_DIR)
    nearest_neighbor_analysis(emb, labels, names, FIGURES_DIR)
    plot_embedding_norms(emb, labels, FIGURES_DIR)
    create_summary(emb, labels, meta, metrics, FIGURES_DIR)
    
    print("\n" + "=" * 60)
    print("ALL ANALYSES COMPLETE")
    print(f"Figures saved to: {FIGURES_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
