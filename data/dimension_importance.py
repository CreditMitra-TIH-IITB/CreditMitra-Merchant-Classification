"""
Embedding Dimension Importance Analysis
==========================================
Which of the 1024 dimensions actually help classify Person vs Merchant?

Analyses:
1. Per-dimension Fisher Discriminant Ratio
2. Per-dimension t-test (statistical significance)
3. Logistic Regression coefficient importance
4. Dimension ablation: Top-K vs Bottom-K vs Random-K accuracy
5. Cumulative accuracy curve (adding dimensions by importance)
6. Redundancy analysis (correlated dimensions)

Output: research/figures/10-15_*.png
"""

import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from scipy import stats
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# CONFIG
# ============================================================================

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
EMBEDDINGS_FILE = os.path.join(BASE_DIR, "embeddings", "embeddings.npz")
FIGURES_DIR = os.path.join(BASE_DIR, "research", "figures")

PERSON_COLOR = '#2196F3'
MERCHANT_COLOR = '#FF5722'

plt.rcParams.update({
    'figure.dpi': 150,
    'figure.figsize': (14, 8),
    'font.size': 11,
    'font.family': 'sans-serif',
    'axes.titlesize': 14,
    'axes.labelsize': 12,
})


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(n_sample=50000):
    """Load embeddings with stratified sampling for speed."""
    print("Loading embeddings...")
    data = np.load(EMBEDDINGS_FILE, allow_pickle=True)
    emb = data['embeddings']
    labels = data['labels']
    names = data['names']

    print(f"  Full dataset: {emb.shape}")

    # Stratified sample
    np.random.seed(42)
    p_idx = np.where(labels == 0)[0]
    m_idx = np.where(labels == 1)[0]
    half = n_sample // 2
    s_p = np.random.choice(p_idx, min(half, len(p_idx)), replace=False)
    s_m = np.random.choice(m_idx, min(half, len(m_idx)), replace=False)
    idx = np.concatenate([s_p, s_m])
    np.random.shuffle(idx)

    emb_s = emb[idx]
    labels_s = labels[idx]
    names_s = names[idx]
    print(f"  Sampled: {emb_s.shape} (stratified)")
    return emb_s, labels_s, names_s, emb, labels


# ============================================================================
# 1. PER-DIMENSION FISHER DISCRIMINANT RATIO
# ============================================================================

def compute_fisher_per_dim(emb, labels):
    """Fisher's discriminant ratio for each dimension independently."""
    print("\n[1/6] Per-dimension Fisher Discriminant Ratio...")

    p_emb = emb[labels == 0]
    m_emb = emb[labels == 1]

    p_mean = p_emb.mean(axis=0)  # (1024,)
    m_mean = m_emb.mean(axis=0)

    p_var = p_emb.var(axis=0)
    m_var = m_emb.var(axis=0)

    # Fisher = (mu1 - mu2)^2 / (var1 + var2)
    fisher = (p_mean - m_mean) ** 2 / (p_var + m_var + 1e-10)

    return fisher


def plot_fisher_per_dim(fisher, fig_dir):
    """Plot per-dimension Fisher scores."""
    dim_order = np.argsort(fisher)[::-1]

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # Top: Full bar chart sorted by importance
    ax = axes[0, 0]
    ax.bar(range(len(fisher)), fisher[dim_order], color='#2196F3', alpha=0.7, width=1.0)
    ax.set_xlabel('Dimension (sorted by Fisher score)')
    ax.set_ylabel('Fisher Discriminant Ratio')
    ax.set_title('All 1024 Dimensions: Fisher Discriminant Score', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    # Mark top 50 zone
    ax.axvline(x=50, color='red', linestyle='--', alpha=0.7, label='Top 50')
    ax.axvline(x=100, color='orange', linestyle='--', alpha=0.7, label='Top 100')
    ax.axvline(x=200, color='green', linestyle='--', alpha=0.7, label='Top 200')
    ax.legend()

    # Top-right: Cumulative Fisher
    ax = axes[0, 1]
    cum_fisher = np.cumsum(fisher[dim_order])
    cum_pct = cum_fisher / cum_fisher[-1] * 100
    ax.plot(range(1, len(fisher) + 1), cum_pct, 'b-', linewidth=2)
    ax.axhline(y=80, color='red', linestyle='--', alpha=0.5, label='80% discriminability')
    ax.axhline(y=90, color='orange', linestyle='--', alpha=0.5, label='90% discriminability')
    ax.axhline(y=95, color='green', linestyle='--', alpha=0.5, label='95% discriminability')
    n_80 = np.argmax(cum_pct >= 80) + 1
    n_90 = np.argmax(cum_pct >= 90) + 1
    n_95 = np.argmax(cum_pct >= 95) + 1
    ax.axvline(x=n_80, color='red', linestyle=':', alpha=0.5)
    ax.axvline(x=n_90, color='orange', linestyle=':', alpha=0.5)
    ax.axvline(x=n_95, color='green', linestyle=':', alpha=0.5)
    ax.set_xlabel('Number of Dimensions')
    ax.set_ylabel('Cumulative Fisher Score (%)')
    ax.set_title(f'Cumulative Discriminability (80%@{n_80}, 90%@{n_90}, 95%@{n_95} dims)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom-left: Histogram of Fisher scores
    ax = axes[1, 0]
    ax.hist(fisher, bins=80, color='#2196F3', alpha=0.7, edgecolor='white')
    ax.axvline(fisher.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {fisher.mean():.4f}')
    ax.axvline(np.median(fisher), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(fisher):.4f}')
    ax.set_xlabel('Fisher Discriminant Ratio')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Fisher Scores Across Dimensions', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom-right: Top 30 dimensions
    ax = axes[1, 1]
    top_30 = dim_order[:30]
    colors = plt.cm.YlOrRd(np.linspace(0.3, 1.0, 30))
    bars = ax.barh(range(30), fisher[top_30[::-1]], color=colors[::-1], edgecolor='white')
    ax.set_yticks(range(30))
    ax.set_yticklabels([f'dim_{d}' for d in top_30[::-1]], fontsize=9)
    ax.set_xlabel('Fisher Discriminant Ratio')
    ax.set_title('Top 30 Most Discriminative Dimensions', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    plt.suptitle('Per-Dimension Fisher Discriminant Analysis', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, '10_fisher_per_dimension.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Top 5 dims: {dim_order[:5]} (Fisher: {fisher[dim_order[:5]]})")
    print(f"  80% discriminability at {n_80} dims, 90% at {n_90}, 95% at {n_95}")
    print("  -> Saved 10_fisher_per_dimension.png")

    return dim_order, n_80, n_90, n_95


# ============================================================================
# 2. PER-DIMENSION T-TEST
# ============================================================================

def compute_ttest_per_dim(emb, labels):
    """Independent t-test per dimension."""
    print("\n[2/6] Per-dimension t-tests...")

    p_emb = emb[labels == 0]
    m_emb = emb[labels == 1]

    t_stats = np.zeros(emb.shape[1])
    p_values = np.zeros(emb.shape[1])

    for d in range(emb.shape[1]):
        t, p = stats.ttest_ind(p_emb[:, d], m_emb[:, d], equal_var=False)
        t_stats[d] = abs(t)
        p_values[d] = p

    # Bonferroni correction
    sig_threshold = 0.05 / emb.shape[1]
    n_significant = (p_values < sig_threshold).sum()

    print(f"  Significant dimensions (Bonferroni alpha=0.05): {n_significant}/{emb.shape[1]}")
    print(f"  Non-significant dimensions: {emb.shape[1] - n_significant}")

    return t_stats, p_values, n_significant


def plot_ttest(t_stats, p_values, fisher, fig_dir):
    """Plot t-test results."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # t-statistic distribution
    ax = axes[0]
    ax.hist(t_stats, bins=80, color='#4CAF50', alpha=0.7, edgecolor='white')
    ax.axvline(t_stats.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean |t|: {t_stats.mean():.1f}')
    ax.set_xlabel('|t-statistic|')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of |t-statistics|', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # -log10(p-value) distribution
    ax = axes[1]
    neg_log_p = -np.log10(p_values + 1e-300)
    bonferroni = -np.log10(0.05 / len(p_values))
    ax.hist(neg_log_p, bins=80, color='#FF9800', alpha=0.7, edgecolor='white')
    ax.axvline(bonferroni, color='red', linestyle='--', linewidth=2, label=f'Bonferroni threshold')
    ax.set_xlabel('-log10(p-value)')
    ax.set_ylabel('Count')
    ax.set_title('Statistical Significance per Dimension', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Fisher vs t-statistic correlation
    ax = axes[2]
    ax.scatter(fisher, t_stats, c='#2196F3', alpha=0.4, s=8, rasterized=True)
    ax.set_xlabel('Fisher Discriminant Ratio')
    ax.set_ylabel('|t-statistic|')
    ax.set_title('Fisher vs t-statistic (should correlate)', fontweight='bold')
    corr = np.corrcoef(fisher, t_stats)[0, 1]
    ax.text(0.05, 0.95, f'r = {corr:.4f}', transform=ax.transAxes, fontsize=13,
            fontweight='bold', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.grid(True, alpha=0.3)

    plt.suptitle('Per-Dimension Statistical Testing', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, '11_ttest_per_dimension.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  -> Saved 11_ttest_per_dimension.png")


# ============================================================================
# 3. LOGISTIC REGRESSION COEFFICIENT IMPORTANCE
# ============================================================================

def lr_coefficient_analysis(emb, labels, fig_dir):
    """Train LR and analyze which dimensions get the highest weights."""
    print("\n[3/6] Logistic Regression coefficient analysis...")

    # Standardize for comparable coefficients
    scaler = StandardScaler()
    emb_scaled = scaler.fit_transform(emb)

    lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    lr.fit(emb_scaled, labels)

    # Coefficients (absolute)
    coefs = np.abs(lr.coef_[0])  # shape (1024,)
    coef_order = np.argsort(coefs)[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # All coefficients sorted
    ax = axes[0]
    ax.bar(range(len(coefs)), coefs[coef_order], color='#9C27B0', alpha=0.7, width=1.0)
    ax.set_xlabel('Dimension (sorted by |coefficient|)')
    ax.set_ylabel('|LR Coefficient|')
    ax.set_title('Logistic Regression: All Dimension Weights', fontweight='bold')
    ax.axvline(x=50, color='red', linestyle='--', alpha=0.7, label='Top 50')
    ax.axvline(x=100, color='orange', linestyle='--', alpha=0.7, label='Top 100')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Top 30 dimensions
    ax = axes[1]
    top_30 = coef_order[:30]
    colors = plt.cm.Purples(np.linspace(0.3, 1.0, 30))
    ax.barh(range(30), coefs[top_30[::-1]], color=colors[::-1], edgecolor='white')
    ax.set_yticks(range(30))
    ax.set_yticklabels([f'dim_{d}' for d in top_30[::-1]], fontsize=9)
    ax.set_xlabel('|LR Coefficient|')
    ax.set_title('Top 30 Highest-Weight Dimensions', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    plt.suptitle('Logistic Regression Coefficient Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, '12_lr_coefficients.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Stats
    print(f"  Top 5 LR dims: {coef_order[:5]} (coefs: {coefs[coef_order[:5]].round(4)})")
    print(f"  Bottom 5 LR dims: {coef_order[-5:]} (coefs: {coefs[coef_order[-5:]].round(4)})")
    print(f"  Coef range: [{coefs.min():.4f}, {coefs.max():.4f}]")
    print("  -> Saved 12_lr_coefficients.png")

    return coefs, coef_order


# ============================================================================
# 4. DIMENSION ABLATION STUDY
# ============================================================================

def dimension_ablation(emb, labels, fisher_order, coef_order, fig_dir):
    """Test accuracy with Top-K, Bottom-K, and Random-K dimensions."""
    print("\n[4/6] Dimension ablation study...")

    k_values = [10, 25, 50, 100, 150, 200, 300, 400, 512, 768, 1024]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = {
        'k': [],
        'top_fisher': [],
        'bottom_fisher': [],
        'top_lr': [],
        'random': [],
    }

    for k in k_values:
        print(f"  k={k}...", end=" ", flush=True)
        results['k'].append(k)

        # Top-K Fisher dimensions
        top_dims = fisher_order[:k]
        lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        scores = cross_val_score(lr, emb[:, top_dims], labels, cv=cv, scoring='accuracy')
        results['top_fisher'].append(scores.mean())

        # Bottom-K Fisher dimensions
        bottom_dims = fisher_order[-k:]
        scores = cross_val_score(lr, emb[:, bottom_dims], labels, cv=cv, scoring='accuracy')
        results['bottom_fisher'].append(scores.mean())

        # Top-K LR coefficient dimensions
        top_lr_dims = coef_order[:k]
        scores = cross_val_score(lr, emb[:, top_lr_dims], labels, cv=cv, scoring='accuracy')
        results['top_lr'].append(scores.mean())

        # Random-K dimensions
        np.random.seed(42)
        rand_dims = np.random.choice(1024, k, replace=False)
        scores = cross_val_score(lr, emb[:, rand_dims], labels, cv=cv, scoring='accuracy')
        results['random'].append(scores.mean())

        print(f"Fisher-top={results['top_fisher'][-1]:.4f}, "
              f"Fisher-bottom={results['bottom_fisher'][-1]:.4f}, "
              f"LR-top={results['top_lr'][-1]:.4f}, "
              f"Random={results['random'][-1]:.4f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    ax = axes[0]
    ax.plot(results['k'], results['top_fisher'], 'b-o', linewidth=2, markersize=6, label='Top-K (Fisher)')
    ax.plot(results['k'], results['top_lr'], 'purple', linestyle='-', marker='s', linewidth=2, markersize=6, label='Top-K (LR coef)')
    ax.plot(results['k'], results['random'], 'gray', linestyle='--', marker='^', linewidth=2, markersize=6, label='Random-K')
    ax.plot(results['k'], results['bottom_fisher'], 'r-v', linewidth=2, markersize=6, label='Bottom-K (Fisher)')
    ax.set_xlabel('Number of Dimensions (K)')
    ax.set_ylabel('5-Fold CV Accuracy')
    ax.set_title('Dimension Ablation: Accuracy vs K Dimensions', fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, 1.0)

    # Gap between top and bottom
    ax = axes[1]
    gap = np.array(results['top_fisher']) - np.array(results['bottom_fisher'])
    ax.bar(range(len(results['k'])), gap * 100, color='#FF5722', alpha=0.7, edgecolor='white')
    ax.set_xticks(range(len(results['k'])))
    ax.set_xticklabels(results['k'], fontsize=9)
    ax.set_xlabel('Number of Dimensions (K)')
    ax.set_ylabel('Accuracy Gap (Top - Bottom) %')
    ax.set_title('How Much Better are Top Dims vs Bottom Dims?', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for i, (k, g) in enumerate(zip(results['k'], gap)):
        if k <= 512:
            ax.text(i, g * 100 + 0.3, f'{g*100:.1f}%', ha='center', fontsize=9, fontweight='bold')

    plt.suptitle('Dimension Ablation Study', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, '13_dimension_ablation.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  -> Saved 13_dimension_ablation.png")

    return results


# ============================================================================
# 5. DIMENSION VALUE DISTRIBUTIONS (TOP vs BOTTOM)
# ============================================================================

def plot_dimension_distributions(emb, labels, fisher_order, fig_dir):
    """Show value distributions for top vs bottom dimensions."""
    print("\n[5/6] Dimension value distributions...")

    fig, axes = plt.subplots(2, 5, figsize=(25, 10))

    # Top 5 most discriminative
    for i, dim in enumerate(fisher_order[:5]):
        ax = axes[0, i]
        p_vals = emb[labels == 0, dim]
        m_vals = emb[labels == 1, dim]
        ax.hist(p_vals, bins=60, alpha=0.5, color=PERSON_COLOR, density=True, label='Person')
        ax.hist(m_vals, bins=60, alpha=0.5, color=MERCHANT_COLOR, density=True, label='Merchant')
        ax.set_title(f'dim_{dim} (BEST #{i+1})', fontweight='bold', fontsize=11)
        ax.legend(fontsize=8)
        ax.set_ylabel('Density' if i == 0 else '')

    # Bottom 5 least discriminative
    for i, dim in enumerate(fisher_order[-5:]):
        ax = axes[1, i]
        p_vals = emb[labels == 0, dim]
        m_vals = emb[labels == 1, dim]
        ax.hist(p_vals, bins=60, alpha=0.5, color=PERSON_COLOR, density=True, label='Person')
        ax.hist(m_vals, bins=60, alpha=0.5, color=MERCHANT_COLOR, density=True, label='Merchant')
        ax.set_title(f'dim_{dim} (WORST #{i+1})', fontweight='bold', fontsize=11)
        ax.legend(fontsize=8)
        ax.set_ylabel('Density' if i == 0 else '')

    axes[0, 0].set_ylabel('Top 5\nDensity', fontweight='bold')
    axes[1, 0].set_ylabel('Bottom 5\nDensity', fontweight='bold')

    plt.suptitle('Value Distributions: Most vs Least Discriminative Dimensions',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, '14_dimension_distributions.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  -> Saved 14_dimension_distributions.png")


# ============================================================================
# 6. CROSS-METHOD AGREEMENT
# ============================================================================

def plot_method_agreement(fisher, coefs, t_stats, fig_dir):
    """Compare Fisher, LR, and t-test rankings."""
    print("\n[6/6] Cross-method agreement analysis...")

    fisher_rank = np.argsort(np.argsort(fisher)[::-1])  # rank 0 = best
    coef_rank = np.argsort(np.argsort(coefs)[::-1])
    ttest_rank = np.argsort(np.argsort(t_stats)[::-1])

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Fisher vs LR
    ax = axes[0]
    ax.scatter(fisher_rank, coef_rank, c='#2196F3', alpha=0.3, s=8, rasterized=True)
    corr = np.corrcoef(fisher_rank, coef_rank)[0, 1]
    ax.set_xlabel('Fisher Rank')
    ax.set_ylabel('LR Coefficient Rank')
    ax.set_title(f'Fisher vs LR Coefficient (Spearman ρ={corr:.3f})', fontweight='bold')
    ax.plot([0, 1024], [0, 1024], 'r--', alpha=0.5)
    ax.grid(True, alpha=0.3)

    # Fisher vs t-test
    ax = axes[1]
    ax.scatter(fisher_rank, ttest_rank, c='#4CAF50', alpha=0.3, s=8, rasterized=True)
    corr = np.corrcoef(fisher_rank, ttest_rank)[0, 1]
    ax.set_xlabel('Fisher Rank')
    ax.set_ylabel('t-test Rank')
    ax.set_title(f'Fisher vs t-test (Spearman ρ={corr:.3f})', fontweight='bold')
    ax.plot([0, 1024], [0, 1024], 'r--', alpha=0.5)
    ax.grid(True, alpha=0.3)

    # LR vs t-test
    ax = axes[2]
    ax.scatter(coef_rank, ttest_rank, c='#FF9800', alpha=0.3, s=8, rasterized=True)
    corr = np.corrcoef(coef_rank, ttest_rank)[0, 1]
    ax.set_xlabel('LR Coefficient Rank')
    ax.set_ylabel('t-test Rank')
    ax.set_title(f'LR vs t-test (Spearman ρ={corr:.3f})', fontweight='bold')
    ax.plot([0, 1024], [0, 1024], 'r--', alpha=0.5)
    ax.grid(True, alpha=0.3)

    plt.suptitle('Cross-Method Agreement: Which Dimensions are Consistently Important?',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, '15_method_agreement.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Find consensus important dimensions (top 50 in ALL three methods)
    fisher_top50 = set(np.argsort(fisher)[::-1][:50])
    coef_top50 = set(np.argsort(coefs)[::-1][:50])
    ttest_top50 = set(np.argsort(t_stats)[::-1][:50])

    consensus = fisher_top50 & coef_top50 & ttest_top50
    any_two = (fisher_top50 & coef_top50) | (fisher_top50 & ttest_top50) | (coef_top50 & ttest_top50)

    print(f"  Consensus top-50 (all 3 methods agree): {len(consensus)} dimensions")
    print(f"  Any-2 top-50 (2+ methods agree): {len(any_two)} dimensions")
    print(f"  Consensus dims: {sorted(consensus)[:20]}...")
    print("  -> Saved 15_method_agreement.png")

    return consensus, any_two


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("EMBEDDING DIMENSION IMPORTANCE ANALYSIS")
    print("=" * 60)

    os.makedirs(FIGURES_DIR, exist_ok=True)
    start = time.time()

    # Load data
    emb, labels, names, emb_full, labels_full = load_data(n_sample=50000)

    # 1. Fisher per dimension
    fisher = compute_fisher_per_dim(emb, labels)
    fisher_order, n_80, n_90, n_95 = plot_fisher_per_dim(fisher, FIGURES_DIR)

    # 2. t-test per dimension
    t_stats, p_values, n_sig = compute_ttest_per_dim(emb, labels)
    plot_ttest(t_stats, p_values, fisher, FIGURES_DIR)

    # 3. LR coefficient analysis
    coefs, coef_order = lr_coefficient_analysis(emb, labels, FIGURES_DIR)

    # 4. Dimension ablation
    ablation = dimension_ablation(emb, labels, fisher_order, coef_order, FIGURES_DIR)

    # 5. Dimension distributions
    plot_dimension_distributions(emb, labels, fisher_order, FIGURES_DIR)

    # 6. Cross-method agreement
    consensus, any_two = plot_method_agreement(fisher, coefs, t_stats, FIGURES_DIR)

    # Summary
    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"ANALYSIS COMPLETE ({elapsed:.0f}s)")
    print(f"{'='*60}")
    print(f"\nKEY FINDINGS:")
    print(f"  - {n_sig}/{emb.shape[1]} dimensions are statistically significant")
    print(f"  - 80% discriminability from top {n_80} dims (of 1024)")
    print(f"  - 90% discriminability from top {n_90} dims")
    print(f"  - 95% discriminability from top {n_95} dims")
    print(f"  - {len(consensus)} dims are top-50 in ALL 3 methods (Fisher, LR, t-test)")
    print(f"  - Ablation: top {ablation['k'][2]} Fisher dims = {ablation['top_fisher'][2]:.4f} accuracy")
    print(f"  - Ablation: all 1024 dims = {ablation['top_fisher'][-1]:.4f} accuracy")
    print(f"  - Bottom {ablation['k'][2]} dims = {ablation['bottom_fisher'][2]:.4f} accuracy")

    # Save dimension importance to npz
    importance_path = os.path.join(BASE_DIR, "embeddings", "dimension_importance.npz")
    np.savez_compressed(
        importance_path,
        fisher_scores=fisher,
        fisher_order=fisher_order,
        lr_coefficients=coefs,
        lr_order=coef_order,
        t_statistics=t_stats,
        p_values=p_values,
        consensus_dims=np.array(sorted(consensus)),
        any_two_dims=np.array(sorted(any_two)),
    )
    print(f"\n  Saved dimension importance to: {importance_path}")


if __name__ == "__main__":
    main()
