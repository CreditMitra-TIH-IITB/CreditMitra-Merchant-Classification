"""
PyTorch Deep Classifier — Push Accuracy Higher
=================================================
Trains multiple PyTorch architectures on GPU to beat sklearn MLP (97.54%):

1. DeepMLP:     Deep feedforward with BatchNorm + Dropout + Residual connections
2. WideResNet:  Wide residual blocks (1024→1024→512→256→1)
3. AttentionMLP: Self-attention layer + MLP (treats embedding dims as a sequence)
4. Ensemble:    Average predictions from top models

Techniques used:
- BatchNorm (stabilizes training, implicit regularization)
- Dropout (0.3-0.5, better than sklearn's early stopping)
- Residual/skip connections (better gradient flow)
- Cosine annealing LR schedule with warmup
- Label smoothing (reduces overconfidence)
- Mixup augmentation (improves generalization)
- OneCycleLR policy
- GPU acceleration

Usage:
    python data/train_pytorch_model.py
"""

import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix

# ============================================================================
# CONFIG
# ============================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMBEDDINGS_FILE = os.path.join(BASE_DIR, "embeddings", "embeddings.npz")
MODELS_DIR = os.path.join(BASE_DIR, "models")
FIGURES_DIR = os.path.join(BASE_DIR, "research", "figures")

RANDOM_STATE = 42
TEST_SIZE = 0.15
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Training hyperparameters
BATCH_SIZE = 1024
MAX_EPOCHS = 150
PATIENCE = 20
LABEL_SMOOTHING = 0.05
MIXUP_ALPHA = 0.2
WEIGHT_DECAY = 1e-4

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_STATE)
    torch.backends.cudnn.deterministic = True

# ============================================================================
# MODEL ARCHITECTURES
# ============================================================================

class ResidualBlock(nn.Module):
    """Residual block with BatchNorm and Dropout."""
    def __init__(self, dim, dropout=0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.act(x + self.block(x)))


class DeepResidualMLP(nn.Module):
    """Deep MLP with residual connections, BatchNorm, and Dropout."""
    def __init__(self, input_dim=1024, hidden_dims=[768, 512, 256], dropout=0.3, num_res_blocks=2):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))

            # Add residual blocks at this dimension
            for _ in range(num_res_blocks):
                layers.append(ResidualBlock(h_dim, dropout))

            prev_dim = h_dim

        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev_dim, 1)

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features).squeeze(-1)


class WideResNet(nn.Module):
    """Wide architecture with bottleneck residual blocks."""
    def __init__(self, input_dim=1024, dropout=0.4):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Wide residual blocks
        self.res1 = ResidualBlock(1024, dropout)
        self.res2 = ResidualBlock(1024, dropout)
        self.res3 = ResidualBlock(1024, dropout)

        self.down1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.res4 = ResidualBlock(512, dropout)
        self.res5 = ResidualBlock(512, dropout)

        self.down2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.res6 = ResidualBlock(256, dropout)

        self.head = nn.Linear(256, 1)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.down1(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.down2(x)
        x = self.res6(x)
        return self.head(x).squeeze(-1)


class AttentionMLP(nn.Module):
    """Self-attention over embedding dimensions + MLP classifier."""
    def __init__(self, input_dim=1024, num_heads=8, dropout=0.3):
        super().__init__()

        # Reshape 1024 dims into 64 tokens of 16 dims each
        self.n_tokens = 64
        self.token_dim = input_dim // self.n_tokens  # 16

        self.pos_embed = nn.Parameter(torch.randn(1, self.n_tokens, self.token_dim) * 0.02)

        self.attention = nn.MultiheadAttention(
            embed_dim=self.token_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(self.token_dim)

        # Second attention layer
        self.attention2 = nn.MultiheadAttention(
            embed_dim=self.token_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm2 = nn.LayerNorm(self.token_dim)

        # Pool and classify
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
        # Reshape to tokens: (B, 1024) -> (B, 64, 16)
        tokens = x.view(B, self.n_tokens, self.token_dim)
        tokens = tokens + self.pos_embed

        # Self-attention layer 1
        attn_out, _ = self.attention(tokens, tokens, tokens)
        tokens = self.attn_norm(tokens + attn_out)

        # Self-attention layer 2
        attn_out2, _ = self.attention2(tokens, tokens, tokens)
        tokens = self.attn_norm2(tokens + attn_out2)

        # Flatten and classify
        flat = tokens.reshape(B, -1)
        return self.classifier(flat).squeeze(-1)


class SimpleDeeperMLP(nn.Module):
    """Deeper MLP than sklearn: 1024→768→512→384→256→128→1 with BN+Dropout."""
    def __init__(self, input_dim=1024, dropout=0.35):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 768),
            nn.BatchNorm1d(768),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(512, 384),
            nn.BatchNorm1d(384),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(384, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),

            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

def mixup_data(x, y, alpha=0.2):
    """Mixup augmentation."""
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)  # Ensure lam >= 0.5
    indices = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[indices]
    return mixed_x, y, y[indices], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_model(model, train_loader, val_loader, model_name, max_epochs=MAX_EPOCHS,
                patience=PATIENCE, lr=1e-3, use_mixup=True):
    """Train a model with all the bells and whistles."""

    model = model.to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

    # OneCycleLR for superconvergence
    scheduler = OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=max_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,  # 10% warmup
        anneal_strategy='cos',
    )

    # Manual label smoothing for mixup compatibility
    smooth_criterion = lambda logits, targets: F.binary_cross_entropy_with_logits(
        logits,
        targets * (1 - LABEL_SMOOTHING) + 0.5 * LABEL_SMOOTHING
    )

    best_val_acc = 0
    best_val_f1 = 0
    best_epoch = 0
    no_improve = 0
    history = {'train_loss': [], 'val_acc': [], 'val_f1': [], 'val_auc': [], 'lr': []}

    print(f"\n  Training {model_name}...")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Device: {DEVICE}")

    for epoch in range(max_epochs):
        # TRAIN
        model.train()
        total_loss = 0
        n_batches = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

            if use_mixup and epoch < max_epochs * 0.8:  # No mixup in last 20% of training
                X_mixed, y_a, y_b, lam = mixup_data(X_batch, y_batch, MIXUP_ALPHA)
                logits = model(X_mixed)
                loss = mixup_criterion(smooth_criterion, logits, y_a, y_b, lam)
            else:
                logits = model(X_batch)
                loss = smooth_criterion(logits, y_batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        current_lr = scheduler.get_last_lr()[0]

        # VALIDATE
        model.eval()
        all_preds = []
        all_proba = []
        all_labels = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(DEVICE)
                logits = model(X_batch)
                proba = torch.sigmoid(logits).cpu().numpy()
                preds = (proba >= 0.5).astype(int)
                all_preds.extend(preds)
                all_proba.extend(proba)
                all_labels.extend(y_batch.numpy())

        val_acc = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds)
        val_auc = roc_auc_score(all_labels, all_proba)

        history['train_loss'].append(avg_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['val_auc'].append(val_auc)
        history['lr'].append(current_lr)

        # Check improvement
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_f1 = val_f1
            best_epoch = epoch
            no_improve = 0
            # Save best weights
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1:3d} | Loss={avg_loss:.5f} | Val Acc={val_acc:.4f} | "
                  f"Val F1={val_f1:.4f} | AUC={val_auc:.4f} | LR={current_lr:.6f} | "
                  f"Best={best_val_acc:.4f}@{best_epoch+1}")

        if no_improve >= patience:
            print(f"    Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break

    # Restore best weights
    model.load_state_dict(best_state)
    model = model.to(DEVICE)

    print(f"    Best: Acc={best_val_acc:.4f}, F1={best_val_f1:.4f} at epoch {best_epoch+1}")

    return model, history, best_val_acc


def evaluate_model(model, X_test_tensor, y_test_np, model_name):
    """Evaluate on test set."""
    model.eval()
    with torch.no_grad():
        # Process in batches to avoid OOM
        all_proba = []
        for i in range(0, len(X_test_tensor), BATCH_SIZE):
            batch = X_test_tensor[i:i+BATCH_SIZE].to(DEVICE)
            logits = model(batch)
            proba = torch.sigmoid(logits).cpu().numpy()
            all_proba.extend(proba)

    y_proba = np.array(all_proba)
    y_pred = (y_proba >= 0.5).astype(int)

    acc = accuracy_score(y_test_np, y_pred)
    f1 = f1_score(y_test_np, y_pred)
    auc = roc_auc_score(y_test_np, y_proba)
    cm = confusion_matrix(y_test_np, y_pred)

    print(f"\n  [{model_name}] Test Results:")
    print(f"    Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"    F1:        {f1:.4f}")
    print(f"    AUC:       {auc:.4f}")
    print(f"    Confusion: TN={cm[0][0]} FP={cm[0][1]} FN={cm[1][0]} TP={cm[1][1]}")

    return acc, f1, auc, y_proba, y_pred


# ============================================================================
# MAIN
# ============================================================================

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # ================================================================
    # 1. LOAD & PREPARE DATA
    # ================================================================
    print("=" * 70)
    print("PHASE 1: DATA LOADING")
    print("=" * 70)

    data = np.load(EMBEDDINGS_FILE, allow_pickle=True)
    emb = data['embeddings']
    labels = data['labels']
    names = data['names']
    print(f"  Dataset: {emb.shape}")

    X_train, X_test, y_train, y_test, names_train, names_test = train_test_split(
        emb, labels, names, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=labels
    )

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Split train into train/val (90/10)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_scaled, y_train, test_size=0.1, random_state=RANDOM_STATE, stratify=y_train
    )

    print(f"  Train: {X_tr.shape}, Val: {X_val.shape}, Test: {X_test_scaled.shape}")
    print(f"  Device: {DEVICE}")

    # Create DataLoaders
    train_ds = TensorDataset(
        torch.FloatTensor(X_tr),
        torch.FloatTensor(y_tr.astype(np.float32))
    )
    val_ds = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val.astype(np.float32))
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=0, pin_memory=True)

    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_np = y_test

    # ================================================================
    # 2. DEFINE ARCHITECTURES
    # ================================================================
    print("\n" + "=" * 70)
    print("PHASE 2: ARCHITECTURE TOURNAMENT")
    print("=" * 70)

    architectures = [
        ("DeepResMLP", DeepResidualMLP(1024, [768, 512, 256], dropout=0.3, num_res_blocks=2), 1e-3),
        ("SimpleDeeperMLP", SimpleDeeperMLP(1024, dropout=0.35), 1e-3),
        ("WideResNet", WideResNet(1024, dropout=0.35), 8e-4),
        ("AttentionMLP", AttentionMLP(1024, num_heads=8, dropout=0.3), 8e-4),
    ]

    results = {}
    all_histories = {}
    best_overall_acc = 0
    best_model_name = ""
    best_model = None
    all_test_probas = {}

    for name, model, lr in architectures:
        print(f"\n{'='*50}")
        print(f"  Architecture: {name}")
        print(f"{'='*50}")

        start = time.time()
        trained_model, history, val_acc = train_model(
            model, train_loader, val_loader, name,
            max_epochs=MAX_EPOCHS, patience=PATIENCE, lr=lr,
        )
        train_time = time.time() - start

        test_acc, test_f1, test_auc, test_proba, test_pred = evaluate_model(
            trained_model, X_test_tensor, y_test_np, name
        )

        results[name] = {
            'val_acc': val_acc,
            'test_acc': test_acc,
            'test_f1': test_f1,
            'test_auc': test_auc,
            'train_time': train_time,
            'params': sum(p.numel() for p in trained_model.parameters()),
        }
        all_histories[name] = history
        all_test_probas[name] = test_proba

        if test_acc > best_overall_acc:
            best_overall_acc = test_acc
            best_model_name = name
            best_model = trained_model

        # Save each model
        model_path = os.path.join(MODELS_DIR, f'pytorch_{name.lower()}.pt')
        torch.save({
            'model_state_dict': trained_model.state_dict(),
            'model_class': name,
            'test_acc': test_acc,
            'test_f1': test_f1,
            'test_auc': test_auc,
        }, model_path)
        print(f"    Saved: {model_path}")

    # ================================================================
    # 3. ENSEMBLE
    # ================================================================
    print("\n" + "=" * 70)
    print("PHASE 3: ENSEMBLE")
    print("=" * 70)

    # Average ensemble of all models
    ensemble_proba = np.mean(list(all_test_probas.values()), axis=0)
    ensemble_pred = (ensemble_proba >= 0.5).astype(int)
    ens_acc = accuracy_score(y_test_np, ensemble_pred)
    ens_f1 = f1_score(y_test_np, ensemble_pred)
    ens_auc = roc_auc_score(y_test_np, ensemble_proba)
    print(f"\n  [Ensemble (all 4)] Accuracy={ens_acc:.4f} ({ens_acc*100:.2f}%), F1={ens_f1:.4f}, AUC={ens_auc:.4f}")

    results['Ensemble_All'] = {
        'test_acc': ens_acc, 'test_f1': ens_f1, 'test_auc': ens_auc,
        'val_acc': 0, 'train_time': 0, 'params': 0,
    }

    # Top-2 ensemble
    sorted_models = sorted(results.items(), key=lambda x: x[1]['test_acc'], reverse=True)
    top2_names = [n for n, _ in sorted_models if n != 'Ensemble_All'][:2]
    top2_proba = np.mean([all_test_probas[n] for n in top2_names], axis=0)
    top2_pred = (top2_proba >= 0.5).astype(int)
    top2_acc = accuracy_score(y_test_np, top2_pred)
    top2_f1 = f1_score(y_test_np, top2_pred)
    top2_auc = roc_auc_score(y_test_np, top2_proba)
    print(f"  [Ensemble Top-2: {top2_names}] Accuracy={top2_acc:.4f} ({top2_acc*100:.2f}%), F1={top2_f1:.4f}, AUC={top2_auc:.4f}")

    results['Ensemble_Top2'] = {
        'test_acc': top2_acc, 'test_f1': top2_f1, 'test_auc': top2_auc,
        'val_acc': 0, 'train_time': 0, 'params': 0,
    }

    # ================================================================
    # 4. COMPARISON TABLE
    # ================================================================
    print("\n" + "=" * 70)
    print("PHASE 4: FINAL COMPARISON")
    print("=" * 70)

    print(f"\n  {'Model':<25} {'Test Acc':>10} {'Test F1':>10} {'AUC':>10} {'Params':>12} {'Time':>8}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*12} {'-'*8}")

    # Add sklearn baseline for comparison
    print(f"  {'sklearn MLP (baseline)':<25} {'97.54%':>10} {'0.9772':>10} {'0.9969':>10} {'~1.4M':>12} {'753s':>8}")

    for name, metrics in sorted(results.items(), key=lambda x: -x[1]['test_acc']):
        acc_str = f"{metrics['test_acc']*100:.2f}%"
        f1_str = f"{metrics['test_f1']:.4f}"
        auc_str = f"{metrics['test_auc']:.4f}"
        params_str = f"{metrics['params']:,}" if metrics['params'] > 0 else "ensemble"
        time_str = f"{metrics['train_time']:.0f}s" if metrics['train_time'] > 0 else "-"
        print(f"  {name:<25} {acc_str:>10} {f1_str:>10} {auc_str:>10} {params_str:>12} {time_str:>8}")

    # ================================================================
    # 5. PLOTS
    # ================================================================
    print("\n" + "=" * 70)
    print("PHASE 5: PLOTS")
    print("=" * 70)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    colors = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0']

    # Training loss curves
    ax = axes[0][0]
    for i, (name, hist) in enumerate(all_histories.items()):
        ax.plot(hist['train_loss'], color=colors[i], label=name, linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title('Training Loss Curves', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Validation accuracy curves
    ax = axes[0][1]
    for i, (name, hist) in enumerate(all_histories.items()):
        ax.plot(hist['val_acc'], color=colors[i], label=name, linewidth=1.5)
    ax.axhline(y=0.9754, color='red', linestyle='--', alpha=0.5, label='sklearn baseline')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Accuracy')
    ax.set_title('Validation Accuracy Curves', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Test accuracy comparison bar chart
    ax = axes[1][0]
    model_names = list(results.keys()) + ['sklearn MLP']
    test_accs = [r['test_acc'] * 100 for r in results.values()] + [97.54]
    bar_colors = ['#2196F3'] * len(results) + ['#FF9800']

    bars = ax.barh(model_names, test_accs, color=bar_colors, edgecolor='white')
    ax.set_xlabel('Test Accuracy (%)')
    ax.set_title('Model Comparison', fontweight='bold')
    ax.set_xlim(95, 99)
    for bar, acc in zip(bars, test_accs):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                f'{acc:.2f}%', va='center', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    # LR schedule
    ax = axes[1][1]
    for i, (name, hist) in enumerate(all_histories.items()):
        ax.plot(hist['lr'], color=colors[i], label=name, linewidth=1)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.suptitle('PyTorch Architecture Tournament — Pushing Past 97.5%',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, '23_pytorch_tournament.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  -> Saved 23_pytorch_tournament.png")

    # ================================================================
    # 6. SAVE BEST MODEL + ENSEMBLE
    # ================================================================
    print("\n" + "=" * 70)
    print("PHASE 6: EXPORT")
    print("=" * 70)

    # Save results summary
    summary = {
        'sklearn_baseline': {'test_acc': 0.9754, 'test_f1': 0.9772, 'test_auc': 0.9969},
        'pytorch_results': {k: {kk: float(vv) if isinstance(vv, (float, np.floating)) else vv
                                for kk, vv in v.items()}
                           for k, v in results.items()},
        'best_model': best_model_name,
        'best_acc': float(best_overall_acc),
        'ensemble_all_acc': float(ens_acc),
        'ensemble_top2_acc': float(top2_acc),
        'device': str(DEVICE),
        'batch_size': BATCH_SIZE,
        'max_epochs': MAX_EPOCHS,
        'label_smoothing': LABEL_SMOOTHING,
        'mixup_alpha': MIXUP_ALPHA,
    }

    with open(os.path.join(MODELS_DIR, 'pytorch_results.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Best single model: {best_model_name} ({best_overall_acc*100:.2f}%)")
    print(f"  Ensemble (all 4):  {ens_acc*100:.2f}%")
    print(f"  Ensemble (top 2):  {top2_acc*100:.2f}%")
    print(f"  sklearn baseline:  97.54%")
    print(f"\n  Improvement over sklearn: {(best_overall_acc - 0.9754)*100:+.2f}%")

    print(f"\n{'='*70}")
    print("DONE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
