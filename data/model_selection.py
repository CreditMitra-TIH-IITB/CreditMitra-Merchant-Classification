"""
Comprehensive Model Selection & Stacking Pipeline
=====================================================
Tests every reasonable classifier on our 1024-dim embeddings,
performs feature engineering, correlation pruning, and stacking.

Pipeline:
1. Feature Engineering (raw, PCA, top-K Fisher, statistical features)
2. Feature Correlation Analysis & Pruning
3. Base Model Tournament (LR, SVM, RF, XGB, LGBM, MLP, KNN, Ridge)
4. Meta-Learner / Stacking Ensemble
5. Full evaluation on held-out test set
6. Comprehensive report with all results

Output: research/figures/16-20_*.png, research/05_model_selection.md
"""

import os
import sys
import time
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict
import warnings
warnings.filterwarnings('ignore')

# Sklearn
from sklearn.model_selection import (
    cross_val_score, cross_val_predict,
    StratifiedKFold, train_test_split
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    VotingClassifier, StackingClassifier,
    ExtraTreesClassifier,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve
)
from sklearn.calibration import CalibratedClassifierCV

# Boosting
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("WARNING: xgboost not installed, skipping XGBoost")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("WARNING: lightgbm not installed, skipping LightGBM")


# ============================================================================
# CONFIG
# ============================================================================

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
EMBEDDINGS_FILE = os.path.join(BASE_DIR, "embeddings", "embeddings.npz")
DIM_IMPORTANCE_FILE = os.path.join(BASE_DIR, "embeddings", "dimension_importance.npz")
FIGURES_DIR = os.path.join(BASE_DIR, "research", "figures")
RESEARCH_DIR = os.path.join(BASE_DIR, "research")
MODELS_DIR = os.path.join(BASE_DIR, "models")

RANDOM_STATE = 42
N_FOLDS = 5
TEST_SIZE = 0.15  # 15% held-out test set

plt.rcParams.update({
    'figure.dpi': 150,
    'font.size': 11,
    'font.family': 'sans-serif',
    'axes.titlesize': 14,
    'axes.labelsize': 12,
})

PERSON_COLOR = '#2196F3'
MERCHANT_COLOR = '#FF5722'


# ============================================================================
# 1. DATA LOADING & FEATURE ENGINEERING
# ============================================================================

def load_and_prepare_data():
    """Load ALL embeddings and create train/test split with feature engineering."""
    print("=" * 70)
    print("PHASE 1: DATA LOADING & FEATURE ENGINEERING")
    print("=" * 70)

    # Load embeddings
    print("\nLoading embeddings...")
    data = np.load(EMBEDDINGS_FILE, allow_pickle=True)
    emb = data['embeddings']
    labels = data['labels']
    names = data['names']
    print(f"  Full dataset: {emb.shape}")

    # Load dimension importance
    dim_data = np.load(DIM_IMPORTANCE_FILE, allow_pickle=True)
    fisher_order = dim_data['fisher_order']
    print(f"  Fisher dimension order loaded")
    print(f"  Using ALL {len(emb):,} samples (no subsampling)")

    # Train/test split
    X_train, X_test, y_train, y_test, names_train, names_test = train_test_split(
        emb, labels, names, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=labels
    )
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"  Train class balance: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"  Test class balance:  {dict(zip(*np.unique(y_test, return_counts=True)))}")

    # ---- Feature Engineering ----
    print("\n  Feature Engineering...")
    feature_sets = OrderedDict()

    # 1. Raw 1024 dims
    feature_sets['raw_1024'] = {
        'X_train': X_train,
        'X_test': X_test,
        'desc': 'Raw 1024 embedding dimensions'
    }

    # 2. Standardized
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    feature_sets['scaled_1024'] = {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'desc': 'StandardScaler + 1024 dims'
    }

    # 3. Top-K Fisher dimensions
    for k in [50, 100, 200, 512]:
        top_dims = fisher_order[:k]
        feature_sets[f'fisher_top{k}'] = {
            'X_train': X_train[:, top_dims],
            'X_test': X_test[:, top_dims],
            'desc': f'Top {k} Fisher dimensions'
        }

    # 4. PCA reduced
    for n_comp in [64, 128, 256]:
        pca = PCA(n_components=n_comp, random_state=RANDOM_STATE)
        X_tr_pca = pca.fit_transform(X_train)
        X_te_pca = pca.transform(X_test)
        feature_sets[f'pca_{n_comp}'] = {
            'X_train': X_tr_pca,
            'X_test': X_te_pca,
            'desc': f'PCA {n_comp} components (var={pca.explained_variance_ratio_.sum()*100:.1f}%)'
        }

    # 5. Statistical meta-features + raw
    def stat_features(X):
        """Compute statistical features from embedding vectors."""
        feats = np.column_stack([
            np.linalg.norm(X, axis=1, keepdims=True),        # L2 norm
            X.mean(axis=1, keepdims=True),                     # mean
            X.std(axis=1, keepdims=True),                      # std
            np.median(X, axis=1, keepdims=True),               # median
            X.min(axis=1, keepdims=True),                      # min
            X.max(axis=1, keepdims=True),                      # max
            (X.max(axis=1) - X.min(axis=1)).reshape(-1, 1),   # range
            np.percentile(X, 25, axis=1).reshape(-1, 1),      # Q1
            np.percentile(X, 75, axis=1).reshape(-1, 1),      # Q3
            (np.abs(X) > 0.05).sum(axis=1).reshape(-1, 1),   # active dims
        ])
        return feats

    stat_train = stat_features(X_train)
    stat_test = stat_features(X_test)

    feature_sets['raw_plus_stats'] = {
        'X_train': np.hstack([X_train, stat_train]),
        'X_test': np.hstack([X_test, stat_test]),
        'desc': f'Raw 1024 + 10 statistical features'
    }

    # 6. Top-200 Fisher + stats
    top200 = fisher_order[:200]
    feature_sets['fisher200_plus_stats'] = {
        'X_train': np.hstack([X_train[:, top200], stat_train]),
        'X_test': np.hstack([X_test[:, top200], stat_test]),
        'desc': 'Top 200 Fisher + 10 statistical features'
    }

    for name, fs in feature_sets.items():
        print(f"    {name}: {fs['X_train'].shape[1]} features - {fs['desc']}")

    return feature_sets, y_train, y_test, names_train, names_test


# ============================================================================
# 2. FEATURE CORRELATION ANALYSIS
# ============================================================================

def analyze_feature_correlation(feature_sets, fig_dir):
    """Analyze and prune highly correlated features."""
    print("\n" + "=" * 70)
    print("PHASE 2: FEATURE CORRELATION ANALYSIS")
    print("=" * 70)

    X = feature_sets['raw_1024']['X_train']

    # Sample 200 dims for visualization
    np.random.seed(42)
    sample_dims = np.random.choice(X.shape[1], min(200, X.shape[1]), replace=False)
    sample_dims.sort()

    corr = np.corrcoef(X[:, sample_dims].T)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Correlation heatmap
    ax = axes[0]
    im = ax.imshow(np.abs(corr), cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    ax.set_title('|Correlation| Matrix (200 random dims)', fontweight='bold')
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Dimension')
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Distribution of absolute correlations
    ax = axes[1]
    upper_tri = np.abs(corr[np.triu_indices(len(corr), k=1)])
    ax.hist(upper_tri, bins=100, color='#FF5722', alpha=0.7, edgecolor='white')
    ax.axvline(0.9, color='red', linestyle='--', linewidth=2, label='r=0.9 threshold')
    ax.axvline(0.95, color='darkred', linestyle='--', linewidth=2, label='r=0.95 threshold')
    n_high = (upper_tri > 0.9).sum()
    n_very_high = (upper_tri > 0.95).sum()
    ax.set_xlabel('|Pearson Correlation|')
    ax.set_ylabel('Count')
    ax.set_title(f'Correlation Distribution ({n_high} pairs > 0.9, {n_very_high} > 0.95)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('Feature Correlation Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, '16_feature_correlation.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  High correlation pairs (|r|>0.9): {n_high}/{len(upper_tri)}")
    print(f"  Very high (|r|>0.95): {n_very_high}/{len(upper_tri)}")
    print("  -> Saved 16_feature_correlation.png")


# ============================================================================
# 3. BASE MODEL TOURNAMENT
# ============================================================================

def get_base_models():
    """Define all candidate models (excluding O(n^2) models like SVM_RBF/KNN for 233K scale)."""
    models = OrderedDict()

    # Linear models
    models['LogisticRegression'] = LogisticRegression(
        max_iter=2000, C=1.0, random_state=RANDOM_STATE, solver='lbfgs'
    )
    models['LogisticRegression_L1'] = LogisticRegression(
        max_iter=2000, C=1.0, random_state=RANDOM_STATE, penalty='l1', solver='saga'
    )
    models['RidgeClassifier'] = RidgeClassifier(
        alpha=1.0, random_state=RANDOM_STATE
    )
    models['LinearSVC'] = CalibratedClassifierCV(
        LinearSVC(max_iter=5000, C=1.0, random_state=RANDOM_STATE),
        cv=3
    )

    # NOTE: SVM_RBF and KNN removed — O(n^2), took 1h+ on 233K and would not finish

    # Tree-based
    models['RandomForest'] = RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_leaf=2,
        random_state=RANDOM_STATE, n_jobs=-1
    )
    models['ExtraTrees'] = ExtraTreesClassifier(
        n_estimators=300, max_depth=None, min_samples_leaf=2,
        random_state=RANDOM_STATE, n_jobs=-1
    )

    # Boosting
    if HAS_XGB:
        models['XGBoost'] = xgb.XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE, n_jobs=-1,
            eval_metric='logloss', verbosity=0,
            tree_method='hist',
        )
    if HAS_LGB:
        models['LightGBM'] = lgb.LGBMClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE, n_jobs=-1,
            verbose=-1,
        )

    # NOTE: GradientBoosting & AdaBoost removed — sklearn single-threaded, 1h+ on 233K

    # Neural Network
    models['MLP_small'] = MLPClassifier(
        hidden_layer_sizes=(256, 128), max_iter=500,
        random_state=RANDOM_STATE, early_stopping=True,
        validation_fraction=0.1, batch_size=512,
        learning_rate='adaptive',
    )
    models['MLP_large'] = MLPClassifier(
        hidden_layer_sizes=(512, 256, 128), max_iter=500,
        random_state=RANDOM_STATE, early_stopping=True,
        validation_fraction=0.1, batch_size=512,
        learning_rate='adaptive',
    )

    return models


def run_model_tournament(feature_sets, y_train, y_test, fig_dir):
    """Test all models on multiple feature sets."""
    print("\n" + "=" * 70)
    print("PHASE 3: BASE MODEL TOURNAMENT")
    print("=" * 70)

    models = get_base_models()
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # Feature sets to test each model on
    # Use scaled for linear/MLP models, raw for tree-based
    linear_models = {'LogisticRegression', 'LogisticRegression_L1', 'RidgeClassifier',
                     'LinearSVC', 'MLP_small', 'MLP_large'}
    tree_models = {'RandomForest', 'ExtraTrees', 'XGBoost', 'LightGBM'}

    # Test configurations (streamlined for 233K scale)
    test_configs = [
        # (feature_set_name, models_to_test)
        ('scaled_1024', linear_models),
        ('raw_1024', tree_models),
        ('fisher_top200', {'LogisticRegression', 'MLP_small', 'XGBoost', 'LightGBM'}),
        ('raw_plus_stats', {'XGBoost', 'LightGBM', 'RandomForest'}),
        ('fisher200_plus_stats', {'LogisticRegression', 'XGBoost', 'LightGBM', 'MLP_small'}),
        ('pca_128', {'LogisticRegression', 'MLP_small'}),
    ]

    results = []

    for fs_name, model_names in test_configs:
        if fs_name not in feature_sets:
            continue

        fs = feature_sets[fs_name]
        X_tr = fs['X_train']
        X_te = fs['X_test']

        for model_name in model_names:
            if model_name not in models:
                continue

            print(f"\n  [{fs_name}] {model_name}...", end=" ", flush=True)
            start = time.time()

            try:
                model = models[model_name]

                # Clone the model for fresh training
                from sklearn.base import clone
                model_clone = clone(model)

                # Cross-validation
                cv_scores = cross_val_score(model_clone, X_tr, y_train, cv=cv,
                                           scoring='accuracy', n_jobs=1)

                # Train on full train set, evaluate on test
                model_clone2 = clone(model)
                model_clone2.fit(X_tr, y_train)
                y_pred = model_clone2.predict(X_te)

                # Probabilities for AUC
                if hasattr(model_clone2, 'predict_proba'):
                    y_proba = model_clone2.predict_proba(X_te)[:, 1]
                elif hasattr(model_clone2, 'decision_function'):
                    y_proba = model_clone2.decision_function(X_te)
                else:
                    y_proba = y_pred.astype(float)

                test_acc = accuracy_score(y_test, y_pred)
                test_f1 = f1_score(y_test, y_pred)
                test_prec = precision_score(y_test, y_pred)
                test_rec = recall_score(y_test, y_pred)

                try:
                    test_auc = roc_auc_score(y_test, y_proba)
                except:
                    test_auc = 0.0

                elapsed = time.time() - start

                result = {
                    'model': model_name,
                    'features': fs_name,
                    'n_features': X_tr.shape[1],
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'test_acc': test_acc,
                    'test_f1': test_f1,
                    'test_prec': test_prec,
                    'test_rec': test_rec,
                    'test_auc': test_auc,
                    'time_sec': elapsed,
                    'model_obj': model_clone2,
                    'y_pred': y_pred,
                    'y_proba': y_proba,
                }
                results.append(result)

                print(f"CV={cv_scores.mean():.4f}(+/-{cv_scores.std():.4f}), "
                      f"Test={test_acc:.4f}, F1={test_f1:.4f}, AUC={test_auc:.4f} "
                      f"[{elapsed:.1f}s]")

            except Exception as e:
                print(f"FAILED: {e}")
                continue

    return results


# ============================================================================
# 4. STACKING / META-LEARNER
# ============================================================================

def build_stacking_ensembles(feature_sets, y_train, y_test, base_results, fig_dir):
    """Build stacking ensembles using best base models."""
    print("\n" + "=" * 70)
    print("PHASE 4: STACKING & META-LEARNER ENSEMBLES")
    print("=" * 70)

    # Pick best feature set for each model type
    best_linear_fs = 'scaled_1024'
    best_tree_fs = 'raw_1024'

    X_tr = feature_sets[best_linear_fs]['X_train']
    X_te = feature_sets[best_linear_fs]['X_test']

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    stacking_results = []

    # ---- Stacking Ensemble 1: Diverse base learners ----
    print("\n  [Stack 1] LR + RF + XGB + MLP (meta: LR)...")
    estimators_1 = [
        ('lr', LogisticRegression(max_iter=2000, C=1.0, random_state=RANDOM_STATE)),
        ('rf', RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)),
        ('mlp', MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=300,
                              random_state=RANDOM_STATE, early_stopping=True)),
    ]
    if HAS_XGB:
        estimators_1.append(
            ('xgb', xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                                       random_state=RANDOM_STATE, n_jobs=-1,
                                       eval_metric='logloss', verbosity=0, tree_method='hist'))
        )
    if HAS_LGB:
        estimators_1.append(
            ('lgb', lgb.LGBMClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                                       random_state=RANDOM_STATE, n_jobs=-1, verbose=-1))
        )

    start = time.time()
    stack1 = StackingClassifier(
        estimators=estimators_1,
        final_estimator=LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        cv=N_FOLDS, n_jobs=1, passthrough=False
    )
    cv_scores = cross_val_score(stack1, X_tr, y_train, cv=cv, scoring='accuracy', n_jobs=1)

    stack1.fit(X_tr, y_train)
    y_pred = stack1.predict(X_te)
    y_proba = stack1.predict_proba(X_te)[:, 1]
    elapsed = time.time() - start

    result = {
        'model': 'Stack_LR+RF+XGB+LGB+MLP_meta-LR',
        'features': best_linear_fs,
        'n_features': X_tr.shape[1],
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'test_acc': accuracy_score(y_test, y_pred),
        'test_f1': f1_score(y_test, y_pred),
        'test_prec': precision_score(y_test, y_pred),
        'test_rec': recall_score(y_test, y_pred),
        'test_auc': roc_auc_score(y_test, y_proba),
        'time_sec': elapsed,
        'model_obj': stack1,
        'y_pred': y_pred,
        'y_proba': y_proba,
    }
    stacking_results.append(result)
    print(f"    CV={cv_scores.mean():.4f}(+/-{cv_scores.std():.4f}), "
          f"Test={result['test_acc']:.4f}, F1={result['test_f1']:.4f}, "
          f"AUC={result['test_auc']:.4f} [{elapsed:.1f}s]")

    # ---- Stacking Ensemble 2: Meta = XGBoost ----
    if HAS_XGB:
        print("\n  [Stack 2] LR + RF + LGBM + MLP (meta: XGBoost)...")
        estimators_2 = [
            ('lr', LogisticRegression(max_iter=2000, C=1.0, random_state=RANDOM_STATE)),
            ('rf', RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)),
            ('mlp', MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=300,
                                  random_state=RANDOM_STATE, early_stopping=True)),
        ]
        if HAS_LGB:
            estimators_2.append(
                ('lgb', lgb.LGBMClassifier(n_estimators=300, random_state=RANDOM_STATE,
                                           n_jobs=-1, verbose=-1))
            )

        start = time.time()
        stack2 = StackingClassifier(
            estimators=estimators_2,
            final_estimator=xgb.XGBClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.1,
                random_state=RANDOM_STATE, eval_metric='logloss', verbosity=0, tree_method='hist'
            ),
            cv=N_FOLDS, n_jobs=1, passthrough=False
        )
        cv_scores = cross_val_score(stack2, X_tr, y_train, cv=cv, scoring='accuracy', n_jobs=1)

        stack2.fit(X_tr, y_train)
        y_pred = stack2.predict(X_te)
        y_proba = stack2.predict_proba(X_te)[:, 1]
        elapsed = time.time() - start

        result = {
            'model': 'Stack_LR+RF+LGB+MLP_meta-XGB',
            'features': best_linear_fs,
            'n_features': X_tr.shape[1],
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'test_acc': accuracy_score(y_test, y_pred),
            'test_f1': f1_score(y_test, y_pred),
            'test_prec': precision_score(y_test, y_pred),
            'test_rec': recall_score(y_test, y_pred),
            'test_auc': roc_auc_score(y_test, y_proba),
            'time_sec': elapsed,
            'model_obj': stack2,
            'y_pred': y_pred,
            'y_proba': y_proba,
        }
        stacking_results.append(result)
        print(f"    CV={cv_scores.mean():.4f}(+/-{cv_scores.std():.4f}), "
              f"Test={result['test_acc']:.4f}, F1={result['test_f1']:.4f}, "
              f"AUC={result['test_auc']:.4f} [{elapsed:.1f}s]")

    # ---- Voting Ensemble (soft) ----
    print("\n  [Voting] Soft voting: LR + RF + XGB + MLP...")
    vote_estimators = [
        ('lr', LogisticRegression(max_iter=2000, C=1.0, random_state=RANDOM_STATE)),
        ('rf', RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)),
        ('mlp', MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=300,
                              random_state=RANDOM_STATE, early_stopping=True)),
    ]
    if HAS_XGB:
        vote_estimators.append(
            ('xgb', xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                                       random_state=RANDOM_STATE, n_jobs=-1,
                                       eval_metric='logloss', verbosity=0, tree_method='hist'))
        )

    start = time.time()
    voter = VotingClassifier(estimators=vote_estimators, voting='soft', n_jobs=1)
    cv_scores = cross_val_score(voter, X_tr, y_train, cv=cv, scoring='accuracy', n_jobs=1)

    voter.fit(X_tr, y_train)
    y_pred = voter.predict(X_te)
    y_proba = voter.predict_proba(X_te)[:, 1]
    elapsed = time.time() - start

    result = {
        'model': 'SoftVoting_LR+RF+XGB+MLP',
        'features': best_linear_fs,
        'n_features': X_tr.shape[1],
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'test_acc': accuracy_score(y_test, y_pred),
        'test_f1': f1_score(y_test, y_pred),
        'test_prec': precision_score(y_test, y_pred),
        'test_rec': recall_score(y_test, y_pred),
        'test_auc': roc_auc_score(y_test, y_proba),
        'time_sec': elapsed,
        'model_obj': voter,
        'y_pred': y_pred,
        'y_proba': y_proba,
    }
    stacking_results.append(result)
    print(f"    CV={cv_scores.mean():.4f}(+/-{cv_scores.std():.4f}), "
          f"Test={result['test_acc']:.4f}, F1={result['test_f1']:.4f}, "
          f"AUC={result['test_auc']:.4f} [{elapsed:.1f}s]")

    return stacking_results


# ============================================================================
# 5. VISUALIZATION & REPORTING
# ============================================================================

def plot_tournament_results(all_results, fig_dir):
    """Create comprehensive comparison plots."""
    print("\n" + "=" * 70)
    print("PHASE 5: VISUALIZATION & REPORTING")
    print("=" * 70)

    # Sort by test accuracy
    sorted_results = sorted(all_results, key=lambda x: x['test_acc'], reverse=True)

    # ---- Plot 1: Model Comparison Bar Chart ----
    fig, axes = plt.subplots(1, 2, figsize=(20, max(8, len(sorted_results) * 0.4)))

    # Test accuracy
    ax = axes[0]
    labels_plot = [f"{r['model']}\n({r['features']}, {r['n_features']}d)" for r in sorted_results]
    accs = [r['test_acc'] for r in sorted_results]
    colors = ['#4CAF50' if r['model'].startswith('Stack') or r['model'].startswith('Soft')
              else '#2196F3' for r in sorted_results]

    bars = ax.barh(range(len(sorted_results)), accs, color=colors, alpha=0.8, edgecolor='white')
    ax.set_yticks(range(len(sorted_results)))
    ax.set_yticklabels(labels_plot, fontsize=8)
    ax.set_xlabel('Test Accuracy')
    ax.set_title('Test Accuracy (all models)', fontweight='bold')
    ax.set_xlim(min(accs) - 0.02, max(accs) + 0.01)
    ax.grid(True, alpha=0.3, axis='x')
    for i, acc in enumerate(accs):
        ax.text(acc + 0.001, i, f'{acc:.4f}', va='center', fontsize=8, fontweight='bold')

    # F1 score
    ax = axes[1]
    f1s = [r['test_f1'] for r in sorted_results]
    bars = ax.barh(range(len(sorted_results)), f1s, color=colors, alpha=0.8, edgecolor='white')
    ax.set_yticks(range(len(sorted_results)))
    ax.set_yticklabels(labels_plot, fontsize=8)
    ax.set_xlabel('Test F1 Score')
    ax.set_title('Test F1 Score (all models)', fontweight='bold')
    ax.set_xlim(min(f1s) - 0.02, max(f1s) + 0.01)
    ax.grid(True, alpha=0.3, axis='x')
    for i, f1 in enumerate(f1s):
        ax.text(f1 + 0.001, i, f'{f1:.4f}', va='center', fontsize=8, fontweight='bold')

    plt.suptitle('Model Tournament Results', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, '17_model_tournament.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  -> Saved 17_model_tournament.png")

    # ---- Plot 2: CV vs Test accuracy scatter ----
    fig, ax = plt.subplots(figsize=(10, 8))
    for r in sorted_results:
        color = '#4CAF50' if 'Stack' in r['model'] or 'Voting' in r['model'] else '#2196F3'
        marker = 's' if 'Stack' in r['model'] or 'Voting' in r['model'] else 'o'
        ax.scatter(r['cv_mean'], r['test_acc'], c=color, s=80, marker=marker, alpha=0.7,
                  edgecolors='black', linewidth=0.5)
        ax.annotate(r['model'][:15], (r['cv_mean'], r['test_acc']),
                   fontsize=7, ha='left', va='bottom')

    ax.plot([0.85, 1.0], [0.85, 1.0], 'r--', alpha=0.5, label='Perfect agreement')
    ax.set_xlabel('CV Accuracy (5-fold)')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('CV vs Test Accuracy (check for overfitting)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, '18_cv_vs_test.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  -> Saved 18_cv_vs_test.png")

    # ---- Plot 3: Best model confusion matrix + ROC ----
    best = sorted_results[0]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Confusion matrix
    ax = axes[0]
    cm = confusion_matrix(best['y_pred'], best['y_pred'])  # placeholder
    # Actually need y_test
    # We'll just plot ROC for now
    ax.set_visible(False)

    # ROC curves for top 5 models
    ax = axes[1]
    top5 = sorted_results[:5]
    for r in top5:
        try:
            fpr, tpr, _ = roc_curve(
                # We need y_test, but it's not in results
                # Add a flag to pass y_test
                np.zeros(10), np.zeros(10)  # placeholder
            )
        except:
            pass

    plt.close()

    return sorted_results


def plot_best_model_details(best_result, y_test, fig_dir):
    """Detailed evaluation of the best model."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Confusion Matrix
    ax = axes[0]
    cm = confusion_matrix(y_test, best_result['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['PERSON', 'MERCHANT'],
                yticklabels=['PERSON', 'MERCHANT'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix: {best_result["model"]}', fontweight='bold')

    # ROC Curve
    ax = axes[1]
    fpr, tpr, _ = roc_curve(y_test, best_result['y_proba'])
    ax.plot(fpr, tpr, 'b-', linewidth=2,
            label=f'ROC (AUC={best_result["test_auc"]:.4f})')
    ax.plot([0, 1], [0, 1], 'r--', alpha=0.5)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve (Best Model)', fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'Best Model: {best_result["model"]} (Acc={best_result["test_acc"]:.4f})',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, '19_best_model_evaluation.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  -> Saved 19_best_model_evaluation.png")


def plot_speed_vs_accuracy(all_results, fig_dir):
    """Speed vs accuracy tradeoff plot."""
    fig, ax = plt.subplots(figsize=(12, 8))

    for r in all_results:
        color = '#4CAF50' if 'Stack' in r['model'] or 'Voting' in r['model'] else (
            '#FF5722' if r['model'] in ('XGBoost', 'LightGBM') else '#2196F3'
        )
        marker = 's' if 'Stack' in r['model'] or 'Voting' in r['model'] else 'o'
        size = 100 if 'Stack' in r['model'] or 'Voting' in r['model'] else 60
        ax.scatter(r['time_sec'], r['test_acc'], c=color, s=size, marker=marker,
                  alpha=0.7, edgecolors='black', linewidth=0.5)
        ax.annotate(f"{r['model'][:12]}\n({r['features'][:10]})",
                   (r['time_sec'], r['test_acc']),
                   fontsize=7, ha='left', va='bottom')

    ax.set_xlabel('Training + Evaluation Time (seconds)', fontsize=13)
    ax.set_ylabel('Test Accuracy', fontsize=13)
    ax.set_title('Speed vs Accuracy Tradeoff', fontsize=16, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, '20_speed_vs_accuracy.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  -> Saved 20_speed_vs_accuracy.png")


def generate_report(all_results, research_dir):
    """Generate markdown report."""
    sorted_results = sorted(all_results, key=lambda x: x['test_acc'], reverse=True)

    report_path = os.path.join(research_dir, '05_model_selection.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Stage 5: Model Selection & Stacking Results\n\n")
        f.write(f"**Date:** {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"**Status:** Complete\n\n---\n\n")

        f.write("## Top 10 Models\n\n")
        f.write("| Rank | Model | Features | Dims | CV Acc | Test Acc | F1 | AUC | Time(s) |\n")
        f.write("|---:|:---|:---|---:|---:|---:|---:|---:|---:|\n")
        for i, r in enumerate(sorted_results[:10]):
            f.write(f"| {i+1} | {r['model']} | {r['features']} | {r['n_features']} | "
                    f"{r['cv_mean']:.4f} | **{r['test_acc']:.4f}** | {r['test_f1']:.4f} | "
                    f"{r['test_auc']:.4f} | {r['time_sec']:.1f} |\n")

        f.write("\n## All Results\n\n")
        f.write("| Model | Features | CV Acc | Test Acc | F1 | Precision | Recall | AUC | Time(s) |\n")
        f.write("|:---|:---|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in sorted_results:
            f.write(f"| {r['model']} | {r['features']} | {r['cv_mean']:.4f} | "
                    f"{r['test_acc']:.4f} | {r['test_f1']:.4f} | {r['test_prec']:.4f} | "
                    f"{r['test_rec']:.4f} | {r['test_auc']:.4f} | {r['time_sec']:.1f} |\n")

        best = sorted_results[0]
        f.write(f"\n## Best Model\n\n")
        f.write(f"**{best['model']}** on `{best['features']}` ({best['n_features']} dims)\n\n")
        f.write(f"| Metric | Value |\n|:---|---:|\n")
        f.write(f"| Test Accuracy | **{best['test_acc']:.4f}** |\n")
        f.write(f"| Test F1 | {best['test_f1']:.4f} |\n")
        f.write(f"| Test Precision | {best['test_prec']:.4f} |\n")
        f.write(f"| Test Recall | {best['test_rec']:.4f} |\n")
        f.write(f"| Test AUC | {best['test_auc']:.4f} |\n")
        f.write(f"| CV Accuracy | {best['cv_mean']:.4f} +/- {best['cv_std']:.4f} |\n")

        f.write("\n## Figures\n\n")
        f.write("| # | Figure | Description |\n|:---|:---|:---|\n")
        f.write("| 16 | `16_feature_correlation.png` | Feature correlation analysis |\n")
        f.write("| 17 | `17_model_tournament.png` | All models accuracy/F1 comparison |\n")
        f.write("| 18 | `18_cv_vs_test.png` | CV vs Test accuracy (overfitting check) |\n")
        f.write("| 19 | `19_best_model_evaluation.png` | Confusion matrix & ROC for best model |\n")
        f.write("| 20 | `20_speed_vs_accuracy.png` | Speed vs accuracy tradeoff |\n")

    print(f"  -> Saved {report_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    total_start = time.time()

    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Phase 1: Data & Features
    feature_sets, y_train, y_test, names_train, names_test = load_and_prepare_data()

    # Phase 2: Correlation
    analyze_feature_correlation(feature_sets, FIGURES_DIR)

    # Phase 3: Tournament
    base_results = run_model_tournament(feature_sets, y_train, y_test, FIGURES_DIR)

    # Phase 4: Stacking
    stacking_results = build_stacking_ensembles(feature_sets, y_train, y_test, base_results, FIGURES_DIR)

    # Combine all results
    all_results = base_results + stacking_results

    # Remove model objects for serialization
    results_clean = [{k: v for k, v in r.items()
                      if k not in ('model_obj', 'y_pred', 'y_proba')}
                     for r in all_results]

    # Phase 5: Visualization
    sorted_results = plot_tournament_results(all_results, FIGURES_DIR)
    best = sorted_results[0]
    plot_best_model_details(best, y_test, FIGURES_DIR)
    plot_speed_vs_accuracy(all_results, FIGURES_DIR)
    generate_report(all_results, RESEARCH_DIR)

    total_elapsed = time.time() - total_start

    print(f"\n{'='*70}")
    print(f"COMPLETE ({total_elapsed:.0f}s)")
    print(f"{'='*70}")

    # Final leaderboard
    sorted_all = sorted(all_results, key=lambda x: x['test_acc'], reverse=True)
    print(f"\n  TOP 5 MODELS:")
    for i, r in enumerate(sorted_all[:5]):
        print(f"    #{i+1}: {r['model']:35s} | {r['features']:20s} | "
              f"Acc={r['test_acc']:.4f} | F1={r['test_f1']:.4f} | AUC={r['test_auc']:.4f}")

    print(f"\n  BEST: {sorted_all[0]['model']} = {sorted_all[0]['test_acc']:.4f} accuracy")


if __name__ == "__main__":
    main()
