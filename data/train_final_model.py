"""
Train Final Production MLP Classifier
========================================
Trains MLP_large (512-256-128) on full training data,
evaluates on held-out test set, and exports model + scaler.

Output:
  - models/mlp_classifier.joblib   (trained MLP)
  - models/scaler.joblib           (fitted StandardScaler)
  - models/model_config.json       (metadata)
  - research/figures/21_final_*.png (evaluation plots)
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
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)

# ============================================================================
# CONFIG
# ============================================================================

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
EMBEDDINGS_FILE = os.path.join(BASE_DIR, "embeddings", "embeddings.npz")
MODELS_DIR = os.path.join(BASE_DIR, "models")
FIGURES_DIR = os.path.join(BASE_DIR, "research", "figures")

RANDOM_STATE = 42
TEST_SIZE = 0.15

plt.rcParams.update({
    'figure.dpi': 150,
    'font.size': 11,
    'font.family': 'sans-serif',
    'axes.titlesize': 14,
    'axes.labelsize': 12,
})


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # ================================================================
    # 1. LOAD DATA
    # ================================================================
    print("=" * 70)
    print("PHASE 1: LOADING DATA")
    print("=" * 70)

    data = np.load(EMBEDDINGS_FILE, allow_pickle=True)
    emb = data['embeddings']
    labels = data['labels']
    names = data['names']
    print(f"  Full dataset: {emb.shape}")
    print(f"  Class 0 (Person):   {(labels == 0).sum():,}")
    print(f"  Class 1 (Merchant): {(labels == 1).sum():,}")

    # Train/test split (same as model selection for comparability)
    X_train, X_test, y_train, y_test, names_train, names_test = train_test_split(
        emb, labels, names, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=labels
    )
    print(f"\n  Train: {X_train.shape}")
    print(f"  Test:  {X_test.shape}")

    # ================================================================
    # 2. SCALE FEATURES
    # ================================================================
    print("\n" + "=" * 70)
    print("PHASE 2: SCALING FEATURES")
    print("=" * 70)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"  Scaler fitted on {X_train.shape[0]:,} samples")
    print(f"  Mean range: [{scaler.mean_.min():.4f}, {scaler.mean_.max():.4f}]")
    print(f"  Scale range: [{scaler.scale_.min():.4f}, {scaler.scale_.max():.4f}]")

    # ================================================================
    # 3. TRAIN FINAL MLP
    # ================================================================
    print("\n" + "=" * 70)
    print("PHASE 3: TRAINING FINAL MLP_LARGE (512-256-128)")
    print("=" * 70)

    model = MLPClassifier(
        hidden_layer_sizes=(512, 256, 128),
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=RANDOM_STATE,
        early_stopping=True,
        validation_fraction=0.1,
        batch_size=512,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        verbose=True,
        n_iter_no_change=15,
    )

    print(f"\n  Architecture: 1024 -> 512 -> 256 -> 128 -> 2")
    print(f"  Activation: ReLU (hidden), Softmax (output)")
    print(f"  Optimizer: Adam, lr=0.001, adaptive")
    print(f"  Early stopping: patience=15, val_fraction=10%")
    print(f"  Batch size: 512")
    print(f"\n  Training on {X_train_scaled.shape[0]:,} samples...\n")

    start = time.time()
    model.fit(X_train_scaled, y_train)
    train_time = time.time() - start

    print(f"\n  Training complete in {train_time:.1f}s")
    print(f"  Epochs run: {model.n_iter_}")
    print(f"  Best validation score: {model.best_validation_score_:.4f}")
    print(f"  Final loss: {model.loss_:.6f}")

    # ================================================================
    # 4. EVALUATE ON TEST SET
    # ================================================================
    print("\n" + "=" * 70)
    print("PHASE 4: EVALUATION ON HELD-OUT TEST SET")
    print("=" * 70)

    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"\n  Test Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Test Precision: {prec:.4f}")
    print(f"  Test Recall:    {rec:.4f}")
    print(f"  Test F1:        {f1:.4f}")
    print(f"  Test AUC:       {auc:.4f}")

    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['PERSON', 'MERCHANT']))

    cm = confusion_matrix(y_test, y_pred)
    print(f"  Confusion Matrix:")
    print(f"                  Predicted")
    print(f"                  PERSON  MERCHANT")
    print(f"  Actual PERSON   {cm[0][0]:>6}  {cm[0][1]:>6}")
    print(f"  Actual MERCHANT {cm[1][0]:>6}  {cm[1][1]:>6}")

    # ================================================================
    # 5. ERROR ANALYSIS
    # ================================================================
    print("\n" + "=" * 70)
    print("PHASE 5: ERROR ANALYSIS")
    print("=" * 70)

    errors = y_pred != y_test
    error_names = names_test[errors]
    error_true = y_test[errors]
    error_pred = y_pred[errors]
    error_proba = y_proba[errors]

    print(f"\n  Total errors: {errors.sum()} / {len(y_test)} ({errors.mean()*100:.2f}%)")

    # False positives (person classified as merchant)
    fp_mask = (error_true == 0)
    fp_names = error_names[fp_mask]
    fp_proba = error_proba[fp_mask]
    print(f"\n  False Positives (Person -> Merchant): {fp_mask.sum()}")
    if len(fp_names) > 0:
        # Sort by confidence (most confident errors first)
        fp_order = np.argsort(-fp_proba)
        print(f"  Top 20 most confident FP errors:")
        for i in fp_order[:20]:
            print(f"    '{fp_names[i]}' (P(merchant)={fp_proba[i]:.4f})")

    # False negatives (merchant classified as person)
    fn_mask = (error_true == 1)
    fn_names = error_names[fn_mask]
    fn_proba = error_proba[fn_mask]
    print(f"\n  False Negatives (Merchant -> Person): {fn_mask.sum()}")
    if len(fn_names) > 0:
        fn_order = np.argsort(fn_proba)  # lowest P(merchant) = most confident FN
        print(f"  Top 20 most confident FN errors:")
        for i in fn_order[:20]:
            print(f"    '{fn_names[i]}' (P(merchant)={fn_proba[i]:.4f})")

    # ================================================================
    # 6. PLOTS
    # ================================================================
    print("\n" + "=" * 70)
    print("PHASE 6: GENERATING EVALUATION PLOTS")
    print("=" * 70)

    # Plot 1: Confusion Matrix + ROC + Loss Curve
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Confusion Matrix
    ax = axes[0]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['PERSON', 'MERCHANT'],
                yticklabels=['PERSON', 'MERCHANT'],
                annot_kws={'size': 16})
    ax.set_xlabel('Predicted', fontsize=13)
    ax.set_ylabel('Actual', fontsize=13)
    ax.set_title('Confusion Matrix', fontweight='bold')

    # ROC Curve
    ax = axes[1]
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC={auc:.4f})')
    ax.fill_between(fpr, tpr, alpha=0.1, color='blue')
    ax.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=13)
    ax.set_ylabel('True Positive Rate', fontsize=13)
    ax.set_title('ROC Curve', fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    # Training Loss Curve
    ax = axes[2]
    ax.plot(model.loss_curve_, 'b-', linewidth=1.5, label='Training Loss')
    if hasattr(model, 'validation_scores_'):
        ax2 = ax.twinx()
        ax2.plot(model.validation_scores_, 'g-', linewidth=1.5, label='Validation Acc')
        ax2.set_ylabel('Validation Accuracy', color='g', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='g')
        ax2.legend(loc='center right', fontsize=10)
    ax.set_xlabel('Epoch', fontsize=13)
    ax.set_ylabel('Loss', color='b', fontsize=12)
    ax.tick_params(axis='y', labelcolor='b')
    ax.set_title('Training Curves', fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'Final MLP Classifier — Accuracy={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, '21_final_evaluation.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  -> Saved 21_final_evaluation.png")

    # Plot 2: Confidence Distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    correct_proba = y_proba[~errors]
    correct_labels = y_test[~errors]
    ax.hist(correct_proba[correct_labels == 0], bins=50, alpha=0.6, color='#2196F3',
            label='Correct Person', density=True)
    ax.hist(correct_proba[correct_labels == 1], bins=50, alpha=0.6, color='#FF5722',
            label='Correct Merchant', density=True)
    ax.set_xlabel('P(Merchant)')
    ax.set_ylabel('Density')
    ax.set_title('Confidence Distribution (Correct Predictions)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    if errors.sum() > 0:
        ax.hist(error_proba[error_true == 0], bins=30, alpha=0.6, color='#2196F3',
                label='FP (Person->Merchant)', density=True)
        ax.hist(error_proba[error_true == 1], bins=30, alpha=0.6, color='#FF5722',
                label='FN (Merchant->Person)', density=True)
    ax.set_xlabel('P(Merchant)')
    ax.set_ylabel('Density')
    ax.set_title('Confidence Distribution (Errors)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('Model Confidence Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, '22_confidence_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  -> Saved 22_confidence_distribution.png")

    # ================================================================
    # 7. EXPORT MODEL
    # ================================================================
    print("\n" + "=" * 70)
    print("PHASE 7: EXPORTING MODEL")
    print("=" * 70)

    # Save model
    model_path = os.path.join(MODELS_DIR, 'mlp_classifier.joblib')
    joblib.dump(model, model_path)
    model_size = os.path.getsize(model_path)
    print(f"  Model saved: {model_path} ({model_size/1024/1024:.1f} MB)")

    # Save scaler
    scaler_path = os.path.join(MODELS_DIR, 'scaler.joblib')
    joblib.dump(scaler, scaler_path)
    scaler_size = os.path.getsize(scaler_path)
    print(f"  Scaler saved: {scaler_path} ({scaler_size/1024:.1f} KB)")

    # Save config
    config = {
        'model_type': 'MLPClassifier',
        'architecture': '1024 -> 512 -> 256 -> 128 -> 2',
        'activation': 'relu',
        'embedding_model': 'Qwen/Qwen3-Embedding-0.6B',
        'embedding_dim': 1024,
        'scaler': 'StandardScaler',
        'classes': ['person', 'merchant'],
        'class_labels': {0: 'person', 1: 'merchant'},
        'train_samples': int(X_train.shape[0]),
        'test_samples': int(X_test.shape[0]),
        'metrics': {
            'test_accuracy': float(acc),
            'test_precision': float(prec),
            'test_recall': float(rec),
            'test_f1': float(f1),
            'test_auc': float(auc),
        },
        'training': {
            'epochs': int(model.n_iter_),
            'best_val_score': float(model.best_validation_score_),
            'final_loss': float(model.loss_),
            'train_time_sec': float(train_time),
        },
        'confusion_matrix': {
            'true_person_pred_person': int(cm[0][0]),
            'true_person_pred_merchant': int(cm[0][1]),
            'true_merchant_pred_person': int(cm[1][0]),
            'true_merchant_pred_merchant': int(cm[1][1]),
        },
        'errors': {
            'total': int(errors.sum()),
            'false_positives': int(fp_mask.sum()),
            'false_negatives': int(fn_mask.sum()),
        },
    }

    config_path = os.path.join(MODELS_DIR, 'model_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"  Config saved: {config_path}")

    # ================================================================
    # 8. INFERENCE DEMO
    # ================================================================
    print("\n" + "=" * 70)
    print("PHASE 8: INFERENCE DEMO")
    print("=" * 70)

    test_names_demo = [
        "Rajesh Kumar", "HDFC Bank", "Swiggy Instamart",
        "Priya Sharma", "Amazon Pay", "Rahul Verma",
        "Zomato", "Flipkart", "Deepak Singh", "Ola Cabs"
    ]

    print(f"\n  Demo predictions (using first 10 test samples as proxy):")
    demo_X = X_test_scaled[:10]
    demo_names = names_test[:10]
    demo_proba = model.predict_proba(demo_X)
    demo_pred = model.predict(demo_X)

    for i in range(10):
        label = 'MERCHANT' if demo_pred[i] == 1 else 'PERSON'
        conf = max(demo_proba[i])
        actual = 'MERCHANT' if y_test[i] == 1 else 'PERSON'
        status = 'OK' if demo_pred[i] == y_test[i] else 'WRONG'
        print(f"    '{demo_names[i]}' -> {label} ({conf:.4f}) [actual: {actual}] {status}")

    # ================================================================
    # DONE
    # ================================================================
    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"\n  Model:    {model_path}")
    print(f"  Scaler:   {scaler_path}")
    print(f"  Config:   {config_path}")
    print(f"  Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"  F1:       {f1:.4f}")
    print(f"  AUC:      {auc:.4f}")
    print(f"\n  Usage:")
    print(f"    model = joblib.load('models/mlp_classifier.joblib')")
    print(f"    scaler = joblib.load('models/scaler.joblib')")
    print(f"    X_scaled = scaler.transform(embedding.reshape(1, -1))")
    print(f"    prediction = model.predict(X_scaled)  # 0=person, 1=merchant")


if __name__ == "__main__":
    main()
