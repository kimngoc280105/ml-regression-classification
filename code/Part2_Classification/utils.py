"""
Utility Functions and Model Implementations for ML Classification Project
CSC14005 - Machine Learning

This module contains:
- Helper functions for data processing, evaluation, and visualization  
- All model class implementations (Logistic Regression, LDA, QDA, etc.)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from typing import Tuple, List, Dict, Optional, Union
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.stats import norm
from scipy.special import expit
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# ============================================================================
# SECTION 1: DATA LOADING & PREPROCESSING FUNCTIONS
# ============================================================================

def load_adult_data(filepath: str = 'adult.csv') -> pd.DataFrame:
    """Load Adult Census Income dataset"""
    try:
        df = pd.read_csv(filepath)
    except:
        column_names = [
            'age', 'workclass', 'fnlwgt', 'education', 'education-num',
            'marital-status', 'occupation', 'relationship', 'race', 'sex',
            'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
        ]
        df = pd.read_csv(filepath, names=column_names, skipinitialspace=True)
    return df


# ============================================================================
# SECTION 2: ACTIVATION & LOSS FUNCTIONS
# ============================================================================

def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return expit(np.clip(z, -500, 500))


# =====================================================================
# LOSS FUNCTIONS
# =====================================================================

def cross_entropy_loss(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-15) -> float:
    """Binary cross-entropy loss"""
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def cross_entropy_loss_multiclass(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-15) -> float:
    """Multi-class cross-entropy"""
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))


def one_hot_encode(y: np.ndarray, n_classes: int = None) -> np.ndarray:
    """One-hot encode labels"""
    if n_classes is None:
        n_classes = len(np.unique(y))
    n_samples = len(y)
    y_onehot = np.zeros((n_samples, n_classes))
    y_onehot[np.arange(n_samples), y.astype(int)] = 1
    return y_onehot


# ============================================================================
# SECTION 3: EVALUATION METRICS
# ============================================================================

def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate accuracy"""
    return np.mean(y_true == y_pred)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 2) -> np.ndarray:
    """Compute confusion matrix"""
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for i in range(len(y_true)):
        cm[int(y_true[i]), int(y_pred[i])] += 1
    return cm


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'binary') -> Dict[str, float]:
    """Calculate precision, recall, F1-score"""
    if average == 'binary':
        cm = confusion_matrix(y_true, y_pred, n_classes=2)
        tn, fp, fn, tp = cm.ravel()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': accuracy_score(y_true, y_pred)
        }
    else:
        classes = np.unique(y_true)
        precisions, recalls, f1s = [], [], []
        
        for cls in classes:
            y_true_cls = (y_true == cls).astype(int)
            y_pred_cls = (y_pred == cls).astype(int)
            
            tp = np.sum((y_true_cls == 1) & (y_pred_cls == 1))
            fp = np.sum((y_true_cls == 0) & (y_pred_cls == 1))
            fn = np.sum((y_true_cls == 1) & (y_pred_cls == 0))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
        
        return {
            'precision_macro': np.mean(precisions),
            'recall_macro': np.mean(recalls),
            'f1_score_macro': np.mean(f1s),
            'accuracy': accuracy_score(y_true, y_pred)
        }


# ============================================================================
# SECTION 4: VISUALIZATION FUNCTIONS
# ============================================================================


# =====================================================================
# VISUALIZATION FUNCTIONS
# =====================================================================

def plot_confusion_matrix(cm: np.ndarray, class_names: List[str] = None, 
                         title: str = 'Confusion Matrix', figsize: Tuple[int, int] = (8, 6)):
    """Plot confusion matrix heatmap"""
    if class_names is None:
        class_names = [f'Class {i}' for i in range(cm.shape[0])]
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()


def plot_decision_boundary(X: np.ndarray, y: np.ndarray, model, 
                          title: str = 'Decision Boundary', 
                          figsize: Tuple[int, int] = (10, 8),
                          h: float = 0.02):
    """Plot decision boundary for 2D data"""
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=figsize)
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, edgecolor='k', cmap='RdYlBu', alpha=0.8)
    plt.xlabel('Feature 1', fontsize=12)
    plt.ylabel('Feature 2', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.colorbar()
    plt.tight_layout()


def plot_roc_curve(y_true: np.ndarray, y_scores: np.ndarray, 
                   title: str = 'ROC Curve', figsize: Tuple[int, int] = (8, 6)):
    """Plot ROC curve and return AUC"""
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, 'darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    return roc_auc


def plot_pr_curve(y_true: np.ndarray, y_scores: np.ndarray,
                  title: str = 'Precision-Recall Curve', figsize: Tuple[int, int] = (8, 6)):
    """Plot PR curve and return AP"""
    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)
    
    plt.figure(figsize=figsize)
    plt.plot(recall, precision, 'darkorange', lw=2, label=f'PR (AP = {ap:.3f})')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tight_layout()
    return ap


def plot_learning_curves(train_losses: List[float], val_losses: List[float] = None,
                         title: str = 'Learning Curves', figsize: Tuple[int, int] = (10, 6)):
    """Plot loss curves"""
    plt.figure(figsize=figsize)
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    if val_losses is not None:
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()


def plot_class_distribution(y, labels=('<=50K', '> 50K'),
                             colors=('#4C72B0', '#DD8452'),
                             title='Phân phối lớp – Adult',
                             figsize=(12, 4)):
    """Bar + pie chart for binary class distribution."""
    counts = [int(np.sum(y == 0)), int(np.sum(y == 1))]
    n_total = len(y)
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    axes[0].bar(list(labels), counts, color=list(colors), edgecolor='k')
    for i, v in enumerate(counts):
        axes[0].text(i, v + 200, f'{v:,}\n({v/n_total*100:.1f}%)',
                     ha='center', fontsize=11)
    axes[0].set_title(title, fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Số mẫu')
    axes[0].set_ylim(0, max(counts) * 1.20)
    axes[1].pie(counts, labels=list(labels), autopct='%1.1f%%',
                colors=list(colors), startangle=90)
    axes[1].set_title(f'Imbalance ratio = {counts[0]/counts[1]:.2f}:1',
                      fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_boxplots_by_class(df, num_cols, y,
                            class_labels=('<=50K', '>50K'),
                            colors=('#4C72B0', '#DD8452'),
                            figsize=(14, 12)):
    """Horizontal boxplots (3×2) for numerical features split by class."""
    fig, axes = plt.subplots(3, 2, figsize=figsize)
    axes = axes.flatten()
    for i, col in enumerate(num_cols):
        data_0 = df.loc[y == 0, col].values
        data_1 = df.loc[y == 1, col].values
        bp = axes[i].boxplot(
            [data_0, data_1], labels=list(class_labels),
            vert=False, patch_artist=True, notch=False,
            medianprops=dict(color='black', linewidth=2),
            whiskerprops=dict(linewidth=1.5),
            capprops=dict(linewidth=1.5)
        )
        bp['boxes'][0].set_facecolor(colors[0]); bp['boxes'][0].set_alpha(0.7)
        bp['boxes'][1].set_facecolor(colors[1]); bp['boxes'][1].set_alpha(0.7)
        for k, data in enumerate([data_0, data_1]):
            med = np.median(data)
            axes[i].text(med, k + 1, f'{med:.1f}',
                         ha='left', va='center', fontsize=8, fontweight='bold',
                         color='white',
                         bbox=dict(boxstyle='round,pad=0.15', facecolor='#333', alpha=0.6))
        axes[i].set_title(col, fontsize=11, fontweight='bold')
        axes[i].set_xlabel('Value')
        axes[i].grid(alpha=0.3, axis='x')
    plt.suptitle('Boxplot – Phân phối Numerical Features theo Income Class\n'
                 '(cam = >50K  |  xanh = <=50K  |  label = median)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()
    print("Nhận xét:")
    for col in num_cols:
        m0 = df.loc[y == 0, col].median()
        m1 = df.loc[y == 1, col].median()
        diff = (m1 - m0) / (m0 + 1e-9) * 100
        print(f"  {col:<18}: median <=50K={m0:.1f}, >50K={m1:.1f}  (Δ={diff:+.1f}%)")


def plot_histograms_by_class(df, num_cols, y,
                              class_labels=('<=50K', '>50K'),
                              colors=('#4C72B0', '#DD8452'),
                              figsize=(15, 8)):
    """Histogram (2×3) of numerical features split by binary class."""
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()
    for i, col in enumerate(num_cols):
        for cls, color, lbl in zip([0, 1], colors, class_labels):
            axes[i].hist(df.loc[y == cls, col], bins=30, alpha=0.6,
                         color=color, label=lbl, density=True)
        axes[i].set_title(col, fontsize=11, fontweight='bold')
        axes[i].legend(fontsize=8)
    plt.suptitle('Phân phối Đặc trưng Số theo Lớp', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_scatter_pairs(df, y, pairs,
                        class_labels=('<=50K', '>50K'),
                        colors=('#4C72B080', '#DD845280'),
                        figsize=(20, 5)):
    """Scatter plots for selected feature pairs, colored by class."""
    fig, axes = plt.subplots(1, len(pairs), figsize=figsize)
    if len(pairs) == 1:
        axes = [axes]
    for ax, (fx, fy) in zip(axes, pairs):
        for cls, color, lbl in zip([0, 1], colors, class_labels):
            mask = y == cls
            ax.scatter(df.loc[mask, fx], df.loc[mask, fy],
                       c=color, label=lbl, s=4, alpha=0.4)
        ax.set_xlabel(fx, fontsize=10); ax.set_ylabel(fy, fontsize=10)
        ax.set_title(f'{fx}  vs  {fy}', fontsize=10, fontweight='bold')
        ax.legend(fontsize=8, markerscale=3); ax.grid(alpha=0.3)
    plt.suptitle('Scatter Plots – Cặp Biến Quan Trọng theo Income',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()
    print("Tương quan Pearson với target (income):")
    num_cols_df = [c for c in df.columns if df[c].dtype != object]
    tmp = df[num_cols_df].copy(); tmp['income'] = y
    corr_target = tmp.corr()['income'].drop('income').sort_values(ascending=False)
    for feat, val in corr_target.items():
        bar = '█' * int(abs(val) * 30)
        print(f"  {feat:<20}: {val:+.4f}  {bar}")


def plot_outlier_detection(df, num_cols, figsize=(16, 8)):
    """IQR and Z-score outlier detection; visualises top-3 most-affected features."""
    from scipy import stats as sp_stats
    print(f"{'Feature':<20} {'IQR #':>8} {'IQR %':>7} {'Z>3 #':>8} {'Z>3 %':>7}  {'Skewness':>10}")
    print("─" * 68)
    iqr_counts = {}
    for col in num_cols:
        vals = df[col].values.astype(float)
        Q1, Q3 = np.percentile(vals, 25), np.percentile(vals, 75)
        IQR = Q3 - Q1
        iqr_out = ((vals < Q1 - 1.5*IQR) | (vals > Q3 + 1.5*IQR)).sum()
        z = (vals - vals.mean()) / (vals.std() + 1e-9)
        z_out = (np.abs(z) > 3).sum()
        skew = sp_stats.skew(vals)
        iqr_counts[col] = iqr_out
        print(f"  {col:<18} {iqr_out:>8,} {iqr_out/len(vals)*100:>6.1f}%"
              f" {z_out:>8,} {z_out/len(vals)*100:>6.1f}%  {skew:>+10.2f}")
    print("─" * 68)
    print("* IQR rule: < Q1−1.5·IQR  hoặc  > Q3+1.5·IQR")
    print("* Z-score:  |z| > 3  (≈ 0.3% dưới normal distribution)")
    top3 = sorted(iqr_counts, key=iqr_counts.get, reverse=True)[:3]
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    for col_i, col in enumerate(top3):
        vals = df[col].values.astype(float)
        Q1, Q3 = np.percentile(vals, 25), np.percentile(vals, 75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
        is_out = (vals < lower) | (vals > upper)
        ax_sc = axes[0, col_i]
        ax_sc.scatter(np.where(~is_out)[0], vals[~is_out], c='#4C72B0', s=1, alpha=0.3, label='Normal')
        ax_sc.scatter(np.where(is_out)[0], vals[is_out], c='#DD8452', s=6, alpha=0.6,
                      label=f'Outlier ({is_out.sum():,})')
        ax_sc.axhline(lower, color='r', ls='--', lw=1.5, label=f'Lower={lower:.0f}')
        ax_sc.axhline(upper, color='r', ls='-.', lw=1.5, label=f'Upper={upper:.0f}')
        ax_sc.set_title(f'{col} – IQR Outliers', fontsize=10, fontweight='bold')
        ax_sc.legend(fontsize=7); ax_sc.grid(alpha=0.3)
        ax_hz = axes[1, col_i]
        z = (vals - vals.mean()) / (vals.std() + 1e-9)
        ax_hz.hist(z[np.abs(z) <= 3], bins=60, color='#4C72B0', alpha=0.7, label='Normal (|z|≤3)')
        ax_hz.hist(z[np.abs(z) > 3], bins=30, color='#DD8452', alpha=0.8,
                   label=f'Outlier |z|>3 ({(np.abs(z)>3).sum():,})')
        ax_hz.axvline(3, color='r', ls='--', lw=1.5)
        ax_hz.axvline(-3, color='r', ls='--', lw=1.5)
        ax_hz.set_title(f'{col} – Z-score Distribution', fontsize=10, fontweight='bold')
        ax_hz.set_xlabel('Z-score'); ax_hz.legend(fontsize=7); ax_hz.grid(alpha=0.3)
    plt.suptitle('Outlier Detection – IQR & Z-score (Top-3 Features)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()


# ============================================================================
# SECTION 4b: EVALUATION & COMPARISON VISUALIZATION HELPERS
# ============================================================================


def plot_loss_and_cm(losses, y_true, y_pred, n_classes=2,
                     class_labels=None, title='', figsize=(13, 5)):
    """Plot training loss curve + confusion matrix side by side."""
    if class_labels is None:
        class_labels = [str(i) for i in range(n_classes)]
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    axes[0].plot(losses, 'b-', lw=2)
    axes[0].set(xlabel='Epoch', ylabel='Cross-Entropy Loss',
                title=f'{title}: Loss vs Epoch' if title else 'Loss vs Epoch')
    axes[0].grid(alpha=0.4)
    cm = confusion_matrix(y_true, y_pred, n_classes=n_classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1],
                xticklabels=class_labels, yticklabels=class_labels)
    axes[1].set_title(f'Confusion Matrix{" \u2013 " + title if title else ""}',
                      fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')
    plt.tight_layout()
    plt.show()


def plot_gd_newton_convergence(gd_model, newton_model, figsize=(14, 5)):
    """Compare GD vs Newton-Raphson: loss vs iterations and loss vs wall-clock time."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    axes[0].plot(gd_model.losses, 'b-', lw=2, label=f'GD ({len(gd_model.losses)} epochs)')
    axes[0].plot(newton_model.losses, 'r-o', lw=2, ms=6,
                 label=f'Newton ({len(newton_model.losses)} iters)')
    axes[0].set(xlabel='Iteration/Epoch', ylabel='Cross-Entropy Loss',
                title='H\u1ed9i t\u1ee5: Loss vs Epoch')
    axes[0].legend(fontsize=11)
    axes[0].grid(alpha=0.4)
    axes[1].plot(gd_model.time_history, gd_model.losses, 'b-', lw=2, label='GD')
    axes[1].plot(newton_model.time_history, newton_model.losses, 'r-o', lw=2, ms=6,
                 label='Newton')
    axes[1].set(xlabel='Wall-clock Time (s)', ylabel='Cross-Entropy Loss',
                title='H\u1ed9i t\u1ee5: Loss vs Time')
    axes[1].legend(fontsize=11)
    axes[1].grid(alpha=0.4)
    plt.suptitle('GD vs Newton-Raphson \u2013 T\u1ed1c \u0111\u1ed9 h\u1ed9i t\u1ee5',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()
    print(f"GD: {len(gd_model.losses)} epochs, loss={gd_model.losses[-1]:.6f}, "
          f"t={gd_model.training_time:.3f}s")
    print(f"Newton: {len(newton_model.losses)} iters, loss={newton_model.losses[-1]:.6f}, "
          f"t={newton_model.training_time:.3f}s")
    ratio = len(gd_model.losses) / max(1, len(newton_model.losses))
    print(f"Newton nhanh h\u01a1n ~{ratio:.0f}x v\u1ec1 s\u1ed1 iterations")


def plot_roc_pr_curves(all_models, y_true, figsize=(16, 6)):
    """ROC + PR curves for all models.
    all_models = {name: (clf, y_pred, y_proba)} with y_proba as 1-D positive-class probabilities.
    """
    from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_models)))
    for (name, (_, _, ypr)), color in zip(all_models.items(), colors):
        ypr = np.asarray(ypr)
        if ypr is not None and ypr.ndim == 1:
            fpr, tpr, _ = roc_curve(y_true, ypr)
            a = auc(fpr, tpr)
            axes[0].plot(fpr, tpr, color=color, lw=1.5, label=f'{name}({a:.3f})')
            prec_c, rec_c, _ = precision_recall_curve(y_true, ypr)
            ap = average_precision_score(y_true, ypr)
            axes[1].plot(rec_c, prec_c, color=color, lw=1.5, label=f'{name}({ap:.3f})')
    axes[0].plot([0, 1], [0, 1], 'k--', lw=1)
    axes[0].set(xlabel='FPR', ylabel='TPR', title='ROC Curves (AUC)')
    axes[0].legend(fontsize=7, loc='lower right')
    axes[0].grid(alpha=0.4)
    bl = (np.asarray(y_true) == 1).mean()
    axes[1].axhline(bl, color='k', ls='--', label=f'Baseline={bl:.3f}')
    axes[1].set(xlabel='Recall', ylabel='Precision', title='PR Curves (AP)')
    axes[1].set_xlim([0, 1])
    axes[1].set_ylim([0, 1.05])
    axes[1].legend(fontsize=7, loc='upper right')
    axes[1].grid(alpha=0.4)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix_grid(all_models, y_true, n_classes=2,
                                class_labels=None, n_cols=5, figsize=(20, 8)):
    """Grid of confusion matrices for all models.
    all_models = {name: (clf, y_pred, y_proba)}
    """
    if class_labels is None:
        class_labels = [str(i) for i in range(n_classes)]
    n_models = len(all_models)
    n_rows = (n_models + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    i = 0
    for i, (name, (_, yp, _)) in enumerate(all_models.items()):
        cm = confusion_matrix(y_true, yp, n_classes=n_classes)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                    xticklabels=class_labels, yticklabels=class_labels, cbar=False)
        acc = accuracy_score(y_true, yp)
        axes[i].set_title(f'{name}\nAcc={acc:.3f}', fontsize=9, fontweight='bold')
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    plt.suptitle('Confusion Matrices \u2013 All Models', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_gradient_loss_curves(loss_models, styles=None,
                               title='Loss Curves \u2013 Gradient-based Models',
                               figsize=(12, 5)):
    """Plot loss curves for multiple gradient-based models.
    loss_models = {name: clf} where clf has a .losses attribute.
    """
    if styles is None:
        styles = ['b-', 'r-o', 'g-', 'm-', 'c-', 'k--']
    fig, ax = plt.subplots(figsize=figsize)
    for (name, clf), sty in zip(loss_models.items(), styles):
        ax.plot(clf.losses, sty, lw=2, ms=4, label=name)
    ax.set(xlabel='Epoch', ylabel='Cross-Entropy Loss', title=title)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.4)
    plt.tight_layout()
    plt.show()


def plot_calibration_curves(calib_models, y_true, n_bins=10, figsize=(15, 5)):
    """Reliability diagrams for model calibration.
    calib_models = [(name, y_proba), ...]
    """
    from sklearn.calibration import calibration_curve as _calib_curve
    fig, axes = plt.subplots(1, len(calib_models), figsize=figsize)
    if len(calib_models) == 1:
        axes = [axes]
    for ax, (name, ypr) in zip(axes, calib_models):
        fraction_pos, mean_pred = _calib_curve(y_true, ypr, n_bins=n_bins)
        ax.plot(mean_pred, fraction_pos, 'o-', color='b', lw=2, ms=6, label=name)
        ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Perfect calibration')
        ax.set(xlabel='Mean Predicted Probability', ylabel='Fraction of Positives',
               title=f'Reliability Diagram \u2013 {name}')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.4)
    plt.suptitle('Calibration Analysis (Reliability Diagrams)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_metrics_comparison_bar(rows_dict, metric_keys=('F1', 'AUC', 'Recall'),
                                 figsize=(14, 5)):
    """Grouped bar chart comparing metrics across models.
    rows_dict = {name: {metric_key: value, ...}}
    """
    df_final = pd.DataFrame(
        {n: {k: rows_dict[n][k] for k in metric_keys} for n in rows_dict}
    ).T
    colors_bar = ['#4C72B0', '#DD8452', '#55A868']
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(df_final))
    w = 0.25
    for i, (key, color) in enumerate(zip(metric_keys, colors_bar)):
        bars = ax.bar(x + (i - 1) * w, df_final[key], w, label=key,
                      color=color, edgecolor='k', alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                    f'{h:.2f}', ha='center', va='bottom', fontsize=6.5,
                    rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(df_final.index, rotation=30, ha='right', fontsize=9)
    ax.set(ylabel='Score',
           title='F1 / AUC / Recall \u2013 T\u1ea5t c\u1ea3 Models', ylim=[0, 1.1])
    ax.legend(fontsize=10)
    ax.grid(alpha=0.4, axis='y')
    plt.tight_layout()
    plt.show()


# ============================================================================
# SECTION 5: STATISTICAL TESTS
# ============================================================================


# =====================================================================
# STATISTICAL TESTS
# =====================================================================

def mcnemar_test(y_true: np.ndarray, y_pred1: np.ndarray, y_pred2: np.ndarray) -> Tuple[float, float]:
    """McNemar's test: (chi2_stat, p_value)"""
    from scipy.stats import chi2
    
    correct1 = (y_pred1 == y_true)
    correct2 = (y_pred2 == y_true)
    
    n01 = np.sum(~correct1 & correct2)
    n10 = np.sum(correct1 & ~correct2)
    
    chi2_stat = (abs(n01 - n10) - 1)**2 / (n01 + n10) if (n01 + n10) > 0 else 0
    p_value = 1 - chi2.cdf(chi2_stat, df=1)
    
    return chi2_stat, p_value


# ============================================================================
# SECTION 6: MODEL IMPLEMENTATIONS
# ============================================================================


class LogisticRegressionGD:
    """
    Logistic Regression using Gradient Descent
    """
    
    def __init__(self, learning_rate=0.01, n_iterations=1000, reg_lambda=0.0, reg_type=None):
        """
        Parameters:
        -----------
        learning_rate : float
            Learning rate for gradient descent
        n_iterations : int
            Number of iterations
        reg_lambda : float
            Regularization parameter
        reg_type : str or None
            'l1', 'l2', or None
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.reg_lambda = reg_lambda
        self.reg_type = reg_type
        self.theta = None
        self.losses = []
        self.time_history = []
        self.training_time = 0
        
    def fit(self, X, y, class_weights=None):
        """
        Fit logistic regression model
        
        Parameters:
        -----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data
        y : np.ndarray of shape (n_samples,)
            Target values (0 or 1)
        class_weights : dict or None
            Weights for each class {0: w0, 1: w1}
        """
        start_time = time.time()
        
        # Add bias term
        m, n = X.shape
        X_bias = np.c_[np.ones((m, 1)), X]
        
        # Initialize parameters
        self.theta = np.zeros(n + 1)
        
        # Compute sample weights
        if class_weights is not None:
            sample_weights = np.array([class_weights[int(yi)] for yi in y])
        else:
            sample_weights = np.ones(m)
        
        # Gradient Descent
        for i in range(self.n_iterations):
            # Forward pass
            z = X_bias @ self.theta
            h = sigmoid(z)
            
            # Compute loss (weighted cross-entropy)
            epsilon = 1e-15
            h_clip = np.clip(h, epsilon, 1 - epsilon)
            loss = -np.mean(sample_weights * (y * np.log(h_clip) + (1 - y) * np.log(1 - h_clip)))
            
            # Add regularization to loss
            if self.reg_type == 'l2':
                loss += (self.reg_lambda / (2 * m)) * np.sum(self.theta[1:]**2)
            elif self.reg_type == 'l1':
                loss += (self.reg_lambda / m) * np.sum(np.abs(self.theta[1:]))
            
            self.losses.append(loss)
            self.time_history.append(time.time() - start_time)
            
            # Compute gradient
            error = h - y
            gradient = (X_bias.T @ (sample_weights * error)) / m
            
            # Add regularization to gradient
            if self.reg_type == 'l2':
                gradient[1:] += (self.reg_lambda / m) * self.theta[1:]
            elif self.reg_type == 'l1':
                gradient[1:] += (self.reg_lambda / m) * np.sign(self.theta[1:])
            
            # Update parameters
            self.theta -= self.learning_rate * gradient
        
        self.training_time = time.time() - start_time
        
    def predict_proba(self, X):
        """Predict probabilities"""
        X_bias = np.c_[np.ones((X.shape[0], 1)), X]
        return sigmoid(X_bias @ self.theta)
    
    def predict(self, X, threshold=0.5):
        """Predict class labels"""
        return (self.predict_proba(X) >= threshold).astype(int)


class LogisticRegressionNewton:
    """
    Logistic Regression using Newton-Raphson / IRLS
    """
    
    def __init__(self, n_iterations=20, tol=1e-6):
        self.n_iterations = n_iterations
        self.tol = tol
        self.theta = None
        self.losses = []
        self.time_history = []
        self.training_time = 0
        
    def fit(self, X, y):
        start_time = time.time()
        
        m, n = X.shape
        X_bias = np.c_[np.ones((m, 1)), X]
        self.theta = np.zeros(n + 1)
        
        for iteration in range(self.n_iterations):
            # Predictions
            z = X_bias @ self.theta
            h = sigmoid(z)
            
            # Loss
            epsilon = 1e-15
            h_clip = np.clip(h, epsilon, 1 - epsilon)
            loss = -np.mean(y * np.log(h_clip) + (1 - y) * np.log(1 - h_clip))
            self.losses.append(loss)
            self.time_history.append(time.time() - start_time)
            
            # Gradient
            gradient = (X_bias.T @ (h - y)) / m
            
            # Hessian without materializing the m x m diagonal matrix
            weights = h * (1 - h)
            X_weighted = X_bias * weights[:, np.newaxis]
            H = (X_bias.T @ X_weighted) / m
            
            # Add small regularization for numerical stability
            H += 1e-5 * np.eye(H.shape[0])
            
            # Newton update
            try:
                delta = np.linalg.solve(H, gradient)
                self.theta -= delta
            except np.linalg.LinAlgError:
                print(f"Warning: Singular Hessian at iteration {iteration}")
                break
            
            # Check convergence
            if np.linalg.norm(delta) < self.tol:
                print(f"Converged at iteration {iteration+1}")
                break
        
        self.training_time = time.time() - start_time
        
    def predict_proba(self, X):
        X_bias = np.c_[np.ones((X.shape[0], 1)), X]
        return sigmoid(X_bias @ self.theta)
    
    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


class SoftmaxRegression:
    """Multi-class Logistic Regression using Softmax"""
    
    def __init__(self, learning_rate=0.01, n_iterations=1000, reg_lambda=0.0, n_classes=None):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.reg_lambda = reg_lambda
        self.theta = None
        self.n_classes = n_classes
        self.losses = []
        self.training_time = 0
        
    def softmax(self, z):
        """Numerically stable softmax"""
        z_shifted = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def fit(self, X, y):
        start_time = time.time()
        
        m, n = X.shape
        if self.n_classes is None:
            self.n_classes = int(np.max(y)) + 1
        
        # Add bias term
        X_bias = np.c_[np.ones((m, 1)), X]
        
        # Initialize: (n_features+1) x n_classes
        self.theta = np.zeros((n + 1, self.n_classes))
        
        # One-hot encode labels
        Y_one_hot = np.zeros((m, self.n_classes))
        Y_one_hot[np.arange(m), y] = 1
        
        # Gradient descent
        for iteration in range(self.n_iterations):
            # Forward pass
            logits = X_bias @ self.theta
            probs = self.softmax(logits)
            
            # Cross-entropy loss
            epsilon = 1e-15
            probs_clip = np.clip(probs, epsilon, 1 - epsilon)
            loss = -np.mean(np.sum(Y_one_hot * np.log(probs_clip), axis=1))
            
            # L2 regularization
            if self.reg_lambda > 0:
                loss += (self.reg_lambda / (2 * m)) * np.sum(self.theta[1:]**2)
            
            self.losses.append(loss)
            
            # Gradient: simplified using Jacobian
            error = probs - Y_one_hot
            gradient = (X_bias.T @ error) / m
            
            if self.reg_lambda > 0:
                gradient[1:] += (self.reg_lambda / m) * self.theta[1:]
            
            self.theta -= self.learning_rate * gradient
        
        self.training_time = time.time() - start_time
    
    def predict_proba(self, X):
        X_bias = np.c_[np.ones((X.shape[0], 1)), X]
        logits = X_bias @ self.theta
        return self.softmax(logits)
    
    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)


class OneVsRestClassifier:
    """One-vs-Rest Multi-class Strategy"""
    
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.classifiers = []
        self.n_classes = None
        self.training_time = 0
        
    def fit(self, X, y):
        start_time = time.time()
        
        self.n_classes = int(np.max(y)) + 1
        self.classifiers = []
        
        for k in range(self.n_classes):
            # Binary labels: class k vs rest
            y_binary = (y == k).astype(int)
            
            clf = LogisticRegressionGD(
                learning_rate=self.learning_rate,
                n_iterations=self.n_iterations
            )
            clf.fit(X, y_binary)
            self.classifiers.append(clf)
        
        self.training_time = time.time() - start_time
    
    def predict_proba(self, X):
        probs = np.zeros((X.shape[0], self.n_classes))
        for k, clf in enumerate(self.classifiers):
            probs[:, k] = clf.predict_proba(X)
        
        # Normalize to sum to 1
        probs = probs / np.sum(probs, axis=1, keepdims=True)
        return probs
    
    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)


    class OneVsOneClassifier:
        """One-vs-One Multi-class Strategy with Voting"""
        
        def __init__(self, learning_rate=0.01, n_iterations=1000):
            self.learning_rate = learning_rate
            self.n_iterations = n_iterations
            self.classifiers = {}
            self.n_classes = None
            self.training_time = 0
            
        def fit(self, X, y):
            start_time = time.time()
            
            self.n_classes = int(np.max(y)) + 1
            self.classifiers = {}
            
            # Train for each pair
            for i in range(self.n_classes):
                for j in range(i + 1, self.n_classes):
                    mask = (y == i) | (y == j)
                    X_pair = X[mask]
                    y_pair = y[mask]
                    
                    # Binary: i=0, j=1
                    y_binary = (y_pair == j).astype(int)
                    
                    clf = LogisticRegressionGD(
                        learning_rate=self.learning_rate,
                        n_iterations=self.n_iterations
                    )
                    clf.fit(X_pair, y_binary)
                    self.classifiers[(i, j)] = clf
            
            self.training_time = time.time() - start_time
        
        def predict(self, X):
            m = X.shape[0]
            votes = np.zeros((m, self.n_classes))
            
            for (i, j), clf in self.classifiers.items():
                preds = clf.predict(X)
                votes[preds == 0, i] += 1
                votes[preds == 1, j] += 1
            
            return np.argmax(votes, axis=1)


class LinearDiscriminantAnalysis:
    """Linear Discriminant Analysis - Generative Classifier"""
    
    def __init__(self):
        self.classes = None
        self.means = {}
        self.priors = {}
        self.shared_cov = None
        self.shared_cov_inv = None
        self.training_time = 0
        
    def fit(self, X, y):
        start_time = time.time()
        
        self.classes = np.unique(y)
        n_samples, n_features = X.shape
        n_classes = len(self.classes)
        
        # Compute class statistics
        overall_mean = np.mean(X, axis=0)
        shared_cov_sum = np.zeros((n_features, n_features))
        
        for k in self.classes:
            X_k = X[y == k]
            n_k = X_k.shape[0]
            
            # Mean of class k
            self.means[k] = np.mean(X_k, axis=0)
            
            # Prior probability
            self.priors[k] = n_k / n_samples
            
            # Centered data
            X_k_centered = X_k - self.means[k]
            
            # Contribution to shared covariance
            shared_cov_sum += X_k_centered.T @ X_k_centered
        
        # Shared covariance matrix
        self.shared_cov = shared_cov_sum / (n_samples - n_classes)
        
        # Add regularization for numerical stability
        self.shared_cov += np.eye(n_features) * 1e-4
        
        # Precompute inverse
        self.shared_cov_inv = np.linalg.inv(self.shared_cov)
        
        self.training_time = time.time() - start_time
        
    def predict_proba(self, X):
        """Compute posterior probabilities"""
        n_samples = X.shape[0]
        n_classes = len(self.classes)
        scores = np.zeros((n_samples, n_classes))
        
        for idx, k in enumerate(self.classes):
            # Discriminant function
            mean_k = self.means[k]
            scores[:, idx] = (
                X @ self.shared_cov_inv @ mean_k
                - 0.5 * mean_k @ self.shared_cov_inv @ mean_k
                + np.log(self.priors[k])
            )
        
        # Convert to probabilities using softmax
        scores_shifted = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores_shifted)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        return probs
    
    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes[np.argmax(probs, axis=1)]
    
    def compute_fisher_ratios(self, X, y):
        """
        Compute Fisher Discriminant Ratio for each feature.
        Higher value = more discriminative.
        """
        n_features = X.shape[1]
        n_samples = X.shape[0]
        fisher_ratios = np.zeros(n_features)
        
        for j in range(n_features):
            # Overall mean for feature j
            mu_j = np.mean(X[:, j])
            
            # Between-class variance
            between_var = 0
            for k in self.classes:
                X_k = X[y == k, j]
                n_k = len(X_k)
                mu_kj = np.mean(X_k)
                between_var += n_k * (mu_kj - mu_j) ** 2
            
            # Within-class variance
            within_var = 0
            for k in self.classes:
                X_k = X[y == k, j]
                mu_kj = np.mean(X_k)
                within_var += np.sum((X_k - mu_kj) ** 2)
            
            # Fisher ratio
            if within_var > 0:
                fisher_ratios[j] = between_var / within_var
            else:
                fisher_ratios[j] = 0
        
        return fisher_ratios
    
    def transform(self, X, n_components=None):
        """
        Project data to lower dimensional space.
        Maximum dimensions: K-1 where K is number of classes.
        """
        n_classes = len(self.classes)
        if n_components is None:
            n_components = min(X.shape[1], n_classes - 1)
        else:
            n_components = min(n_components, n_classes - 1)
        
        # Compute between-class scatter matrix
        overall_mean = np.zeros(X.shape[1])
        for k in self.classes:
            overall_mean += self.priors[k] * self.means[k]
        
        S_B = np.zeros((X.shape[1], X.shape[1]))
        for k in self.classes:
            mean_diff = (self.means[k] - overall_mean).reshape(-1, 1)
            S_B += self.priors[k] * (mean_diff @ mean_diff.T)
        
        # Solve generalized eigenvalue problem: S_B w = λ S_W w
        # S_W is the shared covariance
        try:
            eigenvalues, eigenvectors = np.linalg.eig(
                self.shared_cov_inv @ S_B
            )
            
            # Sort by eigenvalue
            idx = np.argsort(eigenvalues)[::-1]
            eigenvectors = eigenvectors[:, idx]
            eigenvalues = eigenvalues[idx]
            
            # Take top n_components
            W = eigenvectors[:, :n_components].real
            
            # Project
            return X @ W
        except:
            # Fallback: use SVD
            U, s, Vt = np.linalg.svd(S_B)
            W = Vt[:n_components].T
            return X @ W


class QuadraticDiscriminantAnalysis:
    """Quadratic Discriminant Analysis with class-specific covariances"""
    
    def __init__(self):
        self.classes = None
        self.means = {}
        self.priors = {}
        self.covariances = {}
        self.cov_inv = {}
        self.cov_det = {}
        self.training_time = 0
        
    def fit(self, X, y):
        start_time = time.time()
        
        self.classes = np.unique(y)
        n_samples, n_features = X.shape
        
        for k in self.classes:
            X_k = X[y == k]
            n_k = X_k.shape[0]
            
            # Class mean
            self.means[k] = np.mean(X_k, axis=0)
            
            # Prior probability
            self.priors[k] = n_k / n_samples
            
            # Class-specific covariance
            X_k_centered = X_k - self.means[k]
            self.covariances[k] = (X_k_centered.T @ X_k_centered) / (n_k - 1)
            
            # Add regularization for stability
            self.covariances[k] += np.eye(n_features) * 1e-4
            
            # Precompute inverse and determinant
            self.cov_inv[k] = np.linalg.inv(self.covariances[k])
            self.cov_det[k] = np.linalg.det(self.covariances[k])
        
        self.training_time = time.time() - start_time
    
    def predict_proba(self, X):
        """Compute posterior probabilities using quadratic discriminant"""
        n_samples = X.shape[0]
        n_classes = len(self.classes)
        scores = np.zeros((n_samples, n_classes))
        
        for idx, k in enumerate(self.classes):
            # Quadratic discriminant function
            mean_k = self.means[k]
            cov_inv_k = self.cov_inv[k]
            cov_det_k = self.cov_det[k]
            
            # Compute for all samples
            diff = X - mean_k
            mahalanobis = np.sum((diff @ cov_inv_k) * diff, axis=1)
            
            scores[:, idx] = (
                -0.5 * np.log(cov_det_k)
                - 0.5 * mahalanobis
                + np.log(self.priors[k])
            )
        
        # Convert to probabilities
        scores_shifted = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores_shifted)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        return probs
    
    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes[np.argmax(probs, axis=1)]
    
    def count_parameters(self):
        """Count total parameters for comparison with LDA"""
        n_features = len(self.means[self.classes[0]])
        n_classes = len(self.classes)
        
        # Mean vectors: K * D
        mean_params = n_classes * n_features
        
        # Covariance matrices: K * D*(D+1)/2 (symmetric)
        cov_params = n_classes * (n_features * (n_features + 1) // 2)
        
        # Priors: K-1 (sum to 1 constraint)
        prior_params = n_classes - 1
        
        total = mean_params + cov_params + prior_params
        
        return {
            'means': mean_params,
            'covariances': cov_params,
            'priors': prior_params,
            'total': total
        }


class Perceptron:
    """Classic Perceptron Algorithm"""
    
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.w = None
        self.b = None
        self.errors = []
        self.training_time = 0
        
    def fit(self, X, y):
        """
        Train perceptron.
        Expects y in {0, 1}, converts internally to {-1, +1}
        """
        start_time = time.time()
        
        # Convert labels to {-1, +1}
        y_bipolar = 2 * y - 1
        
        n_samples, n_features = X.shape
        
        # Initialize weights
        self.w = np.zeros(n_features)
        self.b = 0
        
        # Training loop
        for iteration in range(self.n_iterations):
            errors = 0
            
            for i in range(n_samples):
                # Prediction
                linear_output = np.dot(self.w, X[i]) + self.b
                y_pred = np.sign(linear_output)
                
                # If misclassified, update
                if y_bipolar[i] * y_pred <= 0:
                    self.w += self.learning_rate * y_bipolar[i] * X[i]
                    self.b += self.learning_rate * y_bipolar[i]
                    errors += 1
            
            self.errors.append(errors)
            
            # Early stopping if no errors
            if errors == 0:
                print(f"Converged at iteration {iteration + 1}")
                break
        
        self.training_time = time.time() - start_time
    
    def predict(self, X):
        """Predict class labels (returns 0 or 1)"""
        linear_output = np.dot(X, self.w) + self.b
        y_pred_bipolar = np.sign(linear_output)
        # Convert back to {0, 1}
        return ((y_pred_bipolar + 1) / 2).astype(int)
    
    def score(self, X, y):
        """Compute accuracy"""
        return np.mean(self.predict(X) == y)


class RegularizedLogisticRegression:
    """Logistic Regression with L1/L2 Regularization and Class Weighting"""
    
    def __init__(self, learning_rate=0.01, n_iterations=1000, 
                 reg_type='none', reg_lambda=0.0, class_weight=None):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.reg_type = reg_type  # 'none', 'l1', 'l2'
        self.reg_lambda = reg_lambda
        self.class_weight = class_weight  # 'balanced' or None
        self.theta = None
        self.losses = []
        self.training_time = 0
        
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def fit(self, X, y):
        start_time = time.time()
        
        m, n = X.shape
        X_bias = np.c_[np.ones((m, 1)), X]
        self.theta = np.zeros(n + 1)
        
        # Compute class weights
        if self.class_weight == 'balanced':
            classes = np.unique(y)
            n_samples = len(y)
            n_classes = len(classes)
            weights = np.ones(m)
            for c in classes:
                n_c = np.sum(y == c)
                weights[y == c] = n_samples / (n_classes * n_c)
        else:
            weights = np.ones(m)
        
        # Gradient descent
        for iteration in range(self.n_iterations):
            # Forward pass
            z = X_bias @ self.theta
            h = self.sigmoid(z)
            
            # Weighted loss
            epsilon = 1e-15
            h_clip = np.clip(h, epsilon, 1 - epsilon)
            loss = -np.mean(weights * (y * np.log(h_clip) + (1 - y) * np.log(1 - h_clip)))
            
            # Add regularization to loss
            if self.reg_type == 'l2':
                loss += (self.reg_lambda / (2 * m)) * np.sum(self.theta[1:]**2)
            elif self.reg_type == 'l1':
                loss += (self.reg_lambda / m) * np.sum(np.abs(self.theta[1:]))
            
            self.losses.append(loss)
            
            # Gradient
            error = h - y
            gradient = (X_bias.T @ (weights * error)) / m
            
            # Add regularization gradient
            if self.reg_type == 'l2':
                gradient[1:] += (self.reg_lambda / m) * self.theta[1:]
            elif self.reg_type == 'l1':
                gradient[1:] += (self.reg_lambda / m) * np.sign(self.theta[1:])
            
            # Update
            self.theta -= self.learning_rate * gradient
        
        self.training_time = time.time() - start_time
    
    def predict_proba(self, X):
        X_bias = np.c_[np.ones((X.shape[0], 1)), X]
        return self.sigmoid(X_bias @ self.theta)
    
    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)
    
    def count_nonzero_weights(self, threshold: float = 1e-6):
        """Count effectively non-zero weights (excluding bias)."""
        return np.sum(np.abs(self.theta[1:]) > threshold)


class ProbitRegression:
    """Probit Regression using Gaussian CDF"""
    
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.theta = None
        self.losses = []
        self.training_time = 0
        
    def probit(self, z):
        """Gaussian CDF (Probit function)"""
        return norm.cdf(z)
    
    def probit_derivative(self, z):
        """Derivative of Gaussian CDF = PDF"""
        return norm.pdf(z)
    
    def fit(self, X, y):
        start_time = time.time()
        
        m, n = X.shape
        X_bias = np.c_[np.ones((m, 1)), X]
        self.theta = np.zeros(n + 1)
        
        for iteration in range(self.n_iterations):
            z = X_bias @ self.theta
            h = self.probit(z)
            
            # Log-likelihood loss
            epsilon = 1e-15
            h_clip = np.clip(h, epsilon, 1 - epsilon)
            loss = -np.mean(y * np.log(h_clip) + (1 - y) * np.log(1 - h_clip))
            self.losses.append(loss)
            
            # Gradient using chain rule
            # d/dθ = (d/dz) * (dz/dθ) where d/dz uses PDF
            pdf_z = self.probit_derivative(z)
            error = (h - y) / (h * (1 - h) + epsilon)  # Rescale by inverse variance
            gradient = (X_bias.T @ (error * pdf_z)) / m
            
            self.theta -= self.learning_rate * gradient
        
        self.training_time = time.time() - start_time
    
    def predict_proba(self, X):
        X_bias = np.c_[np.ones((X.shape[0], 1)), X]
        return self.probit(X_bias @ self.theta)
    
    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


class LogisticRegressionLaplace:
    """Logistic Regression with Laplace Approximation for Uncertainty"""
    
    def __init__(self, learning_rate=0.01, n_iterations=1000, prior_variance=10.0):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.prior_variance = prior_variance
        self.theta_map = None
        self.hessian_inv = None
        self.training_time = 0
        
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def fit(self, X, y):
        start_time = time.time()
        
        m, n = X.shape
        X_bias = np.c_[np.ones((m, 1)), X]
        
        # Find MAP estimate using gradient descent with L2 regularization
        self.theta_map = np.zeros(n + 1)
        lambda_reg = 1.0 / self.prior_variance
        
        for iteration in range(self.n_iterations):
            z = X_bias @ self.theta_map
            h = self.sigmoid(z)
            
            # Gradient of log-posterior (likelihood + prior)
            error = h - y
            gradient = (X_bias.T @ error) / m + lambda_reg * self.theta_map
            
            self.theta_map -= self.learning_rate * gradient
        
        # Compute Hessian at MAP estimate
        z_map = X_bias @ self.theta_map
        h_map = self.sigmoid(z_map)
        
        # R = diag(h * (1-h))
        R = h_map * (1 - h_map)
        
        # Hessian: H = X^T R X + λI
        # For numerical stability, compute X_weighted = sqrt(R) * X
        X_weighted = X_bias * np.sqrt(R)[:, np.newaxis]
        H = X_weighted.T @ X_weighted / m + lambda_reg * np.eye(n + 1)
        
        # Compute inverse (posterior covariance)
        try:
            self.hessian_inv = np.linalg.inv(H)
        except:
            # Add regularization if singular
            self.hessian_inv = np.linalg.inv(H + 1e-5 * np.eye(n + 1))
        
        self.training_time = time.time() - start_time
    
    def predict_proba(self, X):
        """Point estimate (MAP)"""
        X_bias = np.c_[np.ones((X.shape[0], 1)), X]
        return self.sigmoid(X_bias @ self.theta_map)
    
    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)
    
    def predict_with_uncertainty(self, X):
        """
        Predict with uncertainty quantification.
        Returns: probabilities, standard deviations
        """
        X_bias = np.c_[np.ones((X.shape[0], 1)), X]
        
        # Mean prediction
        z_mean = X_bias @ self.theta_map
        
        # Variance of latent variable
        z_var = np.sum((X_bias @ self.hessian_inv) * X_bias, axis=1)
        
        # Approximate probability (probit approximation)
        kappa = 1 / np.sqrt(1 + np.pi * z_var / 8)
        probs = self.sigmoid(kappa * z_mean)
        
        # Standard deviation in probability space (linearization)
        std_devs = np.sqrt(z_var) * probs * (1 - probs)
        
        return probs, std_devs


class KernelLogisticRegression:
    """Logistic Regression with RBF Kernel (Dual Formulation)"""
    
    def __init__(self, learning_rate=0.01, n_iterations=500, 
                 kernel='rbf', gamma=1.0, reg_lambda=0.01):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.kernel = kernel
        self.gamma = gamma  # 1/(2*sigma^2) for RBF
        self.reg_lambda = reg_lambda
        self.alpha = None
        self.X_train = None
        self.y_train = None
        self.K = None
        self.losses = []
        self.training_time = 0
        
    def rbf_kernel(self, X1, X2):
        """
        Compute RBF kernel matrix.
        K[i,j] = exp(-gamma * ||x_i - x_j||^2)
        """
        # Compute pairwise squared distances
        X1_sq = np.sum(X1**2, axis=1).reshape(-1, 1)
        X2_sq = np.sum(X2**2, axis=1).reshape(1, -1)
        distances_sq = X1_sq + X2_sq - 2 * X1 @ X2.T
        
        return np.exp(-self.gamma * distances_sq)
    
    def compute_kernel(self, X1, X2):
        """Compute kernel matrix"""
        if self.kernel == 'rbf':
            return self.rbf_kernel(X1, X2)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def fit(self, X, y):
        start_time = time.time()
        
        self.X_train = X.copy()
        self.y_train = y.copy()
        m = X.shape[0]
        
        # Compute kernel Gram matrix
        self.K = self.compute_kernel(X, X)
        
        # Initialize dual variables
        self.alpha = np.zeros(m)
        
        # Gradient descent on alpha
        for iteration in range(self.n_iterations):
            # Predictions: f(x_i) = sum_j alpha_j K_ij
            f = self.K @ self.alpha
            h = self.sigmoid(f)
            
            # Loss
            epsilon = 1e-15
            h_clip = np.clip(h, epsilon, 1 - epsilon)
            loss = -np.mean(y * np.log(h_clip) + (1 - y) * np.log(1 - h_clip))
            loss += (self.reg_lambda / 2) * np.sum(self.alpha * (self.K @ self.alpha))
            self.losses.append(loss)
            
            # Gradient w.r.t. alpha
            error = h - y
            gradient = (self.K @ error) / m + self.reg_lambda * (self.K @ self.alpha)
            
            self.alpha -= self.learning_rate * gradient
        
        self.training_time = time.time() - start_time
    
    def predict_proba(self, X):
        """Predict probabilities for new data"""
        K_test = self.compute_kernel(X, self.X_train)
        f = K_test @ self.alpha
        return self.sigmoid(f)
    
    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


class GaussianNaiveBayes:
    """Gaussian Naive Bayes with diagonal covariance assumption"""
    
    def __init__(self):
        self.classes = None
        self.means = {}
        self.variances = {}
        self.priors = {}
        self.training_time = 0
        
    def fit(self, X, y):
        start_time = time.time()
        
        self.classes = np.unique(y)
        n_samples, n_features = X.shape
        
        for k in self.classes:
            X_k = X[y == k]
            n_k = X_k.shape[0]
            
            # Mean of each feature for class k
            self.means[k] = np.mean(X_k, axis=0)
            
            # Variance of each feature for class k (diagonal only)
            self.variances[k] = np.var(X_k, axis=0) + 1e-6  # Add small value for stability
            
            # Prior probability
            self.priors[k] = n_k / n_samples
        
        self.training_time = time.time() - start_time
    
    def predict_proba(self, X):
        """Compute posterior probabilities"""
        n_samples = X.shape[0]
        n_classes = len(self.classes)
        scores = np.zeros((n_samples, n_classes))
        
        for idx, k in enumerate(self.classes):
            mean_k = self.means[k]
            var_k = self.variances[k]
            
            # Log probability under Gaussian (independent features)
            # log p(x|y=k) = sum_j [-0.5 * log(2π σ²) - (x_j - μ_j)² / (2σ²)]
            log_prob = -0.5 * np.sum(np.log(2 * np.pi * var_k))
            log_prob -= 0.5 * np.sum(((X - mean_k) ** 2) / var_k, axis=1)
            
            # Add log prior
            scores[:, idx] = log_prob + np.log(self.priors[k])
        
        # Convert to probabilities using softmax
        scores_shifted = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores_shifted)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        return probs
    
    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes[np.argmax(probs, axis=1)]
    
    def get_covariance_structure(self, class_idx):
        """Return the diagonal covariance matrix for visualization"""
        return np.diag(self.variances[class_idx])



# ============================================================================
# MODULE METADATA
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("ML Classification Utils Module")
    print("=" * 80)
    print("\nAvailable Models:")
    print("  1. LogisticRegressionGD - Gradient Descent")
    print("  2. LogisticRegressionNewton - Newton-Raphson/IRLS")
    print("  3. SoftmaxRegression - Multi-class Direct")
    print("  4. OneVsRestClassifier - One-vs-Rest")
    print("  5. OneVsOneClassifier - One-vs-One")
    print("  6. LinearDiscriminantAnalysis - LDA")
    print("  7. QuadraticDiscriminantAnalysis - QDA")
    print("  8. Perceptron - Classic Perceptron")
    print("  9. RegularizedLogisticRegression - L1/L2 Regularization")
    print(" 10. ProbitRegression - Gaussian CDF")
    print(" 11. LogisticRegressionLaplace - Bayesian with Uncertainty")
    print(" 12. KernelLogisticRegression - RBF Kernel")
    print(" 13. GaussianNaiveBayes - Naive Bayes")
    print("\nModule loaded successfully!")
