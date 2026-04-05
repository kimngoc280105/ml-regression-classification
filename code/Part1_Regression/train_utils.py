"""
train_utils.py — Training utilities for Part 1: Regression
Contains helper functions for Ridge, Lasso, Elastic Net, and feature selection.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LassoCV, lasso_path
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error


def select_best_lambda_cv(X_train, y_train, model_class, alphas, k=10):
    """
    Grid Search + K-Fold CV để chọn lambda tối ưu.
    Return: (best_alpha, best_rmse, results)
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    results = []

    for alpha in alphas:
        model = model_class(alpha=alpha, max_iter=10000)
        neg_mse = cross_val_score(
            model, X_train, y_train,
            cv=kf,
            scoring='neg_mean_squared_error'
        )
        rmse_folds = np.sqrt(-neg_mse)
        results.append((alpha, rmse_folds.mean(), rmse_folds.std()))

    best_idx = np.argmin([r[1] for r in results])
    best_alpha = results[best_idx][0]
    best_rmse = results[best_idx][1]
    return best_alpha, best_rmse, results


def select_best_lambda_lasso_path(X_train, y_train, alphas, k=10):
    """
    Dùng LassoCV với built-in warm-start.
    Return: (best_alpha, results, fitted_model)
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    model = LassoCV(
        alphas=alphas,
        cv=kf,
        max_iter=10000,
        random_state=42
    )
    model.fit(X_train, y_train)

    mean_rmse = np.sqrt(model.mse_path_.mean(axis=1))
    std_rmse = model.mse_path_.std(axis=1)
    results = list(zip(model.alphas_, mean_rmse, std_rmse))

    return model.alpha_, results, model


def plot_regularization_path(X_train, y_train, alphas, feature_names,
                              model_class=Ridge, best_alpha=None):
    """
    Vẽ regularization path với feature names thực tế.
    - Ridge: fit tuần tự với alphas
    - Lasso: dùng sklearn.linear_model.lasso_path (built-in warm-start)
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    if model_class == Lasso:
        # lasso_path() yêu cầu alphas sắp xếp giảm dần
        alphas_sorted = np.sort(alphas)[::-1]
        alphas_out, coefs, _ = lasso_path(X_train, y_train, alphas=alphas_sorted)
        
        # coefs shape: (n_features, n_alphas)
        for i, name in enumerate(feature_names):
            ax.plot(np.log10(alphas_out), coefs[i], marker='o', markersize=3, label=name)

        ax.set_title('Lasso Regularization Path (Warm-start via lasso_path)', fontsize=13, fontweight='bold')

    else:  # Ridge
        coefs = []
        for alpha in alphas:
            model = Ridge(alpha=alpha, max_iter=10000)
            model.fit(X_train, y_train)
            coefs.append(model.coef_.copy())

        coefs = np.array(coefs)  # (n_alphas, n_features)
        for i, name in enumerate(feature_names):
            ax.plot(np.log10(alphas), coefs[:, i], marker='o', markersize=3, label=name)

        ax.set_title('Ridge Regularization Path', fontsize=13, fontweight='bold')

    # Đánh dấu best alpha nếu có
    if best_alpha is not None:
        ax.axvline(np.log10(best_alpha), color='red', linestyle='--', linewidth=2,
                   label=f'Best α = {best_alpha:.4f}', zorder=10)

    ax.set_xlabel('log10(λ)', fontsize=11)
    ax.set_ylabel('Coefficient value', fontsize=11)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()  # trái = ít regularization, phải = nhiều
    plt.tight_layout()
    plt.show()



def forward_stepwise_selection(X, y, feature_names):
    """Forward Stepwise Selection."""
    selected_features = []
    remaining_features = list(range(X.shape[1]))
    best_scores = []

    while remaining_features:
        best_feature = None
        best_score = float('inf')

        for feature_idx in remaining_features:
            current_features = selected_features + [feature_idx]
            X_subset = X[:, current_features]

            # Fit OLS
            X_b = np.c_[np.ones(X_subset.shape[0]), X_subset]
            theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
            y_pred = X_b @ theta
            rmse = np.sqrt(mean_squared_error(y, y_pred))

            if rmse < best_score:
                best_score = rmse
                best_feature = feature_idx

        if best_feature is not None:
            selected_features.append(best_feature)
            remaining_features.remove(best_feature)
            best_scores.append(best_score)
        else:
            break

    selected_names = [feature_names[i] for i in selected_features]
    return selected_features, selected_names, best_scores


def backward_elimination(X, y, feature_names):
    """Backward Elimination."""
    current_features = list(range(X.shape[1]))
    best_scores = []

    while len(current_features) > 1:
        worst_feature = None
        best_score = float('inf')

        for feature_idx in current_features:
            temp_features = [f for f in current_features if f != feature_idx]
            X_subset = X[:, temp_features]

            # Fit OLS
            X_b = np.c_[np.ones(X_subset.shape[0]), X_subset]
            theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
            y_pred = X_b @ theta
            rmse = np.sqrt(mean_squared_error(y, y_pred))

            if rmse < best_score:
                best_score = rmse
                worst_feature = feature_idx

        if worst_feature is not None:
            current_features.remove(worst_feature)
            best_scores.append(best_score)
        else:
            break

    selected_names = [feature_names[i] for i in current_features]
    return current_features, selected_names, best_scores


def lasso_feature_selection(X_train, y_train, alpha, feature_names):
    """Feature selection using Lasso coefficients."""
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_train, y_train)
    selected_indices = np.where(lasso.coef_ != 0)[0]
    selected_names = [feature_names[i] for i in selected_indices]
    return selected_indices, selected_names, lasso.coef_


from sklearn.linear_model import ElasticNet
from sklearn.model_selection import KFold, cross_val_score

def plot_elastic_net_heatmap(X_train, y_train, alphas_en, l1_ratios, k=10):
    """
    Grid Search + K-Fold CV cho Elastic Net.
    Bỏ X_val, y_val → dùng CV trên toàn X_train.
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    results_en = []

    for α in alphas_en:
        for l1_r in l1_ratios:
            en = ElasticNet(alpha=α, l1_ratio=l1_r, max_iter=10000)
            neg_mse = cross_val_score(
                en, X_train, y_train,
                cv=kf,
                scoring='neg_mean_squared_error'
            )
            mean_rmse = np.sqrt(-neg_mse).mean()
            std_rmse  = np.sqrt(-neg_mse).std()
            results_en.append((α, l1_r, mean_rmse, std_rmse))

    best_en  = min(results_en, key=lambda x: x[2])

    # Tạo matrix
    rmse_matrix = np.array([r[2] for r in results_en]).reshape(
        len(alphas_en), len(l1_ratios)
    )

    # Vẽ heatmap
    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(rmse_matrix, cmap='viridis', aspect='auto', origin='lower')
    plt.colorbar(im, ax=ax, label='CV RMSE')

    # Label log scale cho trục Y
    ax.set_xticks(range(len(l1_ratios)))
    ax.set_xticklabels([f'{r:.1f}' for r in l1_ratios])
    ax.set_yticks(range(len(alphas_en)))
    ax.set_yticklabels([f'{a:.4g}' for a in alphas_en])   # ← gọn hơn
    ax.set_xlabel('l1_ratio  (0=Ridge, 1=Lasso)', fontsize=11)
    ax.set_ylabel('λ (alpha)', fontsize=11)
    ax.set_title(f'Elastic Net Grid Search — {k}-Fold CV RMSE', 
                 fontsize=13, fontweight='bold')

    # Annotation RMSE vào từng ô
    for i in range(len(alphas_en)):
        for j in range(len(l1_ratios)):
            ax.text(j, i, f'{rmse_matrix[i,j]:,.0f}',
                    ha='center', va='center',
                    fontsize=7,
                    color='white' if rmse_matrix[i,j] > rmse_matrix.mean() else 'black')

    # Đánh dấu best
    best_i = list(alphas_en).index(best_en[0])
    best_j = list(l1_ratios).index(best_en[1])
    ax.plot(best_j, best_i, 'r*', markersize=20,
            markeredgecolor='white', markeredgewidth=1.5,
            label=f'Best: λ={best_en[0]:.4g}, l1={best_en[1]:.1f}, RMSE={best_en[2]:,.0f}')
    ax.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    plt.show()

    return rmse_matrix, best_en, results_en

def plot_feature_selection_comparison(fwd_names, fwd_scores, bwd_names, bwd_scores, 
                                      lasso_names, lasso_rmse):
    """
    Vẽ so sánh 3 phương pháp feature selection.
    """
    methods = ['Forward Stepwise', 'Backward Elimination', 'Lasso-based']
    n_feats = [len(fwd_names), len(bwd_names), len(lasso_names)]
    rmses = [fwd_scores[-1] if fwd_scores else 0, 
             bwd_scores[-1] if bwd_scores else 0, 
             lasso_rmse]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Number of features
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    bars1 = axes[0].bar(methods, n_feats, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[0].set_ylabel('# Features selected', fontsize=11)
    axes[0].set_title('Feature Selection: Số lượng features', fontsize=12, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Thêm giá trị trên các bar
    for bar, n in zip(bars1, n_feats):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(n)}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Plot 2: RMSE performance
    bars2 = axes[1].bar(methods, rmses, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[1].set_ylabel('Validation RMSE', fontsize=11)
    axes[1].set_title('Feature Selection: Performance', fontsize=12, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    
    # Thêm giá trị trên các bar
    for bar, rmse in zip(bars2, rmses):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{rmse:,.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.show()