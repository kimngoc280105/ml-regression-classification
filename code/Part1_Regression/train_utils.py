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






# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# C.2 Ridge và Lasso Regression
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# C.3 Hàm cơ sở phi tuyến tính 
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer


class GaussianRBF(BaseEstimator, TransformerMixin):
    """
    Gaussian Radial Basis Function (RBF) Transform
    φ(x) = exp(-γ · ||x - cᵢ||²)
    
    params
    ------
    n_centers : int (default=20)
        Số điểm neo (chọn ngẫu nhiên từ train set)
    gamma : float (default=1.0)
        Độ rộng của Gaussian kernel
    random_state : int
        Seed cho reproducibility
    """
    def __init__(self, n_centers=20, gamma=1.0, random_state=42):
        self.n_centers = n_centers
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None):
        rng = np.random.RandomState(self.random_state)
        idx = rng.choice(len(X), size=self.n_centers, replace=False)
        self.centers_ = X[idx]
        return self

    def transform(self, X):
        # Shape: (n_samples, n_centers)
        # ||x - c||² = Σ(x_i - c_i)²
        distances_sq = np.sum((X[:, None, :] - self.centers_[None, :, :]) ** 2, axis=2)
        return np.exp(-self.gamma * distances_sq)

    def get_feature_names_out(self, input_features=None):
        return np.array([f'RBF_center_{i}' for i in range(self.n_centers)])


class LogFeatures(BaseEstimator, TransformerMixin):
    """
    Log Transform Features: thêm log(x+1) cho các features skewed/positional.
    
    params
    ------
    feature_indices : list or None
        Index của features cần log transform. 
        None = áp dụng cho tất cả features
    """
    def __init__(self, feature_indices=None):
        self.feature_indices = feature_indices

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.feature_indices is None:
            log_X = np.log1p(np.abs(X))
        else:
            log_X = np.log1p(np.abs(X[:, self.feature_indices]))
        return np.hstack([X, log_X])

    def get_feature_names_out(self, input_features=None):
        n_orig = 8 if self.feature_indices is None else 8 + len(self.feature_indices)
        return np.array([f'feature_{i}' for i in range(n_orig)])


class SigmoidBasis(BaseEstimator, TransformerMixin):
    """
    Sigmoid Basis Function (Logistic Basis): φ(x) = 1 / (1 + exp(-β·(x - cᵢ)))
    
    Tạo non-linear basis bằng cách áp dụng sigmoid function
    với các center điểm được chọn từ dữ liệu huấn luyện.
    
    params
    ------
    n_centers : int (default=15)
        Số center điểm (chọn ngẫu nhiên từ train set)
    beta : float (default=1.0)
        Slope parameter của sigmoid function
    random_state : int
        Seed cho reproducibility
    """
    def __init__(self, n_centers=15, beta=1.0, random_state=42):
        self.n_centers = n_centers
        self.beta = beta
        self.random_state = random_state

    def fit(self, X, y=None):
        rng = np.random.RandomState(self.random_state)
        idx = rng.choice(len(X), size=self.n_centers, replace=False)
        self.centers_ = X[idx]
        return self

    def transform(self, X):
        # σ(β·(x - c)) = 1 / (1 + exp(-β·(x - c)))
        # Shape: (n_samples, n_features, n_centers)
        distances = X[:, :, None] - self.centers_.T[None, :, :]  # (n_samples, n_features, n_centers)
        # Apply sigmoid per dimension
        sigmoid = 1.0 / (1.0 + np.exp(-self.beta * distances))
        # Average across feature dimension: (n_samples, n_centers)
        return np.mean(sigmoid, axis=1)

    def get_feature_names_out(self, input_features=None):
        return np.array([f'Sigmoid_basis_{i}' for i in range(self.n_centers)])


class SplineBasis(BaseEstimator, TransformerMixin):
    """
    Spline Basis Function: Cubic spline interpolation
    
    Wrapper around sklearn.preprocessing.SplineTransformer
    với tự động lựa chọn knots. Không cần tuning parameter riêng
    như RBF/Sigmoid's center và gamma/beta.
    
    params
    ------
    n_knots : int (default=5)
        Số knot điểm cho spline (tự động phân phối đều trên dữ liệu)
    degree : int (default=3)
        Bậc của spline polynomial (3 = cubic)
    """
    def __init__(self, n_knots=5, degree=3, include_bias=False):
        self.n_knots = n_knots
        self.degree = degree
        self.include_bias = include_bias
        self.transformer_ = None

    def fit(self, X, y=None):
        # SplineTransformer tự động tính knots từ dữ liệu
        self.transformer_ = SplineTransformer(
            n_knots=self.n_knots,
            degree=self.degree,
            include_bias=self.include_bias
        )
        self.transformer_.fit(X)
        return self

    def transform(self, X):
        return self.transformer_.transform(X)

    def get_feature_names_out(self, input_features=None):
        return self.transformer_.get_feature_names_out(input_features)


def apply_basis_function(X_train, X_val, X_test, basis_type='polynomial', **kwargs):
    """
    Áp dụng non-linear basis function.
    returns: (X_train_basis, X_val_basis, X_test_basis, transformer)
    """
    if basis_type == 'polynomial':
        degree = kwargs.get('degree', 2)
        transformer = PolynomialFeatures(degree=degree, include_bias=False)
        X_train_basis = transformer.fit_transform(X_train)
        X_val_basis = transformer.transform(X_val)
        X_test_basis = transformer.transform(X_test)
        
    elif basis_type == 'rbf':
        n_centers = kwargs.get('n_centers', 20)
        gamma = kwargs.get('gamma', 1.0)
        transformer = GaussianRBF(n_centers=n_centers, gamma=gamma)
        X_train_basis = transformer.fit_transform(X_train)
        X_val_basis = transformer.transform(X_val)
        X_test_basis = transformer.transform(X_test)
        
    elif basis_type == 'sigmoid':
        n_centers = kwargs.get('n_centers', 15)
        beta = kwargs.get('beta', 1.0)
        transformer = SigmoidBasis(n_centers=n_centers, beta=beta)
        X_train_basis = transformer.fit_transform(X_train)
        X_val_basis = transformer.transform(X_val)
        X_test_basis = transformer.transform(X_test)
        
    elif basis_type == 'spline':
        n_knots = kwargs.get('n_knots', 5)
        degree = kwargs.get('degree', 3)
        transformer = SplineBasis(n_knots=n_knots, degree=degree)
        X_train_basis = transformer.fit_transform(X_train)
        X_val_basis = transformer.transform(X_val)
        X_test_basis = transformer.transform(X_test)
        
    elif basis_type == 'log':
        feature_indices = kwargs.get('feature_indices', None)
        transformer = LogFeatures(feature_indices=feature_indices)
        X_train_basis = transformer.fit_transform(X_train)
        X_val_basis = transformer.transform(X_val)
        X_test_basis = transformer.transform(X_test)
    else:
        raise ValueError(f"Unknown basis_type: {basis_type}")

    return X_train_basis, X_val_basis, X_test_basis, transformer



def plot_validation_curve_polynomial(X_train, y_train, X_val, y_val, degrees=[1, 2, 3, 4, 5],
                                       model_class=Ridge, alpha=1.0):
    """
    Vẽ validation curve: RMSE theo bậc đa thức.
    returns: results_df : DataFrame (RMSE cho mỗi degree)
    """
    from sklearn.preprocessing import PolynomialFeatures
    
    train_rmses = []
    val_rmses = []
    n_features = []
    
    for degree in degrees:
        # Transform
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_train_poly = poly.fit_transform(X_train)
        X_val_poly = poly.transform(X_val)
        
        # Fit model
        model = model_class(alpha=alpha, max_iter=10000)
        model.fit(X_train_poly, y_train)
        
        # Evaluate
        train_rmse = np.sqrt(mean_squared_error(y_train, model.predict(X_train_poly)))
        val_rmse = np.sqrt(mean_squared_error(y_val, model.predict(X_val_poly)))
        
        train_rmses.append(train_rmse)
        val_rmses.append(val_rmse)
        n_features.append(X_train_poly.shape[1])
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(degrees, train_rmses, 'o-', linewidth=2, markersize=8, label='Train RMSE', color='#1f77b4')
    plt.plot(degrees, val_rmses, 's-', linewidth=2, markersize=8, label='Validation RMSE', color='#ff7f0e')
    
    plt.xlabel('Bậc đa thức', fontsize=11, fontweight='bold')
    plt.ylabel('RMSE', fontsize=11, fontweight='bold')
    plt.title('Validation Curve: RMSE theo bậc đa thức', fontsize=13, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Secondary axis: số features
    def degree_to_n_features(x):
        x = np.asarray(x)
        if np.ndim(x) == 0:  # scalar
            idx = int(x)
            return n_features[idx] if idx < len(n_features) else 0
        else:  # array
            return np.array([n_features[int(idx)] if int(idx) < len(n_features) else 0 for idx in x])
    
    ax2 = plt.gca().secondary_xaxis('top', functions=(degree_to_n_features, lambda y: y))
    ax2.set_xlabel('# Features', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    # Return results table
    results_df = pd.DataFrame({
        'Degree': degrees,
        'Num_Features': n_features,
        'Train_RMSE': train_rmses,
        'Val_RMSE': val_rmses,
        'Overfitting_Gap': [val - train for train, val in zip(train_rmses, val_rmses)]
    })
    
    return results_df


def plot_validation_curve_rbf(X_train, y_train, X_val, y_val, n_centers_list=[5, 10, 15, 20, 30, 50],
                               gamma=1.0, model_class=Ridge, alpha=1.0):
    """
    Vẽ validation curve: RMSE theo số RBF centers.
    returns: results_df : DataFrame
        RMSE cho mỗi n_centers
    """
    train_rmses = []
    val_rmses = []
    
    for n_centers in n_centers_list:
        # Transform
        rbf = GaussianRBF(n_centers=n_centers, gamma=gamma)
        X_train_rbf = rbf.fit_transform(X_train)
        X_val_rbf = rbf.transform(X_val)
        
        # Fit model
        model = model_class(alpha=alpha, max_iter=10000)
        model.fit(X_train_rbf, y_train)
        
        # Evaluate
        train_rmse = np.sqrt(mean_squared_error(y_train, model.predict(X_train_rbf)))
        val_rmse = np.sqrt(mean_squared_error(y_val, model.predict(X_val_rbf)))
        
        train_rmses.append(train_rmse)
        val_rmses.append(val_rmse)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(n_centers_list, train_rmses, 'o-', linewidth=2, markersize=8, label='Train RMSE', color='#2ca02c')
    plt.plot(n_centers_list, val_rmses, 's-', linewidth=2, markersize=8, label='Validation RMSE', color='#d62728')
    
    plt.xlabel('Số hàm cơ sở', fontsize=11, fontweight='bold')
    plt.ylabel('RMSE', fontsize=11, fontweight='bold')
    plt.title(f'Validation Curve: RMSE theo số hàm cơ sở RBF (γ={gamma})', fontsize=13, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(n_centers_list)
    
    plt.tight_layout()
    plt.show()
    
    # Return results table
    results_df = pd.DataFrame({
        'N_Centers': n_centers_list,
        'Train_RMSE': train_rmses,
        'Val_RMSE': val_rmses,
        'Overfitting_Gap': [val - train for train, val in zip(train_rmses, val_rmses)]
    })
    
    return results_df


def plot_validation_curve_sigmoid(X_train, y_train, X_val, y_val, n_centers_list=[5, 10, 15, 20, 25, 30],
                                   beta=1.0, model_class=Ridge, alpha=1.0):
    """
    Vẽ validation curve: RMSE theo số Sigmoid Basis centers.
    returns results_df : DataFrame
        RMSE cho mỗi n_centers
    """
    train_rmses = []
    val_rmses = []
    
    for n_centers in n_centers_list:
        # Transform
        sigmoid = SigmoidBasis(n_centers=n_centers, beta=beta)
        X_train_sigmoid = sigmoid.fit_transform(X_train)
        X_val_sigmoid = sigmoid.transform(X_val)
        
        # Fit model
        model = model_class(alpha=alpha, max_iter=10000)
        model.fit(X_train_sigmoid, y_train)
        
        # Evaluate
        train_rmse = np.sqrt(mean_squared_error(y_train, model.predict(X_train_sigmoid)))
        val_rmse = np.sqrt(mean_squared_error(y_val, model.predict(X_val_sigmoid)))
        
        train_rmses.append(train_rmse)
        val_rmses.append(val_rmse)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(n_centers_list, train_rmses, 'o-', linewidth=2, markersize=8, label='Train RMSE', color='#9467bd')
    plt.plot(n_centers_list, val_rmses, 's-', linewidth=2, markersize=8, label='Validation RMSE', color='#e377c2')
    
    plt.xlabel('Số hàm cơ sở', fontsize=11, fontweight='bold')
    plt.ylabel('RMSE', fontsize=11, fontweight='bold')
    plt.title(f'Validation Curve: RMSE theo số hàm cơ sở sigmoid (β={beta})', fontsize=13, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(n_centers_list)
    
    plt.tight_layout()
    plt.show()
    
    # Return results table
    results_df = pd.DataFrame({
        'N_Centers': n_centers_list,
        'Train_RMSE': train_rmses,
        'Val_RMSE': val_rmses,
        'Overfitting_Gap': [val - train for train, val in zip(train_rmses, val_rmses)]
    })
    
    return results_df


def plot_validation_curve_spline(X_train, y_train, X_val, y_val, n_knots_list=[3, 4, 5, 6, 7, 8, 10],
                                  degree=3, model_class=Ridge, alpha=1.0):
    """
    Vẽ validation curve: RMSE theo số Spline knots.
    returns results_df : DataFrame
        RMSE cho mỗi n_knots
    """
    train_rmses = []
    val_rmses = []
    
    for n_knots in n_knots_list:
        # Transform
        spline = SplineBasis(n_knots=n_knots, degree=degree)
        X_train_spline = spline.fit_transform(X_train)
        X_val_spline = spline.transform(X_val)
        
        # Fit model
        model = model_class(alpha=alpha, max_iter=10000)
        model.fit(X_train_spline, y_train)
        
        # Evaluate
        train_rmse = np.sqrt(mean_squared_error(y_train, model.predict(X_train_spline)))
        val_rmse = np.sqrt(mean_squared_error(y_val, model.predict(X_val_spline)))
        
        train_rmses.append(train_rmse)
        val_rmses.append(val_rmse)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(n_knots_list, train_rmses, 'o-', linewidth=2, markersize=8, label='Train RMSE', color='#9467bd')
    plt.plot(n_knots_list, val_rmses, 's-', linewidth=2, markersize=8, label='Validation RMSE', color='#e377c2')
    
    plt.xlabel('Số knot điểm', fontsize=11, fontweight='bold')
    plt.ylabel('RMSE', fontsize=11, fontweight='bold')
    plt.title(f'Validation Curve: RMSE theo số Spline knots (degree={degree})', fontsize=13, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(n_knots_list)
    
    plt.tight_layout()
    plt.show()
    
    # Return results table
    results_df = pd.DataFrame({
        'N_Knots': n_knots_list,
        'Train_RMSE': train_rmses,
        'Val_RMSE': val_rmses,
        'Overfitting_Gap': [val - train for train, val in zip(train_rmses, val_rmses)]
    })
    
    return results_df