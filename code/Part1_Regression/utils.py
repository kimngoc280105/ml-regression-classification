"""
utils.py — Helper functions for Part 1: Regression
CSC14005 - Introduction to Machine Learning
Dataset: California Housing Prices
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import Ridge, Lasso, ElasticNet, LassoCV, lasso_path
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error


# ── Global plot style ───────────────────────────────────────────────────────
PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 130,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
})


# ════════════════════════════════════════════════════════════════════════════
# B.1 — LOAD & DESCRIBE
# ════════════════════════════════════════════════════════════════════════════

def load_data(path: str) -> pd.DataFrame:
    """Load CSV và in thông tin cơ bản về dataset."""
    df = pd.read_csv(path)
    print(f"{'='*55}")
    print(f"  Dataset loaded: {path}")
    print(f"{'='*55}")
    print(f"  Số mẫu (rows)     : {df.shape[0]:,}")
    print(f"  Số features (cols): {df.shape[1]}")
    print(f"\n  Kiểu dữ liệu:")
    for col, dtype in df.dtypes.items():
        null_count = df[col].isnull().sum()
        null_str = f"  ← {null_count} NaN" if null_count > 0 else ""
        print(f"    {col:<35} {str(dtype):<12}{null_str}")
    return df


def describe_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Thống kê mô tả mở rộng (thêm skewness & kurtosis)."""
    numeric = df.select_dtypes(include=np.number)
    desc = numeric.describe().T
    desc["skewness"] = numeric.skew().round(3)
    desc["kurtosis"] = numeric.kurtosis().round(3)
    return desc.round(3)


# ════════════════════════════════════════════════════════════════════════════
# B.2 — MISSING VALUES
# ════════════════════════════════════════════════════════════════════════════

def report_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Báo cáo missing values theo count và phần trăm."""
    missing = df.isnull().sum()
    pct = (missing / len(df) * 100).round(2)
    result = pd.DataFrame({"count": missing, "percent (%)": pct})
    result = result[result["count"] > 0].sort_values("count", ascending=False)
    if result.empty:
        print("Không có missing values.")
    else:
        print(result.to_string())
    return result


def impute_missing(df: pd.DataFrame,
                   col: str = "total_bedrooms",
                   strategy: str = "median") -> pd.DataFrame:
    """
    Impute missing values.
    strategy: 'median' | 'mean' | 'mode'
    """
    df = df.copy()
    if strategy == "median":
        fill_val = df[col].median()
    elif strategy == "mean":
        fill_val = df[col].mean()
    else:
        fill_val = df[col].mode()[0]

    before = df[col].isnull().sum()
    df[col] = df[col].fillna(fill_val)
    print(f"Imputed '{col}': {before} NaN → {strategy} = {fill_val:.2f}")
    return df


# ════════════════════════════════════════════════════════════════════════════
# B.3 — PHÂN BỐ BIẾN TARGET
# ════════════════════════════════════════════════════════════════════════════

def plot_target_distribution(df: pd.DataFrame,
                             target: str = "median_house_value",
                             save: bool = True) -> None:
    """Vẽ histogram + boxplot của biến mục tiêu."""
    y = df[target].dropna()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- Histogram
    ax1 = axes[0]
    ax1.hist(y, bins=60, color="#4C78A8", edgecolor="white", linewidth=0.4, alpha=0.85)
    ax1.axvline(y.median(), color="#E45756", linestyle="--", linewidth=1.6,
                label=f"Median: ${y.median():,.0f}")
    ax1.axvline(y.mean(), color="#F58518", linestyle=":", linewidth=1.6,
                label=f"Mean: ${y.mean():,.0f}")
    ax1.set_xlabel("Giá nhà ($)")
    ax1.set_ylabel("Count")
    ax1.set_title("Histogram")
    ax1.legend(fontsize=9)
    # Format trục x: hiện dạng 100k, 200k, ...
    ax1.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"${int(x/1000)}k")
    )
    ax1.tick_params(axis='x', rotation=0)

    # --- Boxplot (dọc)
    ax2 = axes[1]
    ax2.boxplot(y, vert=True, patch_artist=True,          # ← True
                boxprops=dict(facecolor="#4C78A8", alpha=0.6),
                medianprops=dict(color="#E45756", linewidth=2),
                whiskerprops=dict(linewidth=1.2),
                capprops=dict(linewidth=1.2),
                flierprops=dict(marker="o", markersize=2, alpha=0.3, color="#888"))
    ax2.set_ylabel("Giá nhà ($)")                         # ← đổi sang ylabel
    ax2.set_title("Boxplot")
    ax2.set_xticks([])                                    # ← ẩn trục x thay vì y
    ax2.yaxis.set_major_formatter(                        # ← format trục y
        plt.FuncFormatter(lambda x, _: f"${int(x/1000)}k")
    )
    ax2.tick_params(axis='x', rotation=0)

    fig.suptitle(f"Phân bố biến mục tiêu: {target}", fontsize=13, fontweight="bold")

    capped = (df[target] >= 500_000).sum()
    fig.text(0.5, -0.02,
             f"Lưu ý: {capped:,} mẫu ({capped/len(df)*100:.1f}%) bị cap tại $500,000",
             ha="center", color="#E45756", fontsize=10)

    plt.tight_layout()
    # if save:
        # plt.savefig(f"{PLOT_DIR}/target_distribution.png", bbox_inches="tight")
    plt.show()
    print(f"\nThống kê {target}:")
    print(y.describe().round(2).to_string())


# ════════════════════════════════════════════════════════════════════════════
# B.4 — CORRELATION MATRIX + SCATTER PLOTS
# ════════════════════════════════════════════════════════════════════════════

def plot_correlation_matrix(df: pd.DataFrame,
                            target: str = "median_house_value",
                            save: bool = True) -> pd.Series:
    """Vẽ heatmap tương quan và trả về correlation với target."""
    numeric = df.select_dtypes(include=np.number)
    corr = numeric.corr()

    # Sắp xếp theo tương quan với target
    order = corr[target].abs().sort_values(ascending=False).index
    corr = corr.loc[order, order]

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="RdYlGn", center=0, ax=ax,
                linewidths=0.4, linecolor="#ddd",
                annot_kws={"size": 9})
    ax.set_title("Ma trận tương quan (sắp xếp theo |corr| với target)", fontweight="bold")
    plt.tight_layout()
    # if save:
        # plt.savefig(f"{PLOT_DIR}/correlation_matrix.png", bbox_inches="tight")
    plt.show()

    target_corr = corr[target].drop(target).sort_values(key=abs, ascending=False)
    print(f"\nTương quan với '{target}':")
    print(target_corr.round(3).to_string())
    return target_corr


def plot_scatter_features(df: pd.DataFrame,
                          features: list,
                          target: str = "median_house_value",
                          save: bool = True) -> None:
    """Scatter plot mỗi feature vs target, màu theo density."""
    n = len(features)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(5 * ((n + 1) // 2), 9))
    axes = axes.flatten()

    for i, feat in enumerate(features):
        ax = axes[i]
        x = df[feat].values
        y = df[target].values
        mask = ~(np.isnan(x) | np.isnan(y))
        x, y = x[mask], y[mask]

        # Hexbin density
        hb = ax.hexbin(x, y, gridsize=35, cmap="YlOrRd", mincnt=1, linewidths=0.2)
        ax.set_xlabel(feat)
        ax.set_ylabel(target if i % ((n + 1) // 2) == 0 else "")
        ax.set_title(f"{feat} vs {target}")
        plt.colorbar(hb, ax=ax, label="count")

        # Pearson r
        r, p = stats.pearsonr(x, y)
        ax.text(0.05, 0.92, f"r = {r:.3f}", transform=ax.transAxes,
                fontsize=9, color="#333",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Scatter plots: Features vs Target", fontsize=13, fontweight="bold")
    plt.tight_layout()
    # if save:
        # plt.savefig(f"{PLOT_DIR}/scatter_features.png", bbox_inches="tight")
    plt.show()


# ════════════════════════════════════════════════════════════════════════════
# B.5 — OUTLIER DETECTION
# ════════════════════════════════════════════════════════════════════════════

def detect_outliers_iqr(series: pd.Series, k: float = 1.5) -> pd.Series:
    """Trả về boolean mask của outliers theo IQR."""
    Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
    IQR = Q3 - Q1
    return (series < Q1 - k * IQR) | (series > Q3 + k * IQR)


def detect_outliers_zscore(series: pd.Series, threshold: float = 3.0) -> pd.Series:
    """Trả về boolean mask của outliers theo Z-score."""
    z = np.abs(stats.zscore(series.dropna()))
    result = pd.Series(False, index=series.index)
    result.iloc[series.dropna().index] = z > threshold
    return result


def report_outliers(df: pd.DataFrame,
                    cols: list = None,
                    save: bool = True) -> pd.DataFrame:
    """Báo cáo outliers và vẽ boxplot so sánh."""
    if cols is None:
        cols = ["total_rooms", "total_bedrooms", "population", "households",
                "median_income", "housing_median_age"]

    records = []
    for col in cols:
        series = df[col].dropna()
        iqr_mask = detect_outliers_iqr(series)
        z_mask = detect_outliers_zscore(series)
        records.append({
            "Feature": col,
            "IQR outliers": iqr_mask.sum(),
            "IQR (%)": round(iqr_mask.mean() * 100, 2),
            "Z-score outliers": z_mask.sum(),
            "Z-score (%)": round(z_mask.mean() * 100, 2),
        })

    result_df = pd.DataFrame(records)

    # Boxplots
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    for i, col in enumerate(cols):
        ax = axes[i]
        bp = ax.boxplot(df[col].dropna(), patch_artist=True, vert=True,
                        boxprops=dict(facecolor="#4C78A8", alpha=0.55),
                        medianprops=dict(color="#E45756", linewidth=2),
                        flierprops=dict(marker=".", markersize=2, alpha=0.3, color="#888"))
        ax.set_title(col)
        ax.set_xticks([])
        n_iqr = detect_outliers_iqr(df[col].dropna()).sum()
        ax.text(0.65, 0.93, f"IQR: {n_iqr}", transform=ax.transAxes,
                fontsize=9, color="#E45756",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))

    fig.suptitle("Boxplots — Phát hiện Outliers", fontsize=13, fontweight="bold")
    plt.tight_layout()
    #if save:
        # plt.savefig(f"{PLOT_DIR}/outlier_boxplots.png", bbox_inches="tight")
    plt.show()

    return result_df


# ════════════════════════════════════════════════════════════════════════════
# B.6 — FEATURE ENGINEERING
# ════════════════════════════════════════════════════════════════════════════

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo thêm các ratio features có ý nghĩa thực tế.
    Tham khảo: Géron (2019) Hands-On ML, Chapter 2.
    """
    df = df.copy()
    df["rooms_per_household"]     = df["total_rooms"] / df["households"]
    df["bedrooms_per_room"]       = df["total_bedrooms"] / df["total_rooms"]
    df["population_per_household"] = df["population"] / df["households"]
    print("Đã tạo 3 ratio features:")
    print("  rooms_per_household     = total_rooms / households")
    print("  bedrooms_per_room       = total_bedrooms / total_rooms")
    print("  population_per_household = population / households")
    return df


def encode_categorical(df: pd.DataFrame,
                       col: str = "ocean_proximity") -> pd.DataFrame:
    """One-hot encode biến categorical."""
    categories = df[col].value_counts()
    print(f"\nGiá trị của '{col}':")
    print(categories.to_string())
    df = pd.get_dummies(df, columns=[col], prefix=col, drop_first=False)
    print(f"\nSau one-hot encode: {df.shape[1]} columns")
    return df


# ════════════════════════════════════════════════════════════════════════════
# B.7 — STRATIFIED SPLIT
# ════════════════════════════════════════════════════════════════════════════

def stratified_split(df: pd.DataFrame,
                     target: str = "median_house_value",
                     test_size: float = 0.2,
                     val_size: float = 0.1,
                     random_state: int = 42):
    """
    Stratified train/val/test split theo income category.
    Trả về: X_train, X_val, X_test, y_train, y_val, y_test, feature_names
    """
    df = df.copy()

    # Tạo income_cat để stratify
    df["income_cat"] = pd.cut(
        df["median_income"],
        bins=[0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5]
    )

    # Split 1: tách test set
    split1 = StratifiedShuffleSplit(n_splits=1, test_size=test_size,
                                    random_state=random_state)
    for tv_idx, test_idx in split1.split(df, df["income_cat"]):
        train_val = df.iloc[tv_idx].copy()
        test_set  = df.iloc[test_idx].copy()

    # Split 2: tách val set từ train_val
    actual_val_size = val_size / (1 - test_size)
    split2 = StratifiedShuffleSplit(n_splits=1, test_size=actual_val_size,
                                    random_state=random_state)
    for train_idx, val_idx in split2.split(train_val, train_val["income_cat"]):
        train_set = train_val.iloc[train_idx].copy()
        val_set   = train_val.iloc[val_idx].copy()

    # Xóa cột phụ
    for s in [train_set, val_set, test_set]:
        s.drop(columns=["income_cat"], inplace=True)

    feature_names = [c for c in train_set.columns if c != target]
    X_train = train_set[feature_names].values
    X_val   = val_set[feature_names].values
    X_test  = test_set[feature_names].values
    y_train = train_set[target].values
    y_val   = val_set[target].values
    y_test  = test_set[target].values

    total = len(df)
    print(f"\n{'='*45}")
    print(f"  Train : {len(X_train):>6,}  ({len(X_train)/total*100:.1f}%)")
    print(f"  Val   : {len(X_val):>6,}  ({len(X_val)/total*100:.1f}%)")
    print(f"  Test  : {len(X_test):>6,}  ({len(X_test)/total*100:.1f}%)")
    print(f"  Total : {total:>6,}")
    print(f"  Features: {len(feature_names)}")
    print(f"{'='*45}")

    return X_train, X_val, X_test, y_train, y_val, y_test, feature_names


# ════════════════════════════════════════════════════════════════════════════
# B.8 — STANDARDIZATION
# ════════════════════════════════════════════════════════════════════════════

def scale_features(X_train: np.ndarray,
                   X_val: np.ndarray,
                   X_test: np.ndarray):
    """
    StandardScaler: fit ONLY trên train, transform val và test.
    Trả về: X_train_s, X_val_s, X_test_s, scaler
    """
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)
    print("StandardScaler fitted trên train set.")
    print(f"  Mean (train) ≈ {X_train_s.mean():.6f}  (expected ~0)")
    print(f"  Std  (train) ≈ {X_train_s.std():.6f}   (expected ~1)")
    return X_train_s, X_val_s, X_test_s, scaler


# ════════════════════════════════════════════════════════════════════════════
# B.9 — KIỂM TRA PHÂN PHỐI SAU SPLIT
# ════════════════════════════════════════════════════════════════════════════

def plot_split_distribution(y_train, y_val, y_test,
                            target: str = "median_house_value",
                            save: bool = True) -> None:
    """Kiểm tra phân phối target nhất quán giữa 3 tập."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    sets = [("Train", y_train, "#4C78A8"),
            ("Val",   y_val,   "#F58518"),
            ("Test",  y_test,  "#54A24B")]

    for ax, (name, y, color) in zip(axes, sets):
        ax.hist(y, bins=40, color=color, edgecolor="white",
                linewidth=0.3, alpha=0.85, density=True)
        ax.axvline(np.median(y), color="red", linestyle="--", linewidth=1.4,
                   label=f"Median: {np.median(y):,.0f}")
        ax.set_title(f"{name} set  (n={len(y):,})")
        ax.set_xlabel(f"{target} ($)")
        if ax == axes[0]:
            ax.set_ylabel("Density")
        ax.legend(fontsize=8)

    fig.suptitle("Phân bố target trong 3 tập — kiểm tra stratification",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    # if save:
        # plt.savefig(f"{PLOT_DIR}/split_distribution.png", bbox_inches="tight")
    plt.show()


# ════════════════════════════════════════════════════════════════════════════
# BREUSCH–PAGAN TEST (dùng lại ở phần C)
# ════════════════════════════════════════════════════════════════════════════

def breusch_pagan_test(residuals: np.ndarray,
                       X: np.ndarray,
                       alpha: float = 0.05) -> dict:
    """
    Kiểm định Breusch–Pagan cho heteroscedasticity.
    H0: phương sai đồng nhất (homoscedastic)
    H1: phương sai không đồng nhất (heteroscedastic)
    """
    n = len(residuals)
    e2 = residuals ** 2
    e2_normalized = e2 / e2.mean()

    # OLS của e2_normalized lên X
    X_bp = np.column_stack([np.ones(n), X])
    beta = np.linalg.lstsq(X_bp, e2_normalized, rcond=None)[0]
    e2_hat = X_bp @ beta
    SS_reg = ((e2_hat - e2_normalized.mean()) ** 2).sum()
    SS_tot = ((e2_normalized - e2_normalized.mean()) ** 2).sum()
    LM = n * SS_reg / SS_tot
    p_value = 1 - stats.chi2.cdf(LM, df=X.shape[1])

    result = {
        "LM statistic": round(LM, 4),
        "p-value": round(p_value, 6),
        "df": X.shape[1],
        "conclusion": "Heteroscedastic (vi phạm)" if p_value < alpha else "Homoscedastic (không vi phạm)"
    }
    print(f"\nBreusch–Pagan Test:")
    print(f"  LM statistic : {result['LM statistic']}")
    print(f"  p-value      : {result['p-value']}")
    print(f"  Kết luận     : {result['conclusion']} (α = {alpha})")
    return result






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

    # plt.savefig(f"{PLOT_DIR}/{model_class.__name__.lower()}_regularization_path.png", bbox_inches="tight")
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

def plot_elastic_net_heatmap(X_train, y_train, alphas_en, l1_ratios, k=3):
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
    # plt.savefig(f"{PLOT_DIR}/elastic_net.png", bbox_inches="tight")
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
    # plt.savefig(f"{PLOT_DIR}/feature_selection.png", bbox_inches="tight")
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
        
    elif basis_type == 'spline':
        n_knots = kwargs.get('n_knots', 5)
        degree = kwargs.get('degree', 3)
        transformer = SplineBasis(n_knots=n_knots, degree=degree)
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


# ════════════════════════════════════════════════════════════════════════════
# D. EVALUATION METRICS — Tất cả mô hình
# ════════════════════════════════════════════════════════════════════════════

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def compute_metrics(y_true, y_pred, model_name="Model"):
    """
    Tính toàn bộ các chỉ số đánh giá cho bài toán hồi quy.
    
    Công thức:
    - MSE = (1/N) * Σ(tₙ - yₙ)²
    - RMSE = √MSE
    - MAE = (1/N) * Σ|tₙ - yₙ|
    - R² = 1 - SSres/SStot
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    metrics = {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R²": r2
    }
    
    return metrics


def print_metrics(y_true, y_pred, model_name="Model", decimals=3):
    """In ra các chỉ số đánh giá dạng bảng."""
    metrics = compute_metrics(y_true, y_pred, model_name)
    
    print(f"\n{'='*60}")
    print(f"  Kết quả đánh giá: {model_name}")
    print(f"{'='*60}")
    print(f"  Chỉ số              | Giá trị")
    print(f"  {'-'*58}")
    print(f"  MSE (Mean Sq. Err)  | {metrics['MSE']:>18,.{decimals}f}")
    print(f"  RMSE (Sq. Root MSE) | {metrics['RMSE']:>18,.{decimals}f}")
    print(f"  MAE (Mean Abs. Err) | {metrics['MAE']:>18,.{decimals}f}")
    print(f"  R² (Coefficient)    | {metrics['R²']:>18.{decimals}f}")
    print(f"{'='*60}\n")
    
    return metrics


def evaluate_model_on_splits(model, X_train, X_val, X_test, y_train, y_val, y_test, 
                             model_name="Model"):
    """
    Đánh giá mô hình trên cả 3 tập: train, val, test.
    In ra bảng so sánh các chỉ số trên 3 tập.
    """
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
    metrics_train = compute_metrics(y_train, y_train_pred)
    metrics_val = compute_metrics(y_val, y_val_pred)
    metrics_test = compute_metrics(y_test, y_test_pred)
    
    print(f"\n{'='*80}")
    print(f"  {model_name} — Đánh giá trên Train / Val / Test")
    print(f"{'='*80}")
    print(f"{'Chỉ số':<20} | {'Train':>18} | {'Val':>18} | {'Test':>18}")
    print(f"{'-'*80}")
    
    for key in ['MSE', 'RMSE', 'MAE', 'R²']:
        if key == 'R²':
            print(f"{key:<20} | {metrics_train[key]:>18.4f} | {metrics_val[key]:>18.4f} | {metrics_test[key]:>18.4f}")
        else:
            print(f"{key:<20} | {metrics_train[key]:>18,.2f} | {metrics_val[key]:>18,.2f} | {metrics_test[key]:>18,.2f}")
    
    print(f"{'='*80}\n")
    
    return {
        'train': metrics_train,
        'val': metrics_val,
        'test': metrics_test
    }


def compare_models_on_test(models_dict, X_test, y_test):
    """So sánh nhiều mô hình trên tập test."""
    results = []
    
    for model_name, model in models_dict.items():
        y_pred = model.predict(X_test)
        metrics = compute_metrics(y_test, y_pred, model_name)
        metrics['Model'] = model_name
        results.append(metrics)
    
    df_results = pd.DataFrame(results)
    df_results = df_results[['Model', 'MSE', 'RMSE', 'MAE', 'R²']]
    
    print(f"\n{'='*85}")
    print(f"  SO SÁNH CÁC MÔ HÌNH TRÊN TẬP TEST")
    print(f"{'='*85}")
    print(df_results.to_string(index=False))
    print(f"{'='*85}\n")
    
    return df_results


# ════════════════════════════════════════════════════════════════════════════
# D.2 LEARNING CURVES, RESIDUAL PLOTS, PREDICTED VS ACTUAL
# ════════════════════════════════════════════════════════════════════════════

def plot_learning_curve(X_train, y_train, X_val, y_val, model_class, 
                        param_dict, train_sizes=None, model_name="Model"):
    """
    Vẽ learning curve: RMSE theo số lượng mẫu training.
    Kiểm tra underfitting/overfitting.
    """
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 10)
    
    train_sizes_abs = (train_sizes * len(X_train)).astype(int)
    train_rmses = []
    val_rmses = []
    
    for size in train_sizes_abs:
        X_train_sub = X_train[:size]
        y_train_sub = y_train[:size]
        
        model = model_class(**param_dict, max_iter=10000)
        model.fit(X_train_sub, y_train_sub)
        
        train_rmse = np.sqrt(mean_squared_error(y_train_sub, model.predict(X_train_sub)))
        val_rmse = np.sqrt(mean_squared_error(y_val, model.predict(X_val)))
        
        train_rmses.append(train_rmse)
        val_rmses.append(val_rmse)
    
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(train_sizes_abs, train_rmses, 'o-', linewidth=2.5, markersize=7,
            label='Train RMSE', color='#1f77b4')
    ax.plot(train_sizes_abs, val_rmses, 's-', linewidth=2.5, markersize=7,
            label='Validation RMSE', color='#ff7f0e')
    
    ax.fill_between(train_sizes_abs, train_rmses, val_rmses, 
                    where=np.array(val_rmses) > np.array(train_rmses),
                    alpha=0.2, color='red', label='Overfitting zone')
    
    ax.set_xlabel('# Training samples', fontsize=11, fontweight='bold')
    ax.set_ylabel('RMSE', fontsize=11, fontweight='bold')
    ax.set_title(f'Learning Curve: {model_name}', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    safe_name = model_name.replace(' ', '_').replace('(', '').replace(')', '')
    #plt.savefig(f"{PLOT_DIR}/learning_curve_{safe_name}.png", 
    #            bbox_inches='tight', dpi=130)
    plt.show()
    
    print(f"\nLearning Curve — {model_name}:")
    print(f"  Train RMSE (cuối): {train_rmses[-1]:,.2f}")
    print(f"  Val RMSE (cuối)  : {val_rmses[-1]:,.2f}")
    print(f"  Overfitting gap  : {val_rmses[-1] - train_rmses[-1]:,.2f}")


def plot_residuals(y_true, y_pred, model_name="Model"):
    """
    Vẽ 3 biểu đồ kiểm tra phần dư:
    1. Residuals vs Predicted (ngẫu nhiên?)
    2. Q-Q Plot (chuẩn không?)
    3. Histogram residuals (phân phối)
    """
    residuals = np.asarray(y_true).flatten() - np.asarray(y_pred).flatten()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    
    # 1. Residuals vs Predicted
    ax1 = axes[0]
    ax1.scatter(y_pred, residuals, alpha=0.5, s=20, color='teal', edgecolors='none')
    ax1.axhline(0, color='red', linestyle='--', linewidth=2)
    ax1.set_xlabel('Predicted values ($)', fontsize=10)
    ax1.set_ylabel('Residuals ($)', fontsize=10)
    ax1.set_title('1. Residuals vs Predicted\n(Phải ngẫu nhiên ≈ 0)', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 2. Q-Q Plot
    ax2 = axes[1]
    scipy_stats = __import__('scipy').stats
    scipy_stats.probplot(residuals, dist="norm", plot=ax2)
    ax2.set_title('2. Q-Q Plot\n(Trên đường 45° = chuẩn)', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Histogram
    '''
    ax3 = axes[2]
    ax3.hist(residuals, bins=40, color='skyblue', edgecolor='black', 
             density=True, alpha=0.7)
    mu, sigma = residuals.mean(), residuals.std()
    x = np.linspace(residuals.min(), residuals.max(), 100)
    scipy_stats = __import__('scipy').stats
    ax3.plot(x, scipy_stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal dist')
    ax3.set_xlabel('Residual value ($)', fontsize=10)
    ax3.set_ylabel('Density', fontsize=10)
    ax3.set_title('3. Histogram Residuals\n(Hình chuông = tốt)', fontsize=11, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')
    '''

    fig.suptitle(f'Residual Analysis — {model_name}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    safe_name = model_name.replace(' ', '_').replace('(', '').replace(')', '')
    # plt.savefig(f"{PLOT_DIR}/residuals_{safe_name}.png", 
    #            bbox_inches='tight', dpi=130)
    plt.show()


def plot_predicted_vs_actual(y_true, y_pred, model_name="Model"):
    """Vẽ biểu đồ Predicted vs Actual."""
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.scatter(y_true, y_pred, alpha=0.5, s=30, color='#1f77b4', edgecolors='none')
    
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2.5, label='Perfect prediction')
    
    ax.set_xlabel('Actual values ($)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Predicted values ($)', fontsize=11, fontweight='bold')
    ax.set_title(f'Predicted vs Actual — {model_name}', fontsize=13, fontweight='bold')
    
    r2 = r2_score(y_true, y_pred)
    ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    safe_name = model_name.replace(' ', '_').replace('(', '').replace(')', '')
    # plt.savefig(f"{PLOT_DIR}/predicted_vs_actual_{safe_name}.png", 
    #            bbox_inches='tight', dpi=130)
    plt.show()


# ════════════════════════════════════════════════════════════════════════════
# D.3 K-FOLD CROSS-VALIDATION & STATISTICAL TESTS
# ════════════════════════════════════════════════════════════════════════════

def evaluate_with_kfold(X, y, model_class, param_dict, k=10, model_name="Model"):
    """
    K-Fold Cross-Validation để tính mean ± std cho từng chỉ số.
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    results = {
        'MSE': [], 'RMSE': [], 'MAE': [], 'R²': []
    }
    
    for train_idx, test_idx in kf.split(X):
        X_train_fold, X_test_fold = X[train_idx], X[test_idx]
        y_train_fold, y_test_fold = y[train_idx], y[test_idx]
        
        model = model_class(**param_dict, max_iter=10000)
        model.fit(X_train_fold, y_train_fold)
        
        y_pred = model.predict(X_test_fold)
        
        metrics = compute_metrics(y_test_fold, y_pred)
        for key in results.keys():
            results[key].append(metrics[key])
    
    stats_dict = {}
    for key in results.keys():
        values = np.array(results[key])
        stats_dict[key] = (values.mean(), values.std())
    
    print(f"\n{'='*85}")
    print(f"  {k}-FOLD CROSS-VALIDATION — {model_name}")
    print(f"{'='*85}")
    print(f"{'Chỉ số':<20} | {'Mean':>20} | {'Std':>20}")
    print(f"{'-'*85}")
    
    for key in ['MSE', 'RMSE', 'MAE', 'R²']:
        mean_val, std_val = stats_dict[key]
        if key == 'R²':
            print(f"{key:<20} | {mean_val:>20.4f} | {std_val:>20.4f}")
        else:
            print(f"{key:<20} | {mean_val:>20,.2f} | {std_val:>20,.2f}")
    
    print(f"{'='*85}\n")
    
    return stats_dict


def statistical_test_paired(y_true, y_pred_model1, y_pred_model2, 
                            model1_name="Model 1", model2_name="Model 2", 
                            test_type='t-test'):
    """
    Kiểm định thống kê paired: So sánh 2 mô hình.
    test_type: 't-test' | 'wilcoxon'
    """
    scipy_stats = __import__('scipy').stats
    
    y_true = np.asarray(y_true).flatten()
    y_pred_model1 = np.asarray(y_pred_model1).flatten()
    y_pred_model2 = np.asarray(y_pred_model2).flatten()
    
    error1 = np.abs(y_true - y_pred_model1)
    error2 = np.abs(y_true - y_pred_model2)
    
    print(f"\n{'='*90}")
    print(f"  KIỂM ĐỊNH THỐNG KÊ (PAIRED) — {test_type.upper()}")
    print(f"  So sánh: {model1_name} vs {model2_name}")
    print(f"{'='*90}")
    
    if test_type.lower() == 't-test':
        t_stat, p_value = scipy_stats.ttest_rel(error1, error2)
        test_name = "Paired t-test"
    elif test_type.lower() == 'wilcoxon':
        w_stat, p_value = scipy_stats.wilcoxon(error1, error2)
        t_stat = w_stat
        test_name = "Wilcoxon signed-rank test"
    else:
        raise ValueError("test_type must be 't-test' or 'wilcoxon'")
    
    print(f"\n  Test statistic      : {t_stat:.6f}")
    print(f"  p-value             : {p_value:.6f}")
    print(f"\n  Mean Absolute Error (MAE):")
    print(f"    {model1_name:<20} : {error1.mean():>15,.2f} ± {error1.std():>15,.2f}")
    print(f"    {model2_name:<20} : {error2.mean():>15,.2f} ± {error2.std():>15,.2f}")
    print(f"    Hiệu khác           : {abs(error1.mean() - error2.mean()):>15,.2f}")
    
    alpha = 0.05
    if p_value < alpha:
        better_model = model2_name if error2.mean() < error1.mean() else model1_name
        conclusion = f"CÓ SỰ KHÁC BIỆT CÓ Ý NGHĨA (p = {p_value:.6f} < {alpha})"
        conclusion += f"\n  → {better_model} tốt hơn"
    else:
        conclusion = f"KHÔNG CÓ sự khác biệt (p = {p_value:.6f} ≥ {alpha})"
    
    print(f"\n  Kết luận (α = {alpha}):")
    print(f"  {conclusion}")
    print(f"{'='*90}\n")
    
    return {
        'test': test_name,
        'statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < alpha,
        'conclusion': conclusion
    }