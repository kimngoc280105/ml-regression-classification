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
    if save:
        plt.savefig(f"{PLOT_DIR}/target_distribution.png", bbox_inches="tight")
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
    if save:
        plt.savefig(f"{PLOT_DIR}/correlation_matrix.png", bbox_inches="tight")
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
    if save:
        plt.savefig(f"{PLOT_DIR}/scatter_features.png", bbox_inches="tight")
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
    if save:
        plt.savefig(f"{PLOT_DIR}/outlier_boxplots.png", bbox_inches="tight")
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
    if save:
        plt.savefig(f"{PLOT_DIR}/split_distribution.png", bbox_inches="tight")
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


# 2.2.3

def normal_equations(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Tính nghiệm OLS bằng Normal Equations.
    w* = (ΦᵀΦ)⁻¹Φᵀt
    """
    # Thêm cột bias (φ₀ = 1)
    Phi = np.column_stack([np.ones(len(X)), X])
    
    # w* = (ΦᵀΦ)⁻¹Φᵀt
    w = np.linalg.pinv(Phi.T @ Phi) @ (Phi.T @ y)
    return w


def predict(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Dự đoán với bias term."""
    Phi = np.column_stack([np.ones(len(X)), X])
    return Phi @ w


def compute_metrics(y_true, y_pred) -> dict:
    """Tính MSE, RMSE, MAE, R²."""
    residuals = y_true - y_pred
    ss_res = (residuals ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return {
        "MSE" : round(np.mean(residuals**2), 2),
        "RMSE": round(np.sqrt(np.mean(residuals**2)), 2),
        "MAE" : round(np.mean(np.abs(residuals)), 2),
        "R²"  : round(1 - ss_res / ss_tot, 4),
    }


#gd

def minibatch_gd(X, y, batch_size=256, epochs=200,
                 lr=0.01, schedule='cosine',
                 lr_min=1e-4, drop=0.5, drop_every=50):
    """
    Mini-batch Gradient Descent cho Linear Regression.
    Trả về w, history (loss theo epoch).
    """
    n, d = X.shape
    Phi = np.column_stack([np.ones(n), X])   # thêm bias
    w = np.zeros(Phi.shape[1])               # khởi tạo w = 0
    history = []
    n_batches = max(1, n // batch_size)

    for epoch in range(epochs):
        # Shuffle dữ liệu mỗi epoch
        idx = np.random.permutation(n)
        Phi_s, y_s = Phi[idx], y[idx]

        # Learning rate schedule
        if schedule == 'step':
            lr_t = lr * (drop ** (epoch // drop_every))
        elif schedule == 'cosine':
            lr_t = lr_min + 0.5*(lr - lr_min)*(1 + np.cos(np.pi * epoch / epochs))
        else:
            lr_t = lr  # constant

        # Mini-batch updates
        for b in range(n_batches):
            Phi_b = Phi_s[b*batch_size : (b+1)*batch_size]
            y_b   = y_s  [b*batch_size : (b+1)*batch_size]
            grad = -Phi_b.T @ (y_b - Phi_b @ w) / len(y_b)
            w = w - lr_t * grad

        # Ghi loss sau mỗi epoch (trên toàn train)
        loss = np.mean((y - Phi[:,1:] @ w[1:] - w[0])**2)
        history.append(loss)

    return w, history


def plot_residuals(y_true, y_pred, model_name='OLS', save=True):
    """Residual plot + QQ-plot để kiểm tra giả thiết Gauss-Markov."""
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- Residual vs Fitted
    ax1 = axes[0]
    ax1.scatter(y_pred, residuals, alpha=0.15, s=6, color='#4C78A8')
    ax1.axhline(0, color='#E45756', linewidth=1.5, linestyle='--')
    ax1.set_xlabel('Fitted values ($)')
    ax1.set_ylabel('Residuals ($)')
    ax1.set_title(f'Residual plot — {model_name}')
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'${int(x/1000)}k'))
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'${int(x/1000)}k'))

    # --- QQ-plot
    ax2 = axes[1]
    from scipy.stats import probplot
    probplot(residuals, dist='norm', plot=ax2)
    ax2.set_title(f'QQ-plot — {model_name}')
    ax2.get_lines()[0].set(markersize=2, alpha=0.3, color='#4C78A8')
    ax2.get_lines()[1].set(color='#E45756', linewidth=1.5)

    plt.tight_layout()
    if save:
        plt.savefig(f'plots/residuals_{model_name}.png', bbox_inches='tight')
    plt.show()

    print(f"Residuals — mean: {residuals.mean():.2f}  std: {residuals.std():.2f}")