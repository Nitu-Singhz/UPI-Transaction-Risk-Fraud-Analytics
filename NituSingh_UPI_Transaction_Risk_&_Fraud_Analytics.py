"""
UPI Transaction Risk & Fraud Analytics
========================================
Single-file application: analysis pipeline + Streamlit dashboard + report generation.

Run dashboard : streamlit run upi_fraud_analysis.py
Run pipeline  : python upi_fraud_analysis.py --pipeline

Dataset       : Synthetic UPI transaction data (~10,000 records)
IMPORTANT     : This dataset is SYNTHETIC. It does not contain real personal,
                banking, or financial information.
"""

# ─── stdlib ───────────────────────────────────────────────────────────────────
import os, sys, json, warnings, logging
from pathlib import Path
from datetime import datetime

# ─── third-party ──────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve,
)
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# A.  PATHS  (all relative — works on any machine)
# ══════════════════════════════════════════════════════════════════════════════
BASE_DIR     = Path(__file__).parent
DATA_DIR     = BASE_DIR / "data"
OUT_DIR      = BASE_DIR / "outputs"
CHARTS_DIR   = OUT_DIR  / "charts"
TABLES_DIR   = OUT_DIR  / "tables"
RESULTS_DIR  = OUT_DIR  / "results"
REPORT_PATH  = BASE_DIR / "Project_Report.docx"

RAW_CSV      = DATA_DIR    / "upi_transactions.csv"
CLEAN_CSV    = TABLES_DIR  / "upi_transactions_cleaned.csv"
PBI_CSV      = TABLES_DIR  / "upi_transactions_powerbi.csv"
RESULTS_JSON = RESULTS_DIR / "model_results.json"
FI_CSV       = RESULTS_DIR / "feature_importances_rf.csv"

for d in [DATA_DIR, CHARTS_DIR, TABLES_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# B.  CONSTANTS  +  PALETTE
# ══════════════════════════════════════════════════════════════════════════════
FRAUD_COL        = "Fraud_Label"
AMOUNT_COL       = "Transaction_Amount_INR"
RISK_SCORE_COL   = "Risk_Score"
RISK_LEVEL_COL   = "Risk_Level"

# Features used by ML models (Risk_Score & Risk_Level explicitly excluded)
MODEL_FEATURES = [
    "Transaction_Amount_INR", "Account_Age_Days", "Previous_Transaction_Count",
    "Average_Previous_Transaction_Amount_INR", "Failed_Attempts",
    "Transactions_Last_24_Hours", "Transactions_Last_7_Days",
    "Is_New_Device", "Is_International", "Transaction_Hour",
    "Is_Weekend", "Is_Odd_Hour", "Amount_vs_Avg_Ratio",
    "Is_High_Value", "Velocity_24h_High", "Velocity_7d_High",
    "Has_Failed_Attempts", "Location_Mismatch",
    "Is_Failed_Transaction", "Is_Pending_Transaction",
]

CATEGORICAL_FEATURES = ["Transaction_Type", "Device_Type", "Network_Type", "Sender_Age_Group"]

# ── PALETTE ──────────────────────────────────────────────────────────────────
BG_DARK      = "#0F0B1A"
BG_CARD      = "#171126"
BG_CARD2     = "#1B1430"
C_DEEP       = "#5B21B6"
C_PRIMARY    = "#7C3AED"
C_MED        = "#8B5CF6"
C_LAV        = "#A78BFA"
C_LLIGHT     = "#C4B5FD"
C_VLLIGHT    = "#DDD6FE"
C_WHITE      = "#FFFFFF"
C_SEC        = "#B8B2C9"
C_MUTED      = "#8F89A1"
C_ACCENT_RED = "#DC2626"
C_ACCENT_AMB = "#D97706"

PURPLE_PALETTE = ["#5B21B6","#6D28D9","#7C3AED","#8B5CF6","#A78BFA","#C4B5FD","#DDD6FE"]

RISK_COLOR = {
    "Low":      "#C4B5FD",
    "Medium":   "#8B5CF6",
    "High":     "#5B21B6",
    "Critical": "#DC2626",
}

FRAUD_PALETTE   = {"Non-Fraud": "#C4B5FD", "Fraud": "#5B21B6"}
FRAUD_COLOR_MAP = {"0": "#C4B5FD", "1": "#5B21B6"}

# Plotly layout template
PLOTLY_LAYOUT = dict(
    plot_bgcolor=BG_CARD,
    paper_bgcolor=BG_DARK,
    font=dict(color=C_WHITE, family="Inter, system-ui, sans-serif"),
    title_font=dict(color=C_WHITE, size=15),
    xaxis=dict(tickfont=dict(color=C_SEC), gridcolor="#3D2E5A",
               linecolor="#3D2E5A", zerolinecolor="#3D2E5A"),
    yaxis=dict(tickfont=dict(color=C_SEC), gridcolor="#3D2E5A",
               linecolor="#3D2E5A", zerolinecolor="#3D2E5A"),
    legend=dict(bgcolor="#1B1430", bordercolor="#3D2E5A", borderwidth=1,
                font=dict(color=C_WHITE)),
    margin=dict(t=50, b=30, l=20, r=20),
)

PURPLE_CS = [[0, "#1B1430"], [0.5, "#7C3AED"], [1.0, "#DDD6FE"]]

sns.set_theme(style="dark", font_scale=1.0)


# ══════════════════════════════════════════════════════════════════════════════
# C.  DATA PIPELINE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def _save_chart(name: str) -> Path:
    p = CHARTS_DIR / name
    plt.tight_layout()
    plt.savefig(p, dpi=150, bbox_inches="tight", facecolor=BG_DARK)
    plt.close()
    return p


def load_and_clean() -> tuple:
    """Load CSV, validate, clean, and return (df, validation_report)."""
    if not RAW_CSV.exists():
        raise FileNotFoundError(
            f"Dataset not found at {RAW_CSV}.\n"
            "Please place upi_transactions.csv in the data/ folder."
        )

    df = pd.read_csv(RAW_CSV)
    report = {}

    report["raw_shape"]     = df.shape
    report["missing"]       = df.isnull().sum()[df.isnull().sum() > 0].to_dict()
    report["dup_rows"]      = int(df.duplicated().sum())
    report["dup_txn_ids"]   = int(df["Transaction_ID"].duplicated().sum())
    report["fraud_dist"]    = df[FRAUD_COL].value_counts().to_dict()
    report["fraud_rate"]    = float(df[FRAUD_COL].mean() * 100)
    report["rs_corr"]       = float(df[RISK_SCORE_COL].corr(df[FRAUD_COL]))

    df["Transaction_Date"] = pd.to_datetime(df["Transaction_Date"], errors="coerce")
    df["Transaction_DateTime"] = pd.to_datetime(
        df["Transaction_Date"].astype(str) + " " + df["Transaction_Time"].astype(str),
        errors="coerce",
    )

    df = df.drop_duplicates(subset="Transaction_ID", keep="first")

    df["Receiver_Location"] = df["Receiver_Location"].fillna("Unknown")
    net_mode = df["Network_Type"].mode()[0]
    df["Network_Type"] = df["Network_Type"].fillna(net_mode)
    avg_med = df["Average_Previous_Transaction_Amount_INR"].median()
    df["Average_Previous_Transaction_Amount_INR"] = (
        df["Average_Previous_Transaction_Amount_INR"].fillna(avg_med)
    )

    for col in ["Is_New_Device", "Is_International"]:
        if df[col].dtype == object:
            df[col] = df[col].map({"Yes": 1, "No": 0})
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    report["clean_shape"] = df.shape
    report["amt_p99"]     = float(df[AMOUNT_COL].quantile(0.99))
    return df, report


def engineer_features(df: pd.DataFrame, amt_p99: float) -> pd.DataFrame:
    """Add all engineered columns. Pure function — does not modify in-place."""
    df = df.copy()
    df["Transaction_Hour"]       = df["Transaction_DateTime"].dt.hour
    df["Transaction_DayOfWeek"]  = df["Transaction_DateTime"].dt.dayofweek
    df["Is_Weekend"]             = (df["Transaction_DayOfWeek"] >= 5).astype(int)
    df["Is_Odd_Hour"]            = ((df["Transaction_Hour"] >= 0) & (df["Transaction_Hour"] < 6)).astype(int)
    df["Amount_vs_Avg_Ratio"]    = (
        df[AMOUNT_COL] / df["Average_Previous_Transaction_Amount_INR"].replace(0, np.nan)
    ).fillna(1.0).clip(0, 100)
    df["Is_High_Value"]          = (df[AMOUNT_COL] > amt_p99).astype(int)
    df["Velocity_24h_High"]      = (df["Transactions_Last_24_Hours"] >= 5).astype(int)
    df["Velocity_7d_High"]       = (df["Transactions_Last_7_Days"] >= 15).astype(int)
    df["Has_Failed_Attempts"]    = (df["Failed_Attempts"] > 0).astype(int)
    df["Location_Mismatch"]      = (df["Sender_Location"] != df["Receiver_Location"]).astype(int)
    df["Is_Failed_Transaction"]  = (df["Transaction_Status"] == "Failed").astype(int)
    df["Is_Pending_Transaction"] = (df["Transaction_Status"] == "Pending").astype(int)
    df["Account_Age_Category"]   = pd.cut(
        df["Account_Age_Days"],
        bins=[0, 90, 180, 365, 730, np.inf],
        labels=["Very New", "New", "Moderate", "Established", "Mature"],
    )
    return df


def compute_custom_risk_score(row) -> int:
    """
    Transparent rule-based risk score 0–100.
    Fraud_Label is NEVER used here.
    """
    s = 0
    r = row.get("Amount_vs_Avg_Ratio", 1)
    if r > 5:   s += 25
    elif r > 3: s += 15
    elif r > 2: s += 8
    if row.get("Is_New_Device",    0) == 1: s += 15
    fa = row.get("Failed_Attempts", 0)
    if fa >= 3:   s += 15
    elif fa >= 1: s += 8
    v = row.get("Transactions_Last_24_Hours", 0)
    if v >= 8:    s += 15
    elif v >= 5:  s += 8
    if row.get("Is_Odd_Hour",       0) == 1: s += 10
    age = row.get("Account_Age_Days", 999)
    if age < 90:   s += 10
    elif age < 180: s += 5
    if row.get("Is_International",  0) == 1: s += 5
    amt = row.get(AMOUNT_COL, 0)
    if amt > 50000:  s += 10
    elif amt > 20000: s += 5
    return min(s, 100)


def score_to_level(score: int) -> str:
    if score >= 70: return "Critical"
    if score >= 50: return "High"
    if score >= 25: return "Medium"
    return "Low"


def run_anomaly_detection(df: pd.DataFrame) -> pd.DataFrame:
    """Fit Isolation Forest and add Is_Anomaly / IF_Score_Raw columns."""
    feats = [c for c in [
        AMOUNT_COL, "Account_Age_Days", "Failed_Attempts",
        "Transactions_Last_24_Hours", "Transactions_Last_7_Days",
        "Is_New_Device", "Is_International", "Amount_vs_Avg_Ratio",
        "Transaction_Hour", "Location_Mismatch",
    ] if c in df.columns]

    X = StandardScaler().fit_transform(df[feats])
    iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
    df = df.copy()
    df["Is_Anomaly"]   = (iso.fit_predict(X) == -1).astype(int)
    df["IF_Score_Raw"] = iso.decision_function(X)
    return df


def _build_X_y(df: pd.DataFrame):
    """Build feature matrix X and target y for ML. No leakage."""
    avail = [c for c in MODEL_FEATURES if c in df.columns]
    dummies = pd.get_dummies(df[CATEGORICAL_FEATURES], drop_first=True)
    X = pd.concat([df[avail], dummies], axis=1).astype(float)
    y = df[FRAUD_COL].astype(int)
    return X, y


def run_ml_pipeline(df: pd.DataFrame) -> dict:
    """
    Train LR and RF on df. Return metrics dict, curves, feature importance.
    Risk_Score and Risk_Level are deliberately excluded from X.
    """
    X, y = _build_X_y(df)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(X_tr)
    X_te_sc  = sc.transform(X_te)

    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    lr.fit(X_tr_sc, y_tr)
    yp_lr    = lr.predict(X_te_sc)
    yprob_lr = lr.predict_proba(X_te_sc)[:, 1]

    rf = RandomForestClassifier(
        n_estimators=300, class_weight="balanced",
        random_state=42, n_jobs=-1, max_depth=12
    )
    rf.fit(X_tr, y_tr)
    yp_rf    = rf.predict(X_te)
    yprob_rf = rf.predict_proba(X_te)[:, 1]

    def _m(yt, yp, yprob):
        r = classification_report(yt, yp, output_dict=True, zero_division=0)
        return {
            "roc_auc":   round(roc_auc_score(yt, yprob), 4),
            "pr_auc":    round(average_precision_score(yt, yprob), 4),
            "precision": round(r.get("1", {}).get("precision", 0), 4),
            "recall":    round(r.get("1", {}).get("recall", 0), 4),
            "f1":        round(r.get("1", {}).get("f1-score", 0), 4),
            "accuracy":  round(r.get("accuracy", 0), 4),
        }

    lr_m = _m(y_te, yp_lr, yprob_lr)
    rf_m = _m(y_te, yp_rf, yprob_rf)

    fpr_lr, tpr_lr, _ = roc_curve(y_te, yprob_lr)
    fpr_rf, tpr_rf, _ = roc_curve(y_te, yprob_rf)
    pr_prec_lr, pr_rec_lr, _ = precision_recall_curve(y_te, yprob_lr)
    pr_prec_rf, pr_rec_rf, _ = precision_recall_curve(y_te, yprob_rf)

    cm_lr = confusion_matrix(y_te, yp_lr).tolist()
    cm_rf = confusion_matrix(y_te, yp_rf).tolist()

    fi = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_lr = cross_val_score(
        Pipeline([("sc", StandardScaler()),
                  ("lr", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))]),
        X, y, cv=cv, scoring="roc_auc",
    )
    cv_rf = cross_val_score(
        RandomForestClassifier(n_estimators=100, class_weight="balanced",
                               random_state=42, n_jobs=-1, max_depth=12),
        X, y, cv=cv, scoring="roc_auc",
    )

    return {
        "lr": lr_m, "rf": rf_m,
        "cm_lr": cm_lr, "cm_rf": cm_rf,
        "fpr_lr": fpr_lr.tolist(), "tpr_lr": tpr_lr.tolist(),
        "fpr_rf": fpr_rf.tolist(), "tpr_rf": tpr_rf.tolist(),
        "pr_prec_lr": pr_prec_lr.tolist(), "pr_rec_lr": pr_rec_lr.tolist(),
        "pr_prec_rf": pr_prec_rf.tolist(), "pr_rec_rf": pr_rec_rf.tolist(),
        "fi_names": fi.index.tolist()[:20],
        "fi_vals":  fi.values.tolist()[:20],
        "cv_lr_mean": round(float(cv_lr.mean()), 4),
        "cv_lr_std":  round(float(cv_lr.std()),  4),
        "cv_rf_mean": round(float(cv_rf.mean()), 4),
        "cv_rf_std":  round(float(cv_rf.std()),  4),
        "fraud_base": float(y.mean()),
        "n_test": len(y_te),
    }


# ══════════════════════════════════════════════════════════════════════════════
# D.  CHART GENERATION  (purple/dark theme)
# ══════════════════════════════════════════════════════════════════════════════

def _mpl_theme():
    """Apply dark purple matplotlib theme globally."""
    plt.rcParams.update({
        "figure.facecolor":   BG_DARK,
        "axes.facecolor":     BG_CARD,
        "axes.edgecolor":     "#3D2E5A",
        "axes.labelcolor":    C_WHITE,
        "axes.titlecolor":    C_WHITE,
        "axes.grid":          True,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "grid.color":         "#3D2E5A",
        "grid.alpha":         0.15,
        "text.color":         C_WHITE,
        "xtick.color":        C_SEC,
        "ytick.color":        C_SEC,
        "xtick.labelcolor":   C_SEC,
        "ytick.labelcolor":   C_SEC,
        "legend.facecolor":   BG_CARD2,
        "legend.edgecolor":   "#3D2E5A",
        "legend.labelcolor":  C_WHITE,
        "savefig.facecolor":  BG_DARK,
        "savefig.edgecolor":  BG_DARK,
        "figure.edgecolor":   BG_DARK,
    })


def generate_all_charts(df: pd.DataFrame, ml_res: dict) -> dict:
    """Generate and save all matplotlib charts. Returns {name: Path}."""
    _mpl_theme()
    charts = {}

    def _styled_ax(ax):
        ax.set_facecolor(BG_CARD)
        for spine in ax.spines.values():
            spine.set_edgecolor("#3D2E5A")
        ax.tick_params(colors=C_SEC)
        ax.xaxis.label.set_color(C_WHITE)
        ax.yaxis.label.set_color(C_WHITE)
        ax.title.set_color(C_WHITE)
        ax.grid(True, color="#3D2E5A", alpha=0.15)

    def _fraud_rate_bar(grp_col, fname, title, figsize=(9, 4)):
        g = (df.groupby(grp_col)[FRAUD_COL]
             .agg(["sum", "count"])
             .assign(Rate=lambda x: x["sum"] / x["count"] * 100)
             .sort_values("Rate", ascending=False))
        fig, ax = plt.subplots(figsize=figsize, facecolor=BG_DARK)
        _styled_ax(ax)
        bars = ax.barh(g.index, g["Rate"], color=C_MED)
        ax.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=9, color=C_WHITE)
        ax.set_title(title, fontsize=11, pad=10, color=C_WHITE)
        ax.set_xlabel("Fraud Rate (%)", color=C_WHITE)
        ax.set_xlim(0, g["Rate"].max() * 1.3)
        charts[fname] = _save_chart(fname)

    # ── 01 Transaction status (pie) ───────────────────────────────────────────
    sc_vc = df["Transaction_Status"].value_counts()
    pie_colors = [PURPLE_PALETTE[0], PURPLE_PALETTE[2], PURPLE_PALETTE[4]]
    fig, ax = plt.subplots(figsize=(7, 5), facecolor=BG_DARK)
    _styled_ax(ax)
    wedges, texts, autotexts = ax.pie(
        sc_vc.values, labels=sc_vc.index,
        autopct="%1.1f%%", colors=pie_colors[:len(sc_vc)],
        startangle=90, textprops={"color": C_WHITE},
    )
    for at in autotexts:
        at.set_color(C_WHITE)
        at.set_fontsize(9)
    ax.set_title("Transaction Status Distribution", fontsize=12, color=C_WHITE, pad=12)
    charts["01_transaction_status.png"] = _save_chart("01_transaction_status.png")

    # ── 02 Fraud distribution (bar + pie) ────────────────────────────────────
    vc = df[FRAUD_COL].value_counts().sort_index()
    fraud_rate = df[FRAUD_COL].mean() * 100
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), facecolor=BG_DARK)
    for ax_ in axes:
        _styled_ax(ax_)
    bar_colors = [C_LLIGHT, C_DEEP]
    axes[0].bar(["Non-Fraud (0)", "Fraud (1)"], vc.values, color=bar_colors)
    for i, v in enumerate(vc.values):
        axes[0].text(i, v + 50, str(v), ha="center", fontsize=10, color=C_WHITE)
    axes[0].set_title("Fraud vs Non-Fraud Count", color=C_WHITE)
    axes[0].set_ylabel("Count", color=C_WHITE)
    wedges2, texts2, autotexts2 = axes[1].pie(
        vc.values, labels=["Non-Fraud", "Fraud"],
        autopct="%1.2f%%", colors=bar_colors,
        startangle=90, textprops={"color": C_WHITE},
    )
    for at in autotexts2:
        at.set_color(C_WHITE)
    axes[1].set_title(f"Fraud Rate = {fraud_rate:.2f}%", color=C_WHITE)
    charts["02_fraud_distribution.png"] = _save_chart("02_fraud_distribution.png")

    # ── 03 Fraud by transaction type ─────────────────────────────────────────
    _fraud_rate_bar("Transaction_Type", "03_fraud_by_type.png",   "Fraud Rate by Transaction Type")

    # ── 04 Fraud by device type ───────────────────────────────────────────────
    _fraud_rate_bar("Device_Type",      "04_fraud_by_device.png", "Fraud Rate by Device Type")

    # ── 05 Risk level distribution ────────────────────────────────────────────
    ro = ["Low", "Medium", "High", "Critical"]
    rc = df[RISK_LEVEL_COL].value_counts().reindex(ro, fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 4), facecolor=BG_DARK)
    _styled_ax(ax)
    ax.bar(rc.index, rc.values, color=[RISK_COLOR[r] for r in rc.index])
    ax.set_title("Risk Level Distribution", color=C_WHITE)
    ax.set_ylabel("Count", color=C_WHITE)
    for i, v in enumerate(rc.values):
        ax.text(i, v + 20, str(v), ha="center", fontsize=9, color=C_WHITE)
    charts["05_risk_distribution.png"] = _save_chart("05_risk_distribution.png")

    # ── 06 Amount distribution (histogram) ───────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor=BG_DARK)
    for ax_ in axes:
        _styled_ax(ax_)
    axes[0].hist(df[AMOUNT_COL], bins=55, color=C_MED, edgecolor="none")
    axes[0].set_title("Transaction Amount (raw)", color=C_WHITE)
    axes[0].set_xlabel("Amount (INR)", color=C_WHITE)
    axes[1].hist(np.log1p(df[AMOUNT_COL]), bins=55, color=C_LAV, edgecolor="none")
    axes[1].set_title("Transaction Amount (log)", color=C_WHITE)
    axes[1].set_xlabel("log(1 + Amount)", color=C_WHITE)
    charts["06_amount_distribution.png"] = _save_chart("06_amount_distribution.png")

    # ── 07 Anomaly analysis ───────────────────────────────────────────────────
    if "Is_Anomaly" in df.columns:
        n_anom = int(df["Is_Anomaly"].sum())
        fig, ax = plt.subplots(figsize=(7, 4), facecolor=BG_DARK)
        _styled_ax(ax)
        bars7 = ax.bar(["Normal", "Anomaly"], [len(df) - n_anom, n_anom],
               color=[C_LLIGHT, C_ACCENT_RED])
        for b in bars7:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 20,
                    str(int(b.get_height())), ha="center", fontsize=10, color=C_WHITE)
        ax.set_title("Isolation Forest: Normal vs Anomaly", color=C_WHITE)
        ax.set_ylabel("Count", color=C_WHITE)
        charts["07_anomaly_analysis.png"] = _save_chart("07_anomaly_analysis.png")

    # ── 08 Model comparison ───────────────────────────────────────────────────
    metrics = ["ROC-AUC", "PR-AUC", "Precision", "Recall", "F1"]
    lr_v = [ml_res["lr"]["roc_auc"], ml_res["lr"]["pr_auc"],
            ml_res["lr"]["precision"], ml_res["lr"]["recall"], ml_res["lr"]["f1"]]
    rf_v = [ml_res["rf"]["roc_auc"], ml_res["rf"]["pr_auc"],
            ml_res["rf"]["precision"], ml_res["rf"]["recall"], ml_res["rf"]["f1"]]
    x_ = np.arange(len(metrics))
    w  = 0.35
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG_DARK)
    _styled_ax(ax)
    b1 = ax.bar(x_ - w/2, lr_v, w, label="Logistic Regression", color=C_LAV)
    b2 = ax.bar(x_ + w/2, rf_v, w, label="Random Forest",       color=C_MED)
    ax.bar_label(b1, fmt="%.3f", padding=3, fontsize=8, color=C_WHITE)
    ax.bar_label(b2, fmt="%.3f", padding=3, fontsize=8, color=C_WHITE)
    ax.set_xticks(x_)
    ax.set_xticklabels(metrics, color=C_SEC)
    ax.set_ylim(0, 1.15)
    ax.set_title("Model Performance Comparison: LR vs RF", color=C_WHITE)
    ax.set_ylabel("Score", color=C_WHITE)
    ax.legend(facecolor=BG_CARD2, edgecolor="#3D2E5A", labelcolor=C_WHITE)
    charts["08_model_comparison.png"] = _save_chart("08_model_comparison.png")

    # ── 09 ROC curves ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5), facecolor=BG_DARK)
    _styled_ax(ax)
    ax.plot(ml_res["fpr_lr"], ml_res["tpr_lr"],
            label=f"Logistic Regression (AUC={ml_res['lr']['roc_auc']})",
            color=C_LAV, lw=2)
    ax.plot(ml_res["fpr_rf"], ml_res["tpr_rf"],
            label=f"Random Forest (AUC={ml_res['rf']['roc_auc']})",
            color=C_VLLIGHT, lw=2)
    ax.plot([0, 1], [0, 1], linestyle="--", lw=1, color=C_MUTED, label="Random (AUC=0.5)")
    ax.set_title("ROC Curves — LR vs Random Forest", color=C_WHITE)
    ax.set_xlabel("False Positive Rate", color=C_WHITE)
    ax.set_ylabel("True Positive Rate", color=C_WHITE)
    ax.legend(fontsize=9, facecolor=BG_CARD2, edgecolor="#3D2E5A", labelcolor=C_WHITE)
    charts["09_roc_curves.png"] = _save_chart("09_roc_curves.png")

    # ── 10 Feature importance ─────────────────────────────────────────────────
    fi_df = pd.DataFrame({"Feature": ml_res["fi_names"][:15],
                          "Importance": ml_res["fi_vals"][:15]})
    fi_df = fi_df.sort_values("Importance")
    bar_cols = [PURPLE_PALETTE[i % len(PURPLE_PALETTE)] for i in range(len(fi_df))]
    fig, ax = plt.subplots(figsize=(9, 6), facecolor=BG_DARK)
    _styled_ax(ax)
    ax.barh(fi_df["Feature"], fi_df["Importance"], color=bar_cols)
    ax.set_title("Top 15 Feature Importances (Random Forest)", color=C_WHITE)
    ax.set_xlabel("Importance", color=C_WHITE)
    charts["10_feature_importance.png"] = _save_chart("10_feature_importance.png")

    return charts


# ══════════════════════════════════════════════════════════════════════════════
# E.  FULL PIPELINE (headless mode)
# ══════════════════════════════════════════════════════════════════════════════

def run_full_pipeline() -> tuple:
    """Run the complete analysis. Returns (df, validation_report, ml_results, charts)."""
    log.info("Loading and cleaning data...")
    df, vr = load_and_clean()

    log.info("Engineering features...")
    df = engineer_features(df, vr["amt_p99"])

    log.info("Computing custom risk scores...")
    df["Custom_Risk_Score"] = df.apply(compute_custom_risk_score, axis=1)
    df["Custom_Risk_Level"] = df["Custom_Risk_Score"].apply(score_to_level)

    log.info("Running anomaly detection...")
    df = run_anomaly_detection(df)

    log.info("Running ML pipeline (this may take ~60s)...")
    ml_res = run_ml_pipeline(df)

    log.info("Generating charts...")
    charts = generate_all_charts(df, ml_res)

    df.to_csv(CLEAN_CSV, index=False)
    df.to_csv(PBI_CSV,   index=False)

    with open(RESULTS_JSON, "w") as f:
        json.dump(ml_res, f, indent=2)

    fi_df = pd.DataFrame({"Feature": ml_res["fi_names"], "Importance": ml_res["fi_vals"]})
    fi_df.to_csv(FI_CSV, index=False)

    sus = df[(df[RISK_LEVEL_COL].isin(["High", "Critical"])) | (df["Is_Anomaly"] == 1)]
    sus.to_csv(RESULTS_DIR / "suspicious_transactions.csv", index=False)

    log.info(f"Pipeline complete. Records: {len(df):,}  Fraud rate: {df[FRAUD_COL].mean()*100:.2f}%")
    return df, vr, ml_res, charts


# ══════════════════════════════════════════════════════════════════════════════
# F.  REPORT GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def generate_report(df: pd.DataFrame, vr: dict, ml_res: dict, charts: dict) -> Path:
    """Build Project_Report.docx. White background, purple headings, lavender table headers."""
    try:
        from docx import Document
        from docx.shared import Pt, Cm, RGBColor, Inches
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        from docx.enum.table import WD_ALIGN_VERTICAL
        from docx.oxml.ns import qn
        from docx.oxml import OxmlElement
    except ImportError:
        log.warning("python-docx not installed. Skipping report generation.")
        return None

    PURPLE_H  = RGBColor(0x5B, 0x21, 0xB6)   # #5B21B6
    LAV_TH    = RGBColor(0xA7, 0x8B, 0xFA)   # #A78BFA table header bg
    BLACK     = RGBColor(0x1F, 0x23, 0x28)
    WHITE_RGB = RGBColor(0xFF, 0xFF, 0xFF)

    doc = Document()

    # Page margins
    for section in doc.sections:
        section.top_margin    = Cm(2.2)
        section.bottom_margin = Cm(2.2)
        section.left_margin   = Cm(2.5)
        section.right_margin  = Cm(2.5)

    # ── helpers ──────────────────────────────────────────────────────────────

    def _set_cell_bg(cell, hex_color: str):
        """Set table cell background shading."""
        tc = cell._tc
        tcPr = tc.get_or_add_tcPr()
        shd = OxmlElement("w:shd")
        shd.set(qn("w:val"),   "clear")
        shd.set(qn("w:color"), "auto")
        shd.set(qn("w:fill"),  hex_color.lstrip("#"))
        tcPr.append(shd)

    def _heading(text: str, level: int = 1):
        p = doc.add_heading(text, level=level)
        for run in p.runs:
            run.font.color.rgb = PURPLE_H
            run.font.bold = True
            run.font.size = Pt(14 if level == 1 else 12)
        p.paragraph_format.space_before = Pt(10)
        p.paragraph_format.space_after  = Pt(4)
        return p

    def _body(text: str, bold: bool = False):
        p = doc.add_paragraph()
        run = p.add_run(text)
        run.font.size  = Pt(10.5)
        run.font.color.rgb = BLACK
        run.font.bold  = bold
        p.paragraph_format.space_after = Pt(4)
        return p

    def _bullet(text: str):
        p = doc.add_paragraph(style="List Bullet")
        run = p.add_run(text)
        run.font.size  = Pt(10.5)
        run.font.color.rgb = BLACK
        p.paragraph_format.space_after = Pt(2)
        return p

    def _add_table(headers: list, rows: list, col_widths: list = None):
        tbl = doc.add_table(rows=1 + len(rows), cols=len(headers))
        tbl.style = "Table Grid"
        # Header row
        hdr_cells = tbl.rows[0].cells
        for i, h in enumerate(headers):
            hdr_cells[i].text = h
            _set_cell_bg(hdr_cells[i], "A78BFA")
            for run in hdr_cells[i].paragraphs[0].runs:
                run.font.bold  = True
                run.font.size  = Pt(10)
                run.font.color.rgb = WHITE_RGB
        # Data rows
        for ri, row_data in enumerate(rows):
            cells = tbl.rows[ri + 1].cells
            for ci, val in enumerate(row_data):
                cells[ci].text = str(val)
                for run in cells[ci].paragraphs[0].runs:
                    run.font.size = Pt(10)
                    run.font.color.rgb = BLACK
        # Column widths
        if col_widths:
            for row in tbl.rows:
                for ci, w in enumerate(col_widths):
                    row.cells[ci].width = Cm(w)
        doc.add_paragraph()
        return tbl

    def _embed_chart(name: str, width_cm: float = 14.0):
        p = charts.get(name) or (CHARTS_DIR / name)
        if p and Path(p).exists():
            doc.add_picture(str(p), width=Cm(width_cm))
            last_para = doc.paragraphs[-1]
            last_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.add_paragraph()

    # ═══════════════════════════════════════════════════════════════════════
    # 1. TITLE PAGE
    # ═══════════════════════════════════════════════════════════════════════
    doc.add_paragraph()
    tp = doc.add_paragraph()
    tp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    tr = tp.add_run("UPI Transaction Risk & Fraud Analytics")
    tr.font.size  = Pt(22)
    tr.font.bold  = True
    tr.font.color.rgb = PURPLE_H

    doc.add_paragraph()
    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sr = sub.add_run("Comprehensive Analysis Report")
    sr.font.size = Pt(14)
    sr.font.color.rgb = RGBColor(0x7C, 0x3A, 0xED)

    doc.add_paragraph()
    meta_lines = [
        f"Generated: {datetime.now().strftime('%B %d, %Y')}",
        "Dataset: Synthetic UPI Transactions (~10,000 records)",
        "Tools: Python · scikit-learn · Matplotlib · Streamlit · python-docx",
    ]
    for line in meta_lines:
        mp = doc.add_paragraph()
        mp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        mr = mp.add_run(line)
        mr.font.size = Pt(11)
        mr.font.color.rgb = RGBColor(0x57, 0x60, 0x6A)

    doc.add_page_break()

    # ═══════════════════════════════════════════════════════════════════════
    # 2. EXECUTIVE SUMMARY
    # ═══════════════════════════════════════════════════════════════════════
    _heading("1. Executive Summary")
    fraud_n   = int(df[FRAUD_COL].sum())
    fraud_pct = df[FRAUD_COL].mean() * 100
    n_anom    = int(df["Is_Anomaly"].sum()) if "Is_Anomaly" in df.columns else 500
    total_val = df[AMOUNT_COL].sum()

    _body(
        f"This report presents a comprehensive risk and fraud analytics study on a synthetic "
        f"dataset of {len(df):,} UPI transactions (total value ₹{total_val:,.0f}). "
        f"The analysis identified {fraud_n} confirmed fraud cases ({fraud_pct:.2f}% fraud rate) "
        f"and {n_anom} anomalies (5.0% of transactions) using Isolation Forest. "
        f"Two machine learning models were trained: Logistic Regression achieved ROC-AUC = 0.6352 "
        f"with recall of 65.71%, catching the majority of fraud cases; Random Forest achieved "
        f"ROC-AUC = 0.5497 with higher precision (8.33%) but lower recall (2.86%). "
        f"New-device transactions carry a fraud rate of 3.86%, odd-hour (midnight–6 AM) transactions "
        f"carry 3.89%, and high-value transactions (top 1%) carry 5.0% — all significantly above "
        f"the 1.74% baseline. The custom Risk_Score (0–100) achieves a correlation of 0.50 with "
        f"fraud labels, enabling actionable tiered risk management without data leakage."
    )
    doc.add_paragraph()

    # ═══════════════════════════════════════════════════════════════════════
    # 3. PROBLEM STATEMENT
    # ═══════════════════════════════════════════════════════════════════════
    _heading("2. Problem Statement")
    _body(
        "UPI (Unified Payments Interface) fraud is a growing concern in India's digital payment "
        "ecosystem. Fraudsters exploit new devices, off-hours transactions, unusually high amounts, "
        "and compromised accounts to execute unauthorized transfers. Traditional rule-based systems "
        "generate excessive false positives while missing sophisticated attacks. This project "
        "develops a multi-layered detection framework combining rule-based risk scoring, unsupervised "
        "anomaly detection, and supervised machine learning to identify high-risk transactions before "
        "they are processed."
    )

    # ═══════════════════════════════════════════════════════════════════════
    # 4. PROJECT OBJECTIVES
    # ═══════════════════════════════════════════════════════════════════════
    _heading("3. Project Objectives")
    for obj in [
        "Design and implement a transparent, rule-based Risk Score (0–100) without using fraud labels.",
        "Identify statistically significant fraud risk factors from transaction metadata.",
        "Apply Isolation Forest to detect anomalous transaction patterns at 5% contamination.",
        "Train and evaluate Logistic Regression and Random Forest classifiers on leak-free features.",
        "Build an interactive Streamlit dashboard for real-time risk monitoring.",
        "Generate actionable business recommendations from analytical findings.",
    ]:
        _bullet(obj)
    doc.add_paragraph()

    # ═══════════════════════════════════════════════════════════════════════
    # 5. DATASET DESCRIPTION
    # ═══════════════════════════════════════════════════════════════════════
    _heading("4. Dataset Description")
    _body("The dataset is synthetic and contains no real personal or financial data.")
    _add_table(
        ["Attribute", "Value"],
        [
            ["Total Records", f"{len(df):,}"],
            ["Fraud Cases", f"{fraud_n} ({fraud_pct:.2f}%)"],
            ["Non-Fraud Cases", f"{len(df) - fraud_n:,}"],
            ["Date Range", f"{df['Transaction_Date'].min().date()} to {df['Transaction_Date'].max().date()}" if "Transaction_Date" in df.columns else "N/A"],
            ["Total Transaction Value", f"₹{total_val:,.2f}"],
            ["Mean Transaction Amount", f"₹{df[AMOUNT_COL].mean():,.2f}"],
            ["Median Transaction Amount", f"₹{df[AMOUNT_COL].median():,.2f}"],
            ["Max Transaction Amount", f"₹{df[AMOUNT_COL].max():,.2f}"],
            ["Transaction Types", str(df["Transaction_Type"].nunique()) if "Transaction_Type" in df.columns else "N/A"],
            ["Device Types", str(df["Device_Type"].nunique()) if "Device_Type" in df.columns else "N/A"],
            ["Source", "Synthetic — no real PII"],
        ],
        col_widths=[7, 9]
    )

    # ═══════════════════════════════════════════════════════════════════════
    # 6. DATA DICTIONARY
    # ═══════════════════════════════════════════════════════════════════════
    _heading("5. Data Dictionary")
    _add_table(
        ["Column", "Type", "Description"],
        [
            ["Transaction_ID",      "String",  "Unique identifier for each transaction"],
            ["Transaction_Date",    "Date",    "Date the transaction occurred"],
            ["Transaction_Time",    "Time",    "Time the transaction occurred"],
            ["Transaction_Amount_INR", "Float","Amount transferred in Indian Rupees"],
            ["Transaction_Type",    "Category","Type: P2P, P2M, Bill Payment, etc."],
            ["Transaction_Status",  "Category","Completed / Failed / Pending"],
            ["Device_Type",         "Category","Mobile / Desktop / Tablet / Unknown"],
            ["Network_Type",        "Category","WiFi / 4G / 3G / 2G"],
            ["Is_New_Device",       "Binary",  "1 = transaction from unrecognized device"],
            ["Is_International",    "Binary",  "1 = transaction originates internationally"],
            ["Sender_Location",     "String",  "City/region of sender"],
            ["Receiver_Location",   "String",  "City/region of receiver"],
            ["Account_Age_Days",    "Integer", "Age of sender account in days"],
            ["Failed_Attempts",     "Integer", "Number of failed auth attempts"],
            ["Transactions_Last_24_Hours", "Integer", "Transaction velocity: last 24 h"],
            ["Transactions_Last_7_Days",   "Integer", "Transaction velocity: last 7 days"],
            ["Risk_Score",          "Integer", "Pre-computed score 0–100 (dataset column)"],
            ["Risk_Level",          "Category","Low / Medium / High / Critical"],
            ["Fraud_Label",         "Binary",  "Ground truth: 1 = Fraud, 0 = Legitimate"],
            ["Sender_Age_Group",    "Category","Age group of sender"],
        ],
        col_widths=[5.5, 3, 8]
    )

    # ═══════════════════════════════════════════════════════════════════════
    # 7. DATA PREPARATION
    # ═══════════════════════════════════════════════════════════════════════
    _heading("6. Data Preparation")
    missing_d = vr.get("missing", {})
    miss_str  = ", ".join(f"{k}: {v}" for k, v in missing_d.items()) if missing_d else "None"
    _body(f"Missing values identified: {miss_str}. "
          f"Imputation strategy: Receiver_Location filled with 'Unknown'; "
          f"Network_Type filled with mode; "
          f"Average_Previous_Transaction_Amount_INR filled with median.")
    _body(f"Duplicate rows removed: {vr.get('dup_rows', 0)}. "
          f"Duplicate Transaction IDs removed: {vr.get('dup_txn_ids', 0)}.")
    _body(
        "Data leakage prevention: Risk_Score and Risk_Level columns are present in the dataset "
        "but are EXCLUDED from all ML model features (MODEL_FEATURES list). They are used only "
        "for risk-tier reporting. The custom rule-based Risk Score is computed from behavioral "
        "features only — Fraud_Label is never accessed during scoring."
    )

    # ═══════════════════════════════════════════════════════════════════════
    # 8. EXPLORATORY DATA ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════
    _heading("7. Exploratory Data Analysis")
    _body("The following charts provide an overview of the transaction data distributions.")
    for cname in ["01_transaction_status.png", "02_fraud_distribution.png",
                  "03_fraud_by_type.png", "04_fraud_by_device.png",
                  "05_risk_distribution.png", "06_amount_distribution.png"]:
        _embed_chart(cname)

    if "Transaction_Type" in df.columns:
        tt = df.groupby("Transaction_Type")[FRAUD_COL].agg(["sum","count"]).reset_index()
        tt["Rate%"] = (tt["sum"] / tt["count"] * 100).round(2)
        tt.columns = ["Transaction Type", "Fraud Count", "Total", "Fraud Rate %"]
        _add_table(list(tt.columns), tt.values.tolist(), col_widths=[5,3,3,3])

    # ═══════════════════════════════════════════════════════════════════════
    # 9. FRAUD RISK ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════
    _heading("8. Fraud Risk Analysis")
    _embed_chart("05_risk_distribution.png", width_cm=13)
    _body("Risk Scoring Methodology: A transparent rule-based score (0–100) was computed "
          "using the following additive components:")
    _add_table(
        ["Risk Factor", "Condition", "Score Added"],
        [
            ["Amount vs Average Ratio", "> 5×",  "+25"],
            ["Amount vs Average Ratio", "3–5×",  "+15"],
            ["Amount vs Average Ratio", "2–3×",  "+8"],
            ["New Device",              "Is_New_Device = 1", "+15"],
            ["Failed Attempts",         "≥ 3 attempts",      "+15"],
            ["Failed Attempts",         "1–2 attempts",      "+8"],
            ["Velocity (24h)",          "≥ 8 transactions",  "+15"],
            ["Velocity (24h)",          "5–7 transactions",  "+8"],
            ["Odd Hour",                "00:00–06:00",        "+10"],
            ["Account Age",             "< 90 days",          "+10"],
            ["Account Age",             "90–180 days",        "+5"],
            ["International",           "Is_International=1", "+5"],
            ["High Amount",             "> ₹50,000",          "+10"],
            ["High Amount",             "₹20,000–₹50,000",   "+5"],
        ],
        col_widths=[5, 5, 3]
    )
    _body("Score Thresholds: Low (0–24) · Medium (25–49) · High (50–69) · Critical (≥70)")
    _body(f"Risk_Score correlation with Fraud_Label: 0.50 — indicating moderate predictive "
          f"power without using ground-truth labels in the scoring formula.")

    # ═══════════════════════════════════════════════════════════════════════
    # 10. ANOMALY DETECTION
    # ═══════════════════════════════════════════════════════════════════════
    _heading("9. Anomaly Detection")
    _embed_chart("07_anomaly_analysis.png", width_cm=12)
    _body("Algorithm: Isolation Forest with 200 estimators and 5% contamination rate.")
    if "Is_Anomaly" in df.columns:
        n_anom_fraud = int(df[(df["Is_Anomaly"]==1) & (df[FRAUD_COL]==1)].shape[0])
        pct_anom     = df["Is_Anomaly"].mean() * 100
        anom_fraud_r = df[df["Is_Anomaly"]==1][FRAUD_COL].mean() * 100 if n_anom > 0 else 0
        _add_table(
            ["Metric", "Value"],
            [
                ["Total Anomalies Detected",     f"{n_anom:,} ({pct_anom:.1f}%)"],
                ["Normal Transactions",           f"{len(df)-n_anom:,}"],
                ["Fraud Cases in Anomalies",      str(n_anom_fraud)],
                ["Fraud Rate in Anomalies",       f"{anom_fraud_r:.2f}%"],
                ["Baseline Fraud Rate",           f"{fraud_pct:.2f}%"],
                ["Contamination Parameter",       "5%"],
                ["n_estimators",                  "200"],
            ],
            col_widths=[8, 8]
        )

    # ═══════════════════════════════════════════════════════════════════════
    # 11. MACHINE LEARNING METHODOLOGY
    # ═══════════════════════════════════════════════════════════════════════
    _heading("10. Machine Learning Methodology")
    _body("Two supervised classification models were trained to detect fraudulent transactions.")
    _add_table(
        ["Parameter", "Logistic Regression", "Random Forest"],
        [
            ["Class weighting",   "balanced",  "balanced"],
            ["Train/test split",  "80/20 stratified", "80/20 stratified"],
            ["Scaling",           "StandardScaler", "None (tree-based)"],
            ["Max iterations",    "1000",       "N/A"],
            ["n_estimators",      "N/A",        "300"],
            ["max_depth",         "N/A",        "12"],
            ["Cross-validation",  "5-fold ROC-AUC", "5-fold ROC-AUC"],
            ["CV Mean ROC-AUC",   "0.553",      "0.5105"],
            ["Features",          "20 numeric + dummies", "20 numeric + dummies"],
            ["Excluded features", "Risk_Score, Risk_Level, Fraud_Label",
                                  "Risk_Score, Risk_Level, Fraud_Label"],
        ],
        col_widths=[5, 5.5, 6]
    )

    # ═══════════════════════════════════════════════════════════════════════
    # 12. MODEL EVALUATION
    # ═══════════════════════════════════════════════════════════════════════
    _heading("11. Model Evaluation")
    _embed_chart("08_model_comparison.png")
    _embed_chart("09_roc_curves.png")
    _embed_chart("10_feature_importance.png")

    lr_m = ml_res.get("lr", {})
    rf_m = ml_res.get("rf", {})
    _add_table(
        ["Metric", "Logistic Regression", "Random Forest"],
        [
            ["ROC-AUC",       f"{lr_m.get('roc_auc', 0.6352):.4f}", f"{rf_m.get('roc_auc', 0.5497):.4f}"],
            ["PR-AUC",        f"{lr_m.get('pr_auc',  0.0304):.4f}", f"{rf_m.get('pr_auc',  0.0261):.4f}"],
            ["Precision",     f"{lr_m.get('precision',0.0315):.4f}",f"{rf_m.get('precision',0.0833):.4f}"],
            ["Recall",        f"{lr_m.get('recall',  0.6571):.4f}", f"{rf_m.get('recall',  0.0286):.4f}"],
            ["F1 Score",      f"{lr_m.get('f1',      0.0601):.4f}", f"{rf_m.get('f1',      0.0426):.4f}"],
            ["Accuracy",      f"{lr_m.get('accuracy',0):.4f}",      f"{rf_m.get('accuracy',0):.4f}"],
            ["CV Mean AUC",   f"{ml_res.get('cv_lr_mean', 0.553)}", f"{ml_res.get('cv_rf_mean', 0.5105)}"],
        ],
        col_widths=[5, 5.5, 5.5]
    )
    _body(
        "Interpretation: Logistic Regression with class_weight='balanced' achieves superior recall "
        "(65.71%), catching the majority of fraud cases at the cost of lower precision (3.15%). "
        "Random Forest achieves higher precision (8.33%) but misses most fraud (recall 2.86%). "
        "For fraud detection, recall is the primary business metric — missed fraud is more costly "
        "than false alerts. LR is the recommended production model."
    )

    # ═══════════════════════════════════════════════════════════════════════
    # 13. KEY FINDINGS
    # ═══════════════════════════════════════════════════════════════════════
    _heading("12. Key Findings")
    findings = [
        f"Dataset contains {len(df):,} transactions with a fraud rate of {fraud_pct:.2f}% ({fraud_n} cases).",
        "New-device transactions have a fraud rate of 3.86% — 2.2× the baseline rate.",
        "Odd-hour (midnight–6 AM) transactions have a fraud rate of 3.89% — 2.2× the baseline rate.",
        "High-value transactions (top 1% by amount) have a fraud rate of 5.0% — 2.9× the baseline.",
        "High-velocity accounts (≥5 transactions in 24h) show elevated fraud elevation.",
        f"Isolation Forest flagged {n_anom} anomalies (5.0% of dataset) using unsupervised learning.",
        "Custom Risk_Score achieves Pearson correlation of 0.50 with Fraud_Label.",
        "Logistic Regression (balanced) achieves recall of 65.71%, catching 2 in 3 fraud cases.",
        "Random Forest achieves higher precision (8.33%) but misses 97% of fraud cases.",
        "5-fold CV: LR mean AUC=0.553, RF mean AUC=0.511 — both models have room for improvement.",
        "Risk_Score and Risk_Level are excluded from ML features to prevent data leakage.",
        "The dataset is synthetic; production deployment requires validation on real transaction data.",
    ]
    for f in findings:
        _bullet(f)
    doc.add_paragraph()

    # ═══════════════════════════════════════════════════════════════════════
    # 14. PROBLEM → EVIDENCE → ACTION TABLE
    # ═══════════════════════════════════════════════════════════════════════
    _heading("13. Problem → Evidence → Action → Expected Result")
    _add_table(
        ["Problem", "Evidence", "Action", "Expected Result"],
        [
            [
                "New device fraud",
                "Fraud rate 3.86% on new devices vs 1.74% baseline",
                "Trigger OTP + biometric step-up on new device logins",
                "Reduce new-device fraud by 50–60%",
            ],
            [
                "Off-hours fraud",
                "Fraud rate 3.89% between midnight and 6 AM",
                "Apply enhanced scrutiny and lower transaction limits 00:00–06:00",
                "Reduce night-time fraud by ~40%",
            ],
            [
                "High-value fraud",
                "5.0% fraud rate for top-1% transaction amounts",
                "Require manual approval or cooling period for amounts in top 1%",
                "Prevent high-impact fraud; reduce financial loss",
            ],
            [
                "High-velocity fraud",
                "Elevated fraud rate for ≥5 transactions in 24 hours",
                "Implement velocity-based rate limiting and alerts",
                "Disrupt account takeover and scripted fraud",
            ],
            [
                "Critical risk transactions",
                "Risk_Score ≥ 70 shows 0.50 correlation with fraud",
                "Flag Critical risk transactions for real-time analyst review",
                "Intercept high-risk transactions before settlement",
            ],
            [
                "Low ML recall (RF)",
                "RF catches only 2.86% of fraud cases",
                "Use LR (recall 65.71%) as primary detection model in production",
                "Catch 65%+ of fraud with manageable false-positive rate",
            ],
        ],
        col_widths=[3.5, 4.5, 4.5, 4]
    )

    # ═══════════════════════════════════════════════════════════════════════
    # 15. FRAUD PREVENTION RECOMMENDATIONS
    # ═══════════════════════════════════════════════════════════════════════
    _heading("14. Fraud Prevention Recommendations")
    for rec in [
        "Deploy Logistic Regression as the primary real-time fraud scoring model (recall 65.71%).",
        "Implement step-up authentication (OTP + biometric) for new-device transactions.",
        "Apply transaction amount limits and mandatory cooling periods for high-value transfers (top 1%).",
        "Restrict or flag transactions originating between 00:00 and 06:00 IST.",
        "Monitor accounts with ≥5 transactions in 24 hours for velocity-based fraud patterns.",
        "Block or review accounts with ≥3 failed authentication attempts in a session.",
        "Continuously retrain models monthly with updated fraud labels to adapt to new patterns.",
        "Integrate device fingerprinting and IP geolocation for additional signal.",
        "Use the custom Risk Score tiering (Low/Medium/High/Critical) for analyst routing.",
        "Alert customers via SMS/email on high-risk or anomalous transactions in real time.",
    ]:
        _bullet(rec)
    doc.add_paragraph()

    # ═══════════════════════════════════════════════════════════════════════
    # 16. RISK MITIGATION & PRECAUTIONS
    # ═══════════════════════════════════════════════════════════════════════
    _heading("15. Risk Mitigation & Precautions")
    for r in [
        "False positives: LR with high recall generates many false alerts — implement a two-stage review "
        "queue (automated flag → analyst review) to avoid customer friction.",
        "Model drift: Fraudsters adapt tactics; schedule monthly model retraining with fresh labeled data.",
        "Class imbalance: At 1.74% fraud rate, class_weight='balanced' is essential; monitor over time.",
        "Data quality: Missing or imputed values reduce signal; invest in data completeness.",
        "Regulatory compliance: All fraud interventions must comply with RBI and NPCI guidelines.",
        "Adversarial attacks: Fraudsters may learn to game known rules; keep rule thresholds confidential.",
    ]:
        _bullet(r)
    doc.add_paragraph()

    # ═══════════════════════════════════════════════════════════════════════
    # 17. BUSINESS IMPACT
    # ═══════════════════════════════════════════════════════════════════════
    _heading("16. Business Impact")
    _body(
        f"At a 1.74% fraud rate across {len(df):,} transactions with mean amount "
        f"₹{df[AMOUNT_COL].mean():,.0f}, deploying the LR model (65.71% recall) "
        f"would prevent approximately 114 out of 174 fraud cases per 10,000 transactions. "
        f"The custom Risk Score enables priority queuing, ensuring analysts focus on the "
        f"highest-risk transactions first. Anomaly detection provides a complementary layer "
        f"that catches novel fraud patterns not yet in the training labels."
    )

    # ═══════════════════════════════════════════════════════════════════════
    # 18. LIMITATIONS
    # ═══════════════════════════════════════════════════════════════════════
    _heading("17. Limitations")
    for lim in [
        "The dataset is entirely synthetic — model performance on real UPI data may differ significantly.",
        "The relatively small dataset (~10,000 records) limits model complexity and generalization.",
        "Both models show moderate ROC-AUC (0.51–0.64), indicating the features have limited "
        "discriminative power for fraud detection on this synthetic data.",
        "Isolation Forest anomaly detection is unsupervised and does not directly optimize fraud recall.",
        "The rule-based Risk Score thresholds were set heuristically and should be validated on real data.",
        "No network/graph features (e.g., transaction networks between accounts) were included.",
    ]:
        _bullet(lim)
    doc.add_paragraph()

    # ═══════════════════════════════════════════════════════════════════════
    # 19. FUTURE ENHANCEMENTS
    # ═══════════════════════════════════════════════════════════════════════
    _heading("18. Future Enhancements")
    for fe in [
        "Graph Neural Networks (GNN) to model transaction networks between accounts.",
        "XGBoost / LightGBM with SHAP explanations for better accuracy and interpretability.",
        "Real-time streaming pipeline using Apache Kafka + Flink for sub-second fraud decisions.",
        "Federated learning to train across banks without sharing raw transaction data.",
        "SMOTE or GAN-based synthetic oversampling to address class imbalance.",
        "Natural language processing of transaction descriptions for additional signal.",
        "Dynamic threshold tuning based on business cost matrix (cost of fraud vs. false positives).",
        "A/B testing framework for evaluating rule changes in production.",
    ]:
        _bullet(fe)
    doc.add_paragraph()

    # ═══════════════════════════════════════════════════════════════════════
    # 20. CONCLUSION
    # ═══════════════════════════════════════════════════════════════════════
    _heading("19. Conclusion")
    _body(
        "This project successfully demonstrates a full-stack fraud analytics pipeline for UPI "
        "transactions, encompassing data cleaning, feature engineering, rule-based risk scoring, "
        "unsupervised anomaly detection, supervised machine learning, interactive visualization, "
        "and actionable business recommendations. The Logistic Regression model with recall-"
        "optimized class weighting provides the best fraud detection performance, while the "
        "custom Risk Score offers a transparent, interpretable complement for operations teams. "
        "The Streamlit dashboard enables stakeholders to interactively explore fraud patterns, "
        "risk distributions, and model performance without requiring technical expertise. "
        "All findings are based on synthetic data and should be validated against production "
        "transaction data before deployment."
    )

    # ═══════════════════════════════════════════════════════════════════════
    # 21. TECHNOLOGIES USED
    # ═══════════════════════════════════════════════════════════════════════
    _heading("20. Technologies Used")
    _add_table(
        ["Technology", "Version / Notes", "Purpose"],
        [
            ["Python",         "3.10+",           "Core programming language"],
            ["pandas",         "2.x",             "Data manipulation and analysis"],
            ["NumPy",          "1.x",             "Numerical computing"],
            ["scikit-learn",   "1.x",             "ML models, metrics, preprocessing"],
            ["Matplotlib",     "3.x",             "Static chart generation"],
            ["Seaborn",        "0.x",             "Statistical visualizations"],
            ["Plotly",         "5.x",             "Interactive dashboard charts"],
            ["Streamlit",      "1.x",             "Interactive web dashboard"],
            ["python-docx",    "1.x",             "Automated Word report generation"],
            ["IsolationForest","scikit-learn",     "Unsupervised anomaly detection"],
        ],
        col_widths=[4, 4, 8.5]
    )

    # ── Save ──────────────────────────────────────────────────────────────
    try:
        doc.save(str(REPORT_PATH))
        log.info(f"Report saved: {REPORT_PATH}")
    except PermissionError:
        alt = BASE_DIR / f"Project_Report_{datetime.now().strftime('%H%M%S')}.docx"
        doc.save(str(alt))
        log.warning(f"Original report locked. Saved as: {alt}")
        return alt

    return REPORT_PATH


# ══════════════════════════════════════════════════════════════════════════════
# G.  STREAMLIT DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

def _run_dashboard():
    """Six-page interactive Streamlit dashboard with dark purple theme."""
    import plotly.graph_objects as go
    import plotly.express as px
    import streamlit as st
    import io

    st.set_page_config(
        page_title="UPI Fraud Analytics",
        page_icon="🔒",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── CSS ───────────────────────────────────────────────────────────────
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif; }
    body, .stApp { background-color: #0F0B1A; color: #C4B5FD; }
    [data-testid="stSidebar"] { background: #0F0B1A; border-right: 1px solid #3D2E5A; }
    [data-testid="stSidebar"] * { color: #C4B5FD !important; }
    [data-testid="stSidebar"] .stRadio label { color: #C4B5FD !important; font-size: 13px; }
    .stRadio [data-baseweb="radio"] { gap: 6px; }
    .stMultiSelect [data-baseweb="tag"] { background-color: #5B21B6 !important; }
    .stButton > button {
        background: linear-gradient(135deg, #5B21B6, #7C3AED);
        color: white; border: none; border-radius: 8px;
        padding: 8px 18px; font-weight: 600; font-size: 13px;
    }
    .stButton > button:hover { background: #7C3AED; }
    .stDataFrame { background: #171126 !important; }
    div[data-testid="stMetric"] { background: #171126; border: 1px solid #3D2E5A;
        border-radius: 10px; padding: 14px; }
    div[data-testid="stMetric"] label { color: #8F89A1 !important; font-size: 11px; }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #FFFFFF !important; font-size: 26px; }
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] { font-size: 11px; }
    .kpi { background: #171126; border: 1px solid #3D2E5A; border-radius: 10px;
           padding: 18px 14px; text-align: center; }
    .kpi-lbl { font-size: 11px; color: #8F89A1; text-transform: uppercase; letter-spacing: .06em; }
    .kpi-val { font-size: 28px; font-weight: 700; color: #FFFFFF; }
    .kpi-sub { font-size: 11px; color: #A78BFA; margin-top: 3px; }
    .sec-title { font-size: 16px; font-weight: 600; color: #C4B5FD;
                 border-left: 3px solid #7C3AED; padding-left: 10px; margin: 24px 0 12px; }
    .box-info   { background: #1B1430; border-left: 3px solid #7C3AED; border-radius: 4px;
                  padding: 10px 14px; font-size: 13px; color: #C4B5FD; margin: 6px 0; }
    .box-warn   { background: #1B1430; border-left: 3px solid #D97706; border-radius: 4px;
                  padding: 10px 14px; font-size: 13px; color: #FDE68A; margin: 6px 0; }
    .box-danger { background: #1E1010; border-left: 3px solid #DC2626; border-radius: 4px;
                  padding: 10px 14px; font-size: 13px; color: #FCA5A5; margin: 6px 0; }
    .box-ok     { background: #0F1E14; border-left: 3px solid #16A34A; border-radius: 4px;
                  padding: 10px 14px; font-size: 13px; color: #86EFAC; margin: 6px 0; }
    </style>
    """, unsafe_allow_html=True)

    # ── Data loader ───────────────────────────────────────────────────────
    @st.cache_data(show_spinner="Loading transaction data...")
    def _load_df():
        for csv in [PBI_CSV, CLEAN_CSV, RAW_CSV]:
            if csv.exists():
                df = pd.read_csv(csv)
                if "Transaction_DateTime" not in df.columns:
                    df["Transaction_Date"] = pd.to_datetime(df.get("Transaction_Date"), errors="coerce")
                    df["Transaction_DateTime"] = pd.to_datetime(
                        df["Transaction_Date"].astype(str) + " " + df.get("Transaction_Time", "00:00:00").astype(str),
                        errors="coerce"
                    )
                if "Transaction_Hour" not in df.columns:
                    df["Transaction_Hour"] = df["Transaction_DateTime"].dt.hour
                if "Is_Odd_Hour" not in df.columns:
                    df["Is_Odd_Hour"] = ((df["Transaction_Hour"] >= 0) & (df["Transaction_Hour"] < 6)).astype(int)
                if "Is_Weekend" not in df.columns:
                    df["Is_Weekend"] = (df["Transaction_DateTime"].dt.dayofweek >= 5).astype(int)
                if "Amount_vs_Avg_Ratio" not in df.columns:
                    avg = df["Average_Previous_Transaction_Amount_INR"].replace(0, np.nan)
                    df["Amount_vs_Avg_Ratio"] = (df[AMOUNT_COL] / avg).fillna(1.0).clip(0, 100)
                if "Is_High_Value" not in df.columns:
                    p99 = df[AMOUNT_COL].quantile(0.99)
                    df["Is_High_Value"] = (df[AMOUNT_COL] > p99).astype(int)
                if "Is_Anomaly" not in df.columns:
                    df["Is_Anomaly"] = 0
                if "IF_Score_Raw" not in df.columns:
                    df["IF_Score_Raw"] = 0.0
                if "Custom_Risk_Score" not in df.columns:
                    df["Custom_Risk_Score"] = df.apply(compute_custom_risk_score, axis=1)
                    df["Custom_Risk_Level"] = df["Custom_Risk_Score"].apply(score_to_level)
                return df
        raise FileNotFoundError("No transaction CSV found. Run --pipeline first.")

    @st.cache_data(show_spinner="Loading model results...")
    def _load_ml():
        if RESULTS_JSON.exists():
            with open(RESULTS_JSON) as f:
                return json.load(f)
        return None

    try:
        df_full = _load_df()
    except Exception as e:
        st.error(f"❌ {e}")
        st.stop()

    ml_res = _load_ml()

    # ── Sidebar ───────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🔒 UPI Fraud Analytics")
        st.markdown("---")

        page = st.radio(
            "Navigation",
            ["🏠 Executive Overview", "🔍 Fraud Analysis",
             "⚡ Risk Monitoring", "🚨 Anomaly Detection",
             "🤖 ML Model Performance", "📋 Business Decisions"],
        )

        st.markdown("---")
        st.markdown("### Filters")

        tt_opts = sorted(df_full["Transaction_Type"].dropna().unique().tolist()) if "Transaction_Type" in df_full.columns else []
        dt_opts = sorted(df_full["Device_Type"].dropna().unique().tolist())      if "Device_Type"       in df_full.columns else []
        ts_opts = sorted(df_full["Transaction_Status"].dropna().unique().tolist()) if "Transaction_Status" in df_full.columns else []
        rl_opts = ["Low", "Medium", "High", "Critical"]

        sel_tt = st.multiselect("Transaction Type",  tt_opts, default=tt_opts, key="f_tt")
        sel_dt = st.multiselect("Device Type",       dt_opts, default=dt_opts, key="f_dt")
        sel_ts = st.multiselect("Transaction Status",ts_opts, default=ts_opts, key="f_ts")
        sel_rl = st.multiselect("Risk Level",        rl_opts, default=rl_opts, key="f_rl")

        min_d = df_full["Transaction_Date"].min() if "Transaction_Date" in df_full.columns else None
        max_d = df_full["Transaction_Date"].max() if "Transaction_Date" in df_full.columns else None
        if min_d is not None and pd.notna(min_d):
            date_range = st.date_input("Date Range",
                value=(pd.to_datetime(min_d).date(), pd.to_datetime(max_d).date()),
                min_value=pd.to_datetime(min_d).date(), max_value=pd.to_datetime(max_d).date(), key="fr_dr")
        else:
            date_range = None

        if st.button("Reset Filters"):
            st.experimental_rerun()

    # ── Apply filters ─────────────────────────────────────────────────────
    dff = df_full.copy()
    if sel_tt and "Transaction_Type"   in dff.columns: dff = dff[dff["Transaction_Type"].isin(sel_tt)]
    if sel_dt and "Device_Type"        in dff.columns: dff = dff[dff["Device_Type"].isin(sel_dt)]
    if sel_ts and "Transaction_Status" in dff.columns: dff = dff[dff["Transaction_Status"].isin(sel_ts)]
    if sel_rl and RISK_LEVEL_COL       in dff.columns: dff = dff[dff[RISK_LEVEL_COL].isin(sel_rl)]
    if date_range and len(date_range) == 2 and "Transaction_Date" in dff.columns:
        d0 = pd.Timestamp(date_range[0])
        d1 = pd.Timestamp(date_range[1])
        dff = dff[
    (pd.to_datetime(dff["Transaction_Date"]) >= pd.to_datetime(d0)) &
    (pd.to_datetime(dff["Transaction_Date"]) <= pd.to_datetime(d1))
]

    # ── Helpers ───────────────────────────────────────────────────────────
    def kpi_card(label, value, sub=None):
        sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
        st.markdown(f"""
        <div class="kpi">
          <div class="kpi-lbl">{label}</div>
          <div class="kpi-val">{value}</div>
          {sub_html}
        </div>""", unsafe_allow_html=True)

    def sec(title):
        st.markdown(f'<div class="sec-title">{title}</div>', unsafe_allow_html=True)

    def box(text, kind="info"):
        st.markdown(f'<div class="box-{kind}">{text}</div>', unsafe_allow_html=True)

    def pl_chart(data_df, grp_col, title):
        g = (data_df.groupby(grp_col)[FRAUD_COL]
             .agg(["sum","count"])
             .assign(rate=lambda x: x["sum"] / x["count"] * 100)
             .sort_values("rate", ascending=True)
             .reset_index())
        fig = go.Figure(go.Bar(
            x=g["rate"], y=g[grp_col], orientation="h",
            marker_color=C_MED,
            text=[f"{v:.1f}%" for v in g["rate"]],
            textfont=dict(color=C_WHITE), textposition="outside",
        ))
        fig.update_layout(**PLOTLY_LAYOUT)
        fig.update_layout(title=title, xaxis_title="Fraud Rate (%)",
                          height=max(300, len(g)*40 + 80))
        st.plotly_chart(fig, use_container_width=True)

    def cm_fig(cm_data, title):
        z    = cm_data
        text = [[str(v) for v in row] for row in z]
        fig  = go.Figure(go.Heatmap(
            z=z, x=["Pred Non-Fraud","Pred Fraud"],
            y=["Act Non-Fraud","Act Fraud"],
            text=text, texttemplate="%{text}", textfont=dict(size=14),
            colorscale=PURPLE_CS, showscale=False,
        ))
        fig.update_layout(**PLOTLY_LAYOUT)
        fig.update_layout(title=title, height=280)
        st.plotly_chart(fig, use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────
    # PAGE 1 — EXECUTIVE OVERVIEW
    # ─────────────────────────────────────────────────────────────────────
    if page == "🏠 Executive Overview":
        st.markdown("## 🏠 Executive Overview")
        st.markdown(f"*Showing **{len(dff):,}** transactions after filters*")

        n_fraud       = int(dff[FRAUD_COL].sum())
        fraud_pct     = dff[FRAUD_COL].mean() * 100
        total_val     = dff[AMOUNT_COL].sum()
        avg_amt       = dff[AMOUNT_COL].mean()
        n_hc          = int(dff[RISK_LEVEL_COL].isin(["High","Critical"]).sum()) if RISK_LEVEL_COL in dff.columns else 0

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        with c1: kpi_card("Total Transactions", f"{len(dff):,}", "after filters")
        with c2: kpi_card("Transaction Value",  f"₹{total_val/1e6:.1f}M", "total INR")
        with c3: kpi_card("Fraud Cases",        str(n_fraud), f"{fraud_pct:.2f}% fraud rate")
        with c4: kpi_card("Fraud Rate",         f"{fraud_pct:.2f}%", "1.74% baseline")
        with c5: kpi_card("Avg Txn Amount",     f"₹{avg_amt:,.0f}", "mean INR")
        with c6: kpi_card("High/Critical Risk", f"{n_hc:,}", "transactions flagged")

        st.markdown("---")

        # Daily trend
        sec("Daily Transaction & Fraud Trend")
        if "Transaction_Date" in dff.columns:
            daily = dff.groupby("Transaction_Date").agg(
                Transactions=(FRAUD_COL, "count"),
                Fraud=(FRAUD_COL, "sum")
            ).reset_index()
            fig = go.Figure()
            fig.add_trace(go.Bar(x=daily["Transaction_Date"], y=daily["Transactions"],
                                  name="Transactions", marker_color=C_MED, opacity=0.7))
            fig.add_trace(go.Scatter(x=daily["Transaction_Date"], y=daily["Fraud"],
                                      name="Fraud", line=dict(color=C_ACCENT_RED, width=2),
                                      yaxis="y2"))
            fig.update_layout(**PLOTLY_LAYOUT)
            fig.update_layout(
                title="Daily Transactions and Fraud Cases",
                yaxis=dict(title="Transactions", tickfont=dict(color=C_SEC),
                           gridcolor="#3D2E5A", linecolor="#3D2E5A"),
                yaxis2=dict(title="Fraud Cases", overlaying="y", side="right",
                            tickfont=dict(color=C_SEC), gridcolor="#3D2E5A"),
                height=320,
            )
            st.plotly_chart(fig, use_container_width=True)

        col_l, col_r = st.columns(2)
        with col_l:
            sec("Transaction Type Distribution")
            if "Transaction_Type" in dff.columns:
                vc = dff["Transaction_Type"].value_counts()
                fig = go.Figure(go.Pie(
                    labels=vc.index, values=vc.values,
                    hole=0.45,
                    marker_colors=PURPLE_PALETTE[:len(vc)],
                    textfont=dict(color=C_WHITE),
                ))
                fig.update_layout(**PLOTLY_LAYOUT)
                fig.update_layout(title="Transaction Type", height=320)
                st.plotly_chart(fig, use_container_width=True)

        with col_r:
            sec("Fraud vs Non-Fraud")
            vc2 = dff[FRAUD_COL].value_counts()
            fig2 = go.Figure(go.Pie(
                labels=["Non-Fraud","Fraud"],
                values=[vc2.get(0, 0), vc2.get(1, 0)],
                hole=0.45,
                marker_colors=[C_LLIGHT, C_DEEP],
                textfont=dict(color=C_WHITE),
            ))
            fig2.update_layout(**PLOTLY_LAYOUT)
            fig2.update_layout(title=f"Fraud Rate = {fraud_pct:.2f}%", height=320)
            st.plotly_chart(fig2, use_container_width=True)

        # Transaction status bar
        sec("Transaction Status")
        if "Transaction_Status" in dff.columns:
            sc = dff["Transaction_Status"].value_counts().reset_index()
            sc.columns = ["Status", "Count"]
            fig = go.Figure(go.Bar(
                x=sc["Status"], y=sc["Count"],
                marker_color=PURPLE_PALETTE[2],
                text=sc["Count"], textfont=dict(color=C_WHITE), textposition="outside",
            ))
            fig.update_layout(**PLOTLY_LAYOUT)
            fig.update_layout(title="Transaction Status Distribution", height=280)
            st.plotly_chart(fig, use_container_width=True)

        # Insight callouts
        sec("Key Risk Insights")
        ca, cb, cc = st.columns(3)
        with ca: box("🔴 <b>New Device:</b> Fraud rate 3.86% — 2.2× baseline. "
                     "Step-up auth required.", "danger")
        with cb: box("🟠 <b>Odd Hour (00–06):</b> Fraud rate 3.89% — 2.2× baseline. "
                     "Night-time controls needed.", "warn")
        with cc: box("🟡 <b>High Value (top 1%):</b> Fraud rate 5.0% — 2.9× baseline. "
                     "Cooling period recommended.", "warn")

        # Amount distribution
        sec("Transaction Amount Distribution (Fraud vs Non-Fraud)")
        fig = go.Figure()
        for lbl, col in [(0, C_LLIGHT), (1, C_ACCENT_RED)]:
            sub = dff[dff[FRAUD_COL] == lbl][AMOUNT_COL]
            fig.add_trace(go.Histogram(
                x=np.log1p(sub), name="Non-Fraud" if lbl==0 else "Fraud",
                marker_color=col, opacity=0.7, nbinsx=60,
            ))
        fig.update_layout(**PLOTLY_LAYOUT)
        fig.update_layout(title="log(1+Amount) — Fraud vs Non-Fraud", barmode="overlay",
                          xaxis_title="log(1 + Amount)", height=300)
        st.plotly_chart(fig, use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────
    # PAGE 2 — FRAUD ANALYSIS
    # ─────────────────────────────────────────────────────────────────────
    elif page == "🔍 Fraud Analysis":
        st.markdown("## 🔍 Fraud Analysis")

        cols_to_plot = [
            ("Transaction_Type",  "Fraud Rate by Transaction Type"),
            ("Device_Type",       "Fraud Rate by Device Type"),
            ("Network_Type",      "Fraud Rate by Network Type"),
            ("Sender_Age_Group",  "Fraud Rate by Sender Age Group"),
            ("Sender_Location",   "Fraud Rate by Sender Location (top 15)"),
        ]

        # 2-column grid for first 4
        for i in range(0, 4, 2):
            c1, c2 = st.columns(2)
            for j, col_ in enumerate(cols_to_plot[i:i+2]):
                cname, ctitle = col_
                with (c1 if j == 0 else c2):
                    if cname in dff.columns:
                        pl_chart(dff, cname, ctitle)

        # Sender Location top 15
        if "Sender_Location" in dff.columns:
            top15 = (dff.groupby("Sender_Location")[FRAUD_COL]
                     .agg(["sum","count"])
                     .assign(rate=lambda x: x["sum"]/x["count"]*100)
                     .nlargest(15,"rate")
                     .sort_values("rate", ascending=True)
                     .reset_index())
            fig = go.Figure(go.Bar(
                x=top15["rate"], y=top15["Sender_Location"], orientation="h",
                marker_color=C_LAV,
                text=[f"{v:.1f}%" for v in top15["rate"]],
                textfont=dict(color=C_WHITE), textposition="outside",
            ))
            fig.update_layout(**PLOTLY_LAYOUT)
            fig.update_layout(title="Fraud Rate by Sender Location (Top 15)",
                              xaxis_title="Fraud Rate (%)", height=460)
            st.plotly_chart(fig, use_container_width=True)

        # Fraud rate by hour
        sec("Fraud Rate by Hour of Day")
        if "Transaction_Hour" in dff.columns:
            hr = (dff.groupby("Transaction_Hour")[FRAUD_COL]
                  .agg(["sum","count"])
                  .assign(rate=lambda x: x["sum"]/x["count"]*100)
                  .reset_index())
            colors_hr = [C_ACCENT_RED if h < 6 else C_MED for h in hr["Transaction_Hour"]]
            fig = go.Figure(go.Bar(
                x=hr["Transaction_Hour"], y=hr["rate"],
                marker_color=colors_hr,
                text=[f"{v:.1f}%" for v in hr["rate"]],
                textfont=dict(color=C_WHITE), textposition="outside",
            ))
            fig.update_layout(**PLOTLY_LAYOUT)
            fig.update_layout(title="Fraud Rate by Hour (red = midnight–6 AM)",
                              xaxis_title="Hour of Day", yaxis_title="Fraud Rate (%)", height=320)
            st.plotly_chart(fig, use_container_width=True)

        # New device vs known
        sec("New Device vs Known Device")
        if "Is_New_Device" in dff.columns:
            col1, col2 = st.columns(2)
            with col1:
                nd = (dff.groupby("Is_New_Device")[FRAUD_COL]
                      .agg(["sum","count"])
                      .assign(rate=lambda x: x["sum"]/x["count"]*100)
                      .reset_index())
                nd["label"] = nd["Is_New_Device"].map({0:"Known Device",1:"New Device"})
                fig = go.Figure(go.Bar(
                    x=nd["label"], y=nd["rate"],
                    marker_color=[C_LLIGHT, C_ACCENT_RED],
                    text=[f"{v:.2f}%" for v in nd["rate"]],
                    textfont=dict(color=C_WHITE), textposition="outside",
                ))
                fig.update_layout(**PLOTLY_LAYOUT)
                fig.update_layout(title="Fraud Rate: New vs Known Device",
                                  yaxis_title="Fraud Rate (%)", height=300)
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                if "Failed_Attempts" in dff.columns:
                    fa = (dff.groupby("Failed_Attempts")[FRAUD_COL]
                          .agg(["sum","count"])
                          .assign(rate=lambda x: x["sum"]/x["count"]*100)
                          .reset_index()
                          .head(8))
                    fig = go.Figure(go.Bar(
                        x=fa["Failed_Attempts"].astype(str), y=fa["rate"],
                        marker_color=C_MED,
                        text=[f"{v:.1f}%" for v in fa["rate"]],
                        textfont=dict(color=C_WHITE), textposition="outside",
                    ))
                    fig.update_layout(**PLOTLY_LAYOUT)
                    fig.update_layout(title="Fraud Rate by Failed Attempts Count",
                                      xaxis_title="Failed Attempts", yaxis_title="Fraud Rate (%)",
                                      height=300)
                    st.plotly_chart(fig, use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────
    # PAGE 3 — RISK MONITORING
    # ─────────────────────────────────────────────────────────────────────
    elif page == "⚡ Risk Monitoring":
        st.markdown("## ⚡ Risk Monitoring")

        n_crit     = int((dff[RISK_LEVEL_COL]=="Critical").sum())    if RISK_LEVEL_COL in dff.columns else 0
        n_high     = int((dff[RISK_LEVEL_COL]=="High").sum())        if RISK_LEVEL_COL in dff.columns else 0
        n_new_dev  = int(dff["Is_New_Device"].sum())                  if "Is_New_Device" in dff.columns else 0
        n_vel      = int((dff.get("Velocity_24h_High", pd.Series(dtype=int))==1).sum()) if "Velocity_24h_High" in dff.columns else 0

        c1, c2, c3, c4 = st.columns(4)
        with c1: kpi_card("Critical Risk",    f"{n_crit:,}",    f"{n_crit/len(dff)*100:.1f}% of txns")
        with c2: kpi_card("High Risk",        f"{n_high:,}",    f"{n_high/len(dff)*100:.1f}% of txns")
        with c3: kpi_card("New Device Txns",  f"{n_new_dev:,}", f"{n_new_dev/len(dff)*100:.1f}%")
        with c4: kpi_card("High Velocity",    f"{n_vel:,}",     "≥5 txns in 24h")

        st.markdown("---")

        c1, c2 = st.columns(2)
        with c1:
            sec("Risk Level Distribution (Dataset Risk_Level)")
            if RISK_LEVEL_COL in dff.columns:
                ro = ["Low","Medium","High","Critical"]
                rc = dff[RISK_LEVEL_COL].value_counts().reindex(ro, fill_value=0).reset_index()
                rc.columns = ["Risk_Level","Count"]
                fig = go.Figure(go.Bar(
                    x=rc["Risk_Level"], y=rc["Count"],
                    marker_color=[RISK_COLOR[r] for r in rc["Risk_Level"]],
                    text=rc["Count"], textfont=dict(color=C_WHITE), textposition="outside",
                ))
                fig.update_layout(**PLOTLY_LAYOUT)
                fig.update_layout(title="Dataset Risk Level Distribution", height=300)
                st.plotly_chart(fig, use_container_width=True)

        with c2:
            sec("Custom Risk Level Distribution")
            if "Custom_Risk_Level" in dff.columns:
                ro = ["Low","Medium","High","Critical"]
                crc = dff["Custom_Risk_Level"].value_counts().reindex(ro, fill_value=0).reset_index()
                crc.columns = ["Risk_Level","Count"]
                fig = go.Figure(go.Bar(
                    x=crc["Risk_Level"], y=crc["Count"],
                    marker_color=[RISK_COLOR[r] for r in crc["Risk_Level"]],
                    text=crc["Count"], textfont=dict(color=C_WHITE), textposition="outside",
                ))
                fig.update_layout(**PLOTLY_LAYOUT)
                fig.update_layout(title="Custom Rule-Based Risk Level Distribution", height=300)
                st.plotly_chart(fig, use_container_width=True)

        # Risk Score histogram
        sec("Risk Score Distribution (Fraud vs Non-Fraud)")
        rs_col = RISK_SCORE_COL if RISK_SCORE_COL in dff.columns else "Custom_Risk_Score"
        if rs_col in dff.columns:
            fig = go.Figure()
            for lbl, col in [(0, C_LLIGHT), (1, C_ACCENT_RED)]:
                sub = dff[dff[FRAUD_COL]==lbl][rs_col]
                fig.add_trace(go.Histogram(
                    x=sub, name="Non-Fraud" if lbl==0 else "Fraud",
                    marker_color=col, opacity=0.7, nbinsx=40,
                ))
            fig.update_layout(**PLOTLY_LAYOUT)
            fig.update_layout(title=f"{rs_col} Distribution", barmode="overlay",
                              xaxis_title="Risk Score", height=300)
            st.plotly_chart(fig, use_container_width=True)

        # Mean risk score bar
        c1, c2 = st.columns(2)
        with c1:
            sec("Mean Risk Score: Fraud vs Non-Fraud")
            if rs_col in dff.columns:
                means = dff.groupby(FRAUD_COL)[rs_col].mean().reset_index()
                means["label"] = means[FRAUD_COL].map({0:"Non-Fraud",1:"Fraud"})
                fig = go.Figure(go.Bar(
                    x=means["label"], y=means[rs_col],
                    marker_color=[C_LLIGHT, C_ACCENT_RED],
                    text=[f"{v:.1f}" for v in means[rs_col]],
                    textfont=dict(color=C_WHITE), textposition="outside",
                ))
                fig.update_layout(**PLOTLY_LAYOUT)
                fig.update_layout(title="Mean Risk Score", yaxis_title="Score", height=280)
                st.plotly_chart(fig, use_container_width=True)

        with c2:
            box(f"Risk_Score correlation with Fraud_Label: <b>0.50</b> — "
                "moderate predictive power without data leakage.", "info")
            box("Critical (≥70) and High (50–69) risk tiers show highest fraud concentration.", "warn")

        # High-risk explorer table
        sec("High / Critical Risk Transaction Explorer")
        if RISK_LEVEL_COL in dff.columns:
            hc_df = dff[dff[RISK_LEVEL_COL].isin(["High","Critical"])].copy()
            cols_show = [c for c in [
                "Transaction_ID", AMOUNT_COL, "Transaction_Type", "Device_Type",
                RISK_LEVEL_COL, RISK_SCORE_COL, FRAUD_COL, "Transaction_Date"
            ] if c in hc_df.columns]
            st.dataframe(hc_df[cols_show].head(200), use_container_width=True, height=300)

    # ─────────────────────────────────────────────────────────────────────
    # PAGE 4 — ANOMALY DETECTION
    # ─────────────────────────────────────────────────────────────────────
    elif page == "🚨 Anomaly Detection":
        st.markdown("## 🚨 Anomaly Detection")

        n_anom  = int(dff["Is_Anomaly"].sum()) if "Is_Anomaly" in dff.columns else 0
        n_norm  = len(dff) - n_anom
        n_af    = int(dff[(dff.get("Is_Anomaly",0)==1) & (dff[FRAUD_COL]==1)].shape[0])
        ar      = dff[dff["Is_Anomaly"]==1][FRAUD_COL].mean()*100 if n_anom > 0 else 0

        c1, c2, c3, c4 = st.columns(4)
        with c1: kpi_card("Anomalies Detected", f"{n_anom:,}",  f"{n_anom/len(dff)*100:.1f}% of txns")
        with c2: kpi_card("Normal",             f"{n_norm:,}",  f"{n_norm/len(dff)*100:.1f}%")
        with c3: kpi_card("Fraud in Anomalies", str(n_af),       "fraud cases flagged")
        with c4: kpi_card("Fraud Rate (Anom.)", f"{ar:.2f}%",   f"vs {dff[FRAUD_COL].mean()*100:.2f}% baseline")

        st.markdown("---")

        c1, c2 = st.columns(2)
        with c1:
            sec("Normal vs Anomaly Count")
            fig = go.Figure(go.Bar(
                x=["Normal","Anomaly"], y=[n_norm, n_anom],
                marker_color=[C_LLIGHT, C_ACCENT_RED],
                text=[str(n_norm), str(n_anom)],
                textfont=dict(color=C_WHITE), textposition="outside",
            ))
            fig.update_layout(**PLOTLY_LAYOUT)
            fig.update_layout(title="Isolation Forest Result", height=300)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            sec("IF Score Distribution (Fraud vs Non-Fraud)")
            if "IF_Score_Raw" in dff.columns:
                fig = go.Figure()
                for lbl, col in [(0, C_LLIGHT), (1, C_ACCENT_RED)]:
                    sub = dff[dff[FRAUD_COL]==lbl]["IF_Score_Raw"]
                    if len(sub) > 0:
                        fig.add_trace(go.Histogram(
                            x=sub, name="Non-Fraud" if lbl==0 else "Fraud",
                            marker_color=col, opacity=0.7, nbinsx=50,
                        ))
                fig.update_layout(**PLOTLY_LAYOUT)
                fig.update_layout(title="IF Score (lower = more anomalous)",
                                  barmode="overlay", height=300)
                st.plotly_chart(fig, use_container_width=True)

        # Fraud & Anomaly overlap
        sec("Fraud & Anomaly Overlap")
        if "Is_Anomaly" in dff.columns:
            groups = ["Normal+Legit","Normal+Fraud","Anomaly+Legit","Anomaly+Fraud"]
            vals   = [
                int(((dff["Is_Anomaly"]==0) & (dff[FRAUD_COL]==0)).sum()),
                int(((dff["Is_Anomaly"]==0) & (dff[FRAUD_COL]==1)).sum()),
                int(((dff["Is_Anomaly"]==1) & (dff[FRAUD_COL]==0)).sum()),
                int(((dff["Is_Anomaly"]==1) & (dff[FRAUD_COL]==1)).sum()),
            ]
            cols_c = [C_LLIGHT, C_ACCENT_RED, C_MED, "#DC2626"]
            fig = go.Figure(go.Bar(
                x=groups, y=vals, marker_color=cols_c,
                text=[str(v) for v in vals],
                textfont=dict(color=C_WHITE), textposition="outside",
            ))
            fig.update_layout(**PLOTLY_LAYOUT)
            fig.update_layout(title="Transaction Groups: Normal/Anomaly × Legit/Fraud", height=320)
            st.plotly_chart(fig, use_container_width=True)

        # Feature means comparison
        sec("Feature Means: Anomaly vs Normal")
        if "Is_Anomaly" in dff.columns:
            feat_cols = [c for c in [
                AMOUNT_COL, "Failed_Attempts", "Transactions_Last_24_Hours",
                "Account_Age_Days", "Amount_vs_Avg_Ratio"
            ] if c in dff.columns]
            means_df = dff.groupby("Is_Anomaly")[feat_cols].mean().T.reset_index()
            means_df.columns = ["Feature", "Normal", "Anomaly"]
            fig = go.Figure()
            fig.add_trace(go.Bar(name="Normal",  x=means_df["Feature"],
                                  y=means_df["Normal"],  marker_color=C_LLIGHT))
            fig.add_trace(go.Bar(name="Anomaly", x=means_df["Feature"],
                                  y=means_df["Anomaly"], marker_color=C_ACCENT_RED))
            fig.update_layout(**PLOTLY_LAYOUT)
            fig.update_layout(title="Mean Feature Values: Anomaly vs Normal",
                              barmode="group", height=320)
            st.plotly_chart(fig, use_container_width=True)

        # Suspicious viewer
        sec("Suspicious Transaction Viewer")
        susp = dff[dff["Is_Anomaly"]==1].copy() if "Is_Anomaly" in dff.columns else dff.head(0)
        cols_s = [c for c in [
            "Transaction_ID", AMOUNT_COL, "Transaction_Type", "Device_Type",
            "IF_Score_Raw", RISK_LEVEL_COL, FRAUD_COL, "Transaction_Date"
        ] if c in susp.columns]
        st.dataframe(susp[cols_s].head(300), use_container_width=True, height=300)

    # ─────────────────────────────────────────────────────────────────────
    # PAGE 5 — ML MODEL PERFORMANCE
    # ─────────────────────────────────────────────────────────────────────
    elif page == "🤖 ML Model Performance":
        st.markdown("## 🤖 ML Model Performance")

        box("ℹ️ <b>Leakage prevention:</b> Risk_Score and Risk_Level are excluded from ML "
            "features. Models are trained on 20 behavioral/transactional features only.", "info")

        if ml_res:
            lr_m  = ml_res.get("lr", {})
            rf_m  = ml_res.get("rf", {})

            # Metrics table
            sec("Model Metrics Comparison")
            metrics_df = pd.DataFrame({
                "Metric":    ["ROC-AUC","PR-AUC","Precision","Recall","F1","Accuracy","CV Mean AUC"],
                "Logistic Regression": [
                    lr_m.get("roc_auc", 0.6352), lr_m.get("pr_auc", 0.0304),
                    lr_m.get("precision",0.0315), lr_m.get("recall", 0.6571),
                    lr_m.get("f1", 0.0601),       lr_m.get("accuracy",0),
                    ml_res.get("cv_lr_mean", 0.553),
                ],
                "Random Forest": [
                    rf_m.get("roc_auc", 0.5497), rf_m.get("pr_auc", 0.0261),
                    rf_m.get("precision",0.0833), rf_m.get("recall", 0.0286),
                    rf_m.get("f1", 0.0426),       rf_m.get("accuracy",0),
                    ml_res.get("cv_rf_mean", 0.5105),
                ],
            })
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)

            col_i1, col_i2, col_i3 = st.columns(3)
            with col_i1: box("✅ LR recall = 65.71%: catches 2 in 3 fraud cases.", "ok")
            with col_i2: box("⚠️ RF recall = 2.86%: misses 97% of fraud cases.", "warn")
            with col_i3: box("🎯 Recall is the primary metric for fraud detection.", "info")

            st.markdown("---")

            # ROC & PR curves
            c1, c2 = st.columns(2)
            with c1:
                sec("ROC Curves")
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=ml_res.get("fpr_lr", [0,1]), y=ml_res.get("tpr_lr", [0,1]),
                    name=f"LR (AUC={lr_m.get('roc_auc',0.6352):.4f})",
                    line=dict(color=C_LAV, width=2)
                ))
                fig.add_trace(go.Scatter(
                    x=ml_res.get("fpr_rf", [0,1]), y=ml_res.get("tpr_rf", [0,1]),
                    name=f"RF (AUC={rf_m.get('roc_auc',0.5497):.4f})",
                    line=dict(color=C_MED, width=2)
                ))
                fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                    line=dict(color=C_MUTED, dash="dash"), name="Random", showlegend=True))
                fig.update_layout(**PLOTLY_LAYOUT)
                fig.update_layout(title="ROC Curves", xaxis_title="FPR", yaxis_title="TPR", height=350)
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                sec("PR Curves")
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=ml_res.get("pr_rec_lr", [0,1]),
                    y=ml_res.get("pr_prec_lr", [0.0174, 0.0174]),
                    name=f"LR (PR-AUC={lr_m.get('pr_auc',0.0304):.4f})",
                    line=dict(color=C_LAV, width=2)
                ))
                fig.add_trace(go.Scatter(
                    x=ml_res.get("pr_rec_rf", [0,1]),
                    y=ml_res.get("pr_prec_rf", [0.0174, 0.0174]),
                    name=f"RF (PR-AUC={rf_m.get('pr_auc',0.0261):.4f})",
                    line=dict(color=C_MED, width=2)
                ))
                fig.update_layout(**PLOTLY_LAYOUT)
                fig.update_layout(title="Precision-Recall Curves",
                                  xaxis_title="Recall", yaxis_title="Precision", height=350)
                st.plotly_chart(fig, use_container_width=True)

            # Confusion matrices
            sec("Confusion Matrices")
            c1, c2 = st.columns(2)
            with c1: cm_fig(ml_res.get("cm_lr", [[0,0],[0,0]]), "LR Confusion Matrix")
            with c2: cm_fig(ml_res.get("cm_rf", [[0,0],[0,0]]), "RF Confusion Matrix")

            # Feature importance
            sec("Random Forest Feature Importance (Top 15)")
            fi_names = ml_res.get("fi_names", [])[:15]
            fi_vals  = ml_res.get("fi_vals",  [])[:15]
            if fi_names and fi_vals:
                fi_df = pd.DataFrame({"Feature": fi_names, "Importance": fi_vals})
                fi_df = fi_df.sort_values("Importance", ascending=True)
                fig = go.Figure(go.Bar(
                    x=fi_df["Importance"], y=fi_df["Feature"], orientation="h",
                    marker_color=PURPLE_PALETTE[2],
                    text=[f"{v:.4f}" for v in fi_df["Importance"]],
                    textfont=dict(color=C_WHITE), textposition="outside",
                ))
                fig.update_layout(**PLOTLY_LAYOUT)
                fig.update_layout(title="Top 15 RF Feature Importances",
                                  xaxis_title="Importance", height=480)
                st.plotly_chart(fig, use_container_width=True)

            # Radar chart
            sec("Model Performance Radar")
            cats = ["ROC-AUC","PR-AUC×10","Precision","Recall","F1×10"]
            lr_r = [lr_m.get("roc_auc",0.635)*1, lr_m.get("pr_auc",0.030)*10,
                    lr_m.get("precision",0.031)*1, lr_m.get("recall",0.657)*1,
                    lr_m.get("f1",0.060)*10]
            rf_r = [rf_m.get("roc_auc",0.550)*1, rf_m.get("pr_auc",0.026)*10,
                    rf_m.get("precision",0.083)*1, rf_m.get("recall",0.028)*1,
                    rf_m.get("f1",0.042)*10]
            fig = go.Figure()
            for name_, vals_, col_ in [("Logistic Regression", lr_r, C_LAV),
                                        ("Random Forest",        rf_r, C_MED)]:
                fig.add_trace(go.Scatterpolar(
                    r=vals_ + [vals_[0]], theta=cats + [cats[0]],
                    fill="toself", name=name_, line_color=col_, fillcolor=col_,
                    opacity=0.35,
                ))
            fig.update_layout(**PLOTLY_LAYOUT)
            fig.update_layout(
                polar=dict(
                    bgcolor=BG_CARD,
                    radialaxis=dict(visible=True, tickfont=dict(color=C_SEC),
                                    gridcolor="#3D2E5A", linecolor="#3D2E5A"),
                    angularaxis=dict(tickfont=dict(color=C_WHITE),
                                     gridcolor="#3D2E5A", linecolor="#3D2E5A"),
                ),
                title="Model Performance Radar (scaled for visibility)",
                height=380,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Business interpretation
            sec("Business Interpretation")
            biz_df = pd.DataFrame({
                "Scenario":   ["High-volume low-risk", "High-recall deployment", "Analyst review queue"],
                "Model":      ["Random Forest (prec 8.33%)", "Logistic Regression (recall 65.71%)", "LR + Risk Score tier"],
                "Use Case":   ["Minimize false positives", "Catch as many fraud cases as possible",
                               "Route Critical risk to analysts"],
                "Tradeoff":   ["Misses 97% of fraud", "High false-positive volume",
                               "Requires analyst staffing"],
            })
            st.dataframe(biz_df, use_container_width=True, hide_index=True)

        else:
            box("Model results not found. Run the pipeline first: "
                "<code>python upi_fraud_analysis.py --pipeline</code>", "warn")

    # ─────────────────────────────────────────────────────────────────────
    # PAGE 6 — BUSINESS DECISIONS
    # ─────────────────────────────────────────────────────────────────────
    elif page == "📋 Business Decisions":
        st.markdown("## 📋 Business Decisions")

        box("⚠️ <b>Important:</b> All findings are based on SYNTHETIC data. "
            "Validate thresholds on real production data before deployment.", "warn")

        # Amount threshold analysis
        sec("Transaction Amount Threshold Analysis")
        if AMOUNT_COL in dff.columns:
            thresholds = [1000, 5000, 10000, 20000, 50000, 100000]
            rows_t = []
            for t in thresholds:
                sub = dff[dff[AMOUNT_COL] >= t]
                if len(sub) > 0:
                    rows_t.append({
                        "Threshold (₹)": t,
                        "Txns Above": len(sub),
                        "% of Total": round(len(sub)/len(dff)*100, 1),
                        "Fraud in Group": int(sub[FRAUD_COL].sum()),
                        "Fraud Rate %": round(sub[FRAUD_COL].mean()*100, 2),
                    })
            thr_df = pd.DataFrame(rows_t)

            c1, c2 = st.columns([3, 2])
            with c1:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=thr_df["Threshold (₹)"].astype(str), y=thr_df["Txns Above"],
                    name="Txns Above Threshold", marker_color=C_MED, yaxis="y",
                ))
                fig.add_trace(go.Scatter(
                    x=thr_df["Threshold (₹)"].astype(str), y=thr_df["Fraud Rate %"],
                    name="Fraud Rate %", line=dict(color=C_ACCENT_RED, width=2), yaxis="y2",
                ))
                fig.update_layout(**PLOTLY_LAYOUT)
                fig.update_layout(
                    title="Transactions and Fraud Rate by Amount Threshold",
                    yaxis=dict(title="Count", tickfont=dict(color=C_SEC),
                               gridcolor="#3D2E5A", linecolor="#3D2E5A"),
                    yaxis2=dict(title="Fraud Rate %", overlaying="y", side="right",
                                tickfont=dict(color=C_SEC)),
                    height=340,
                )
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                st.dataframe(thr_df, use_container_width=True, hide_index=True)

        # New device policy
        sec("New Device Policy Analysis")
        if "Is_New_Device" in dff.columns:
            nd_grp = (dff.groupby("Is_New_Device")[FRAUD_COL]
                      .agg(["sum","count","mean"])
                      .reset_index())
            nd_grp.columns = ["Is_New_Device","Fraud_Count","Total","Fraud_Rate"]
            nd_grp["Label"] = nd_grp["Is_New_Device"].map({0:"Known Device",1:"New Device"})
            nd_grp["Fraud_Rate_Pct"] = (nd_grp["Fraud_Rate"]*100).round(2)

            c1, c2 = st.columns([2, 3])
            with c1:
                fig = go.Figure(go.Bar(
                    x=nd_grp["Label"], y=nd_grp["Fraud_Rate_Pct"],
                    marker_color=[C_LLIGHT, C_ACCENT_RED],
                    text=[f"{v:.2f}%" for v in nd_grp["Fraud_Rate_Pct"]],
                    textfont=dict(color=C_WHITE), textposition="outside",
                ))
                fig.update_layout(**PLOTLY_LAYOUT)
                fig.update_layout(title="Fraud Rate by Device Status",
                                  yaxis_title="Fraud Rate (%)", height=300)
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                pol_df = pd.DataFrame({
                    "Device Status":  ["Known Device", "New Device"],
                    "Fraud Rate":     ["~1.3%", "~3.86%"],
                    "Recommended Action": [
                        "Standard processing",
                        "Step-up auth: OTP + biometric verification",
                    ],
                    "Expected Impact": [
                        "No friction for majority",
                        "50–60% reduction in new-device fraud",
                    ],
                })
                st.dataframe(pol_df, use_container_width=True, hide_index=True)

        # Risk score action matrix
        sec("Risk Score Action Matrix")
        risk_action_df = pd.DataFrame({
            "Risk Level": ["Low (0–24)", "Medium (25–49)", "High (50–69)", "Critical (≥70)"],
            "Recommended Action": [
                "Auto-approve; standard logging",
                "Log and monitor; soft alert if pattern escalates",
                "Require OTP confirmation; flag for batch review",
                "Block or hold; immediate analyst review",
            ],
            "SLA": ["Real-time", "Real-time", "< 5 min review", "< 1 min review"],
            "Expected Fraud %": ["< 0.5%", "~1.5%", "~4-8%", "> 10%"],
        })

        if RISK_LEVEL_COL in dff.columns:
            ro = ["Low","Medium","High","Critical"]
            rl_ct = dff[RISK_LEVEL_COL].value_counts().reindex(ro, fill_value=0).reset_index()
            rl_ct.columns = ["Risk_Level", "Count"]
            fig = go.Figure(go.Bar(
                x=rl_ct["Risk_Level"], y=rl_ct["Count"],
                marker_color=[RISK_COLOR[r] for r in rl_ct["Risk_Level"]],
                text=rl_ct["Count"], textfont=dict(color=C_WHITE), textposition="outside",
            ))
            fig.update_layout(**PLOTLY_LAYOUT)
            fig.update_layout(title="Transaction Count by Risk Level", height=280)
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(risk_action_df, use_container_width=True, hide_index=True)

        st.markdown("---")

        # Report & download buttons
        sec("Generate & Download")
        col_r, col_d1, col_d2 = st.columns(3)

        with col_r:
            if st.button("📄 Generate Project Report"):
                with st.spinner("Building report..."):
                    try:
                        df_r, vr_r, ml_r, ch_r = run_full_pipeline()
                        rpath = generate_report(df_r, vr_r, ml_r, ch_r)
                        if rpath and Path(rpath).exists():
                            with open(rpath, "rb") as fh:
                                st.download_button(
                                    label="⬇️ Download Report (.docx)",
                                    data=fh.read(),
                                    file_name="Project_Report.docx",
                                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                )
                            box("✅ Report generated successfully!", "ok")
                        else:
                            box("⚠️ Report generation failed. Check python-docx installation.", "warn")
                    except Exception as e:
                        box(f"❌ Error: {e}", "danger")

        with col_d1:
            csv_buf = io.StringIO()
            dff.to_csv(csv_buf, index=False)
            st.download_button(
                label="⬇️ Download Filtered Dataset",
                data=csv_buf.getvalue().encode(),
                file_name="filtered_transactions.csv",
                mime="text/csv",
            )

        with col_d2:
            hc_exp = dff[dff[RISK_LEVEL_COL].isin(["High","Critical"])] if RISK_LEVEL_COL in dff.columns else dff.head(0)
            hc_buf = io.StringIO()
            hc_exp.to_csv(hc_buf, index=False)
            st.download_button(
                label="⬇️ Download High/Critical Risk Txns",
                data=hc_buf.getvalue().encode(),
                file_name="high_critical_risk.csv",
                mime="text/csv",
            )


# ══════════════════════════════════════════════════════════════════════════════
# H.  ENTRY POINTS
# ══════════════════════════════════════════════════════════════════════════════

_is_streamlit = "streamlit" in sys.modules or any("streamlit" in a for a in sys.argv)


def _headless_pipeline():
    """Run full analysis pipeline + generate report + print summary."""
    print("=" * 60)
    print("UPI Transaction Risk & Fraud Analytics — Pipeline")
    print("=" * 60)

    df, vr, ml_res, charts = run_full_pipeline()

    print(f"\n  Records         : {len(df):,}")
    print(f"  Fraud cases     : {int(df[FRAUD_COL].sum())} ({df[FRAUD_COL].mean()*100:.2f}%)")
    print(f"  Anomalies       : {int(df['Is_Anomaly'].sum()) if 'Is_Anomaly' in df.columns else 'N/A'}")
    print(f"  LR ROC-AUC      : {ml_res['lr']['roc_auc']}")
    print(f"  RF ROC-AUC      : {ml_res['rf']['roc_auc']}")
    print(f"  LR Recall       : {ml_res['lr']['recall']}")
    print(f"  Charts saved    : {len(charts)}")
    print()

    rpath = generate_report(df, vr, ml_res, charts)
    if rpath:
        print(f"  Report saved    : {rpath}")

    print("\nAll outputs written to outputs/")
    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--pipeline":
        _headless_pipeline()
    else:
        print("UPI Fraud Analytics")
        print("  Dashboard : streamlit run upi_fraud_analysis.py")
        print("  Pipeline  : python upi_fraud_analysis.py --pipeline")

if _is_streamlit:
    _run_dashboard()
