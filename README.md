# UPI Transaction Risk & Fraud Analytics

> **Internship-Level Data Analytics Project** | Single-file Python application  
> ⚠️ Dataset is **fully synthetic** — no real personal, banking, or financial data is included.

---

## Project Overview

An end-to-end fraud analytics and machine learning project built on a synthetic UPI transaction dataset (~10,000 records). The project covers data validation, cleaning, feature engineering, exploratory data analysis, rule-based risk scoring, unsupervised anomaly detection, and supervised fraud classification — all accessible from a single Python file with an interactive Streamlit dashboard.

---

## Features

- **Interactive Dashboard** — 6-page Streamlit app with global filters, KPI cards, Plotly charts
- **Fraud Analysis** — fraud rates by transaction type, device, location, hour, velocity
- **Risk Monitoring** — pre-computed and custom risk scores, high-risk transaction explorer
- **Anomaly Detection** — Isolation Forest with overlap analysis
- **ML Model Comparison** — Logistic Regression vs Random Forest, ROC/PR curves, confusion matrices, feature importance
- **Business Decisions** — data-driven recommendations with live threshold charts
- **Report Generation** — `Nitu Singh_Project Report.docx` with embedded charts, via dashboard button or CLI

---

## Architecture — Single File

The entire project runs from **one Python file**:

```
Nitu Singh_UPI Transaction Risk & Fraud Analytics.py
```

It contains:
- Data loading, validation, cleaning
- Feature engineering
- Custom risk scoring (rule-based, no leakage)
- Anomaly detection (Isolation Forest)
- ML pipeline (Logistic Regression + Random Forest)
- Chart generation (matplotlib/seaborn)
- Report generation (python-docx)
- Streamlit dashboard (Plotly charts, 6 pages)

---

## Dataset

| Attribute | Value |
|---|---|
| File | `data/upi_transactions.csv` |
| Records | ~10,000 synthetic UPI transactions |
| Columns | 22 |
| Target variable | `Fraud_Label` (1 = Fraud, 0 = Non-Fraud) |
| Fraud rate | ~1.74% |
| Nature | **Fully synthetic — no real data** |

**Dataset link:** `DATASET_LINK_TO_BE_ADDED`

### Important Limitations

- Dataset is **synthetic** and does not represent real-world UPI fraud patterns.
- Only 174 fraud cases — severe class imbalance limits ML model performance.
- `Risk_Score` (corr = 0.50 with Fraud_Label) is **excluded from ML features** to prevent data leakage.
- No device fingerprinting or IP metadata available in this dataset.

---

## Technologies

| Technology | Purpose |
|---|---|
| Python 3.10+ | Core language |
| Pandas / NumPy | Data processing |
| Matplotlib / Seaborn | Static chart generation |
| Scikit-learn | ML models, anomaly detection, evaluation |
| Streamlit | Interactive dashboard |
| Plotly | Dashboard charts |
| python-docx | Project report (DOCX) |

---

## Installation

```bash
git clone <repo-url>
cd UPI-Transaction-Risk-Fraud-Analytics
pip install -r requirements.txt
```

---

## How to Run

### Launch the dashboard

```bash
python -m streamlit run "Nitu Singh_UPI Transaction Risk & Fraud Analytics.py"
```

Open **http://localhost:8501** in your browser.

### Run the analysis pipeline (headless)

```bash
python "Nitu Singh_UPI Transaction Risk & Fraud Analytics.py" --pipeline
```

This generates all charts, tables, results, and `Nitu Singh_Project Report.docx` without opening a browser.

### Generate the report from the dashboard

Navigate to **📋 Business Decisions → Generate Project Report** and click the button.

---

## Dashboard Pages

| Page | Content |
|---|---|
| 🏠 Executive Overview | KPI cards, daily trend, fraud split, amount distribution |
| 🔍 Fraud Analysis | Fraud rates by type/device/network/age/location/hour |
| ⚡ Risk Monitoring | Risk level distribution, score analysis, high-risk explorer |
| 🚨 Anomaly Detection | IF results, fraud/anomaly overlap, suspicious transaction viewer |
| 🤖 ML Model Performance | Metrics table, ROC/PR curves, confusion matrices, feature importance, radar chart |
| 📋 Business Decisions | Threshold charts, device policy, action matrix, report generation, data export |

---

## ML Methodology

- **Models**: Logistic Regression (baseline), Random Forest (ensemble)
- **Split**: 80/20 stratified train/test
- **Imbalance handling**: `class_weight='balanced'` on both models
- **Validation**: 5-fold stratified cross-validation (ROC-AUC)
- **Leakage protection**: `Risk_Score` and `Risk_Level` excluded from all model features
- **Key metrics**: Recall (primary), PR-AUC, ROC-AUC, Confusion Matrix

---

## Anomaly Detection

- **Method**: Isolation Forest (unsupervised)
- **Contamination**: 5% assumed anomaly rate
- **Output**: `Is_Anomaly` (binary), `IF_Score_Raw` (continuous)
- **Limitation**: Anomalies ≠ Fraud. Only ~9.8% of anomalies are confirmed fraud cases.

---

## Risk Scoring

A transparent rule-based score (0–100) computed from:
- Amount vs historical average (≤25 pts)
- New device usage (15 pts)
- Failed attempts (≤15 pts)
- Transaction velocity (≤15 pts)
- Odd-hour transaction (10 pts)
- Account age (≤10 pts)
- International flag (5 pts)
- Absolute high value (≤10 pts)

Categories: **Low** (0–24) | **Medium** (25–49) | **High** (50–69) | **Critical** (70–100)

`Fraud_Label` is **never used** in this calculation.

---

## Project Structure

```
UPI-Transaction-Risk-Fraud-Analytics/
│
├── Nitu Singh_UPI Transaction Risk & Fraud Analytics.py   ← Single Python file: analysis + dashboard + report
├── requirements.txt
├── README.md
├── Nitu Singh_Project Report.docx                         ← Generated by pipeline or dashboard button
│
├── data/
│   └── upi_transactions.csv  ← Place dataset here
│
├── outputs/
│   ├── charts/               ← Generated charts (10 PNGs)
│   ├── tables/               ← Cleaned CSV, Power BI CSV
│   └── results/              ← model_results.json, feature importances, suspicious transactions
│
└── dashboard/
    └── README.txt            ← Power BI dashboard specification
```

---

## Key Findings

1. Fraud rate: **1.74%** — severe class imbalance.
2. New-device transactions: **3.86% fraud rate** — 2.2× the average.
3. Midnight–6 AM transactions: **3.89% fraud rate**.
4. Top-1% high-value transactions: **5.0% fraud rate**.
5. `Risk_Score` effectively separates fraud (mean 72.7) from non-fraud (mean 22.5).
6. Logistic Regression achieves **65.71% recall** (ROC-AUC 0.6352); Random Forest achieves 8.33% precision but only 2.86% recall.
7. Isolation Forest anomalies have elevated fraud concentration vs baseline.

---

## Business Recommendations

> Based on synthetic data patterns only — validate before deployment.

1. Require re-authentication for new-device transactions above ₹2,000.
2. Step-up verification when ≥8 transactions in 24 hours.
3. Alert users for large odd-hour transactions.
4. Apply progressive limits for accounts under 90 days old.
5. Route Critical-risk (score ≥ 70) transactions to manual review.
6. Combine rule-based scoring + anomaly detection + ML for multi-layer defence.

---

## Author

**Nitu Singh**  
Data Analytics Intern  
Email:nitusinghz1106@gmail.com

---

*This project uses a fully synthetic dataset. No real personal, banking, or financial data is included.*
