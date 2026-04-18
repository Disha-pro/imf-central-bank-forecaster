# 🏦 IMF Central Bank Total Assets Forecaster
### ML-Powered Time-Series Forecasting on Real IMF Monetary & Financial Statistics Data

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4+-orange?logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red?logo=streamlit)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-green?logo=pandas)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

> **Portfolio Project #3** by Disha | 


## 📌 Project Overview

This project builds an end-to-end machine learning forecasting pipeline on real IMF Monetary and Financial Statistics (MFS) — Central Bank Survey data. It targets **Chile's Central Bank Total Assets** across 29 years (1997–2026) and forecasts values 12 months into the future.

The project demonstrates a complete ML workflow: raw data ingestion → cleaning → feature engineering → multi-model benchmarking → diagnosis of model failure → future forecasting.

---

## 🎯 Business Problem

Central banks publish balance sheet data that economists and financial analysts use to understand monetary policy. Can we use historical patterns in total assets to forecast near-future values? This project answers that question using real IMF data.

---

## 📊 Dataset

| Property | Detail |
|---|---|
| Source | IMF Monetary and Financial Statistics (MFS) — Central Bank Survey |
| File | `MF_STA_MFS_CBS_24_0_0_csv.csv` |
| Raw Shape | ~16,000 rows × 549 columns |
| Indicator Used | Assets, Total assets, All sectors (CB1SR) |
| Country Focus | Chile (476 monthly data points — most data-rich) |
| Date Range | October 1997 — January 2026 |

> ⚠️ Dataset is not committed due to size. Download from [IMF Data Portal](https://data.imf.org/).

---

## 🔁 Project Pipeline

```
Raw IMF CSV
    ↓
Data Cleaning (remove empty cols, filter indicator)
    ↓
Wide → Long Format (melt 473 time columns)
    ↓
Date Parsing (quarterly / monthly / annual)
    ↓
Country Selection (Chile — most data)
    ↓
Train/Test Split (80/20 — COVID period as test)
    ↓
Feature Engineering (LAG_1, LAG_3, LAG_6, LAG_12, MA_3, MA_12)
    ↓
Model Training (Linear Regression, Random Forest, Gradient Boosting)
    ↓
Evaluation + Diagnosis (extrapolation failure of tree models)
    ↓
Feature Importance Analysis
    ↓
12-Month Future Forecast (Feb 2026 → Jan 2027)
```

---

## 🤖 Models Trained

| Model | MAE | RMSE | R² |
|---|---|---|---|
| **Linear Regression** ✅ | 2,570,770 | 3,989,045 | **0.8798** |
| Random Forest | 12,483,215 | 14,520,664 | -0.5933 |
| Gradient Boosting | 16,220,186 | 17,923,964 | -1.4277 |

### 🔍 Key ML Insight

Tree-based models (Random Forest, Gradient Boosting) **cannot extrapolate beyond training value ranges**. Chile's central bank assets doubled post-COVID — a structural break that tree models had never seen. Linear Regression, following a global trend line, outperformed both. This is a real-world demonstration of the **extrapolation problem** in tree-based time-series forecasting.

> **Fix Applied:** First-order differencing was applied to tree models. RF improved from R² -0.59 → -0.20, confirming the diagnosis. Linear Regression remains the winner for this dataset.

---

## 📈 Features Engineered

| Feature | Description |
|---|---|
| LAG_1 | Total assets from 1 month ago |
| LAG_3 | Total assets from 3 months ago |
| LAG_6 | Total assets from 6 months ago |
| LAG_12 | Total assets from 12 months ago |
| MA_3 | 3-month rolling average |
| MA_12 | 12-month rolling average |

---

## 🖥️ Streamlit App

The project includes an interactive Streamlit app for exploring forecasts.

LINK :
https://imf-central-bank-forecaster-hqzvprtt4czeehdtyaugzl.streamlit.app/
```

**App Features:**
- Upload your own IMF CSV
- Select any country from the dataset
- View train/test split visualization
- Compare all 3 model predictions
- See 12-month future forecast
- Download forecast as CSV

---

## 🗂️ Repository Structure

```
imf-central-bank-forecaster/
│
├── imf_ml_project.py         # Full Colab pipeline (all 18 steps)
├── streamlit_app.py          # Interactive Streamlit app
├── requirements.txt          # Python dependencies
├── .gitignore                # Files excluded from Git
├── README.md                 # This file
│
├── outputs/
│   ├── chile_model_comparison.png      # Model predictions vs actual
│   ├── feature_importance.png          # Feature importance charts
│   └── chile_forecast_2026_2027.png   # Future forecast plot
│
└── notebooks/
    └── IMF_ML_Project.ipynb            # Google Colab notebook
```

---

## ⚙️ Setup & Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/imf-central-bank-forecaster.git
cd imf-central-bank-forecaster

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run Streamlit app
streamlit run streamlit_app.py
```

---

## 📹 YouTube Walkthrough

Full project walkthrough available on **Redefine Apex**:
👉 [Watch on YouTube](https://youtube.com/@RedefineApex)

Topics covered in the video:
- Why I chose IMF real-world data
- Why tree models failed (extrapolation problem)
- How differencing fixes tree models
- Feature importance interpretation
- Live demo of the Streamlit app

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **Pandas** — data manipulation
- **NumPy** — numerical operations
- **Scikit-Learn** — ML models & evaluation
- **Matplotlib / Seaborn** — static visualizations
- **Plotly** — interactive plots
- **Streamlit** — web app

---

## 👩‍💻 About

Built by **Disha** as part of an AI Product Management portfolio sprint.

- 💼 LinkedIn: https://www.linkedin.com/in/dishasonkar

---

## 📄 License

MIT License — free to use, learn from, and build upon with attribution.
