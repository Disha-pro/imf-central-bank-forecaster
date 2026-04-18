import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IMF Central Bank Forecaster",
    page_icon="🏦",
    layout="wide",
)

# ── Header ───────────────────────────────────────────────────────────────────
st.title("🏦 IMF Central Bank Total Assets Forecaster")
st.markdown(
    "**ML-powered time-series forecasting on real IMF Monetary & Financial Statistics data** "
    "| Built by [Redefine Apex](https://youtube.com/@RedefineApex)"
)
st.markdown("---")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    uploaded_file = st.file_uploader("Upload IMF CSV File", type=["csv"])
    st.markdown("---")
    st.markdown("**About this app**")
    st.markdown(
        "Upload the IMF MFS Central Bank Survey CSV, "
        "pick a country, and see 3 ML models compete to forecast "
        "central bank total assets."
    )
    st.markdown("---")
    st.markdown("🎥 [Watch on YouTube](https://youtube.com/@RedefineApex)")

# ── Helper functions ──────────────────────────────────────────────────────────

def parse_date(date_str):
    date_str = str(date_str).strip()
    try:
        if '-Q' in date_str:
            year, q = date_str.split('-Q')
            month = (int(q) - 1) * 3 + 1
            return pd.Timestamp(year=int(year), month=month, day=1)
        elif '-M' in date_str:
            year, m = date_str.split('-M')
            return pd.Timestamp(year=int(year), month=int(m), day=1)
        else:
            return pd.Timestamp(year=int(date_str), month=1, day=1)
    except Exception:
        return pd.NaT


def evaluate_model(name, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    return {"Model": name, "MAE": f"{mae:,.0f}", "RMSE": f"{rmse:,.0f}", "R²": f"{r2:.4f}"}


@st.cache_data(show_spinner=False)
def load_and_clean(file_bytes):
    df = pd.read_csv(file_bytes, low_memory=False)
    empty_cols = df.columns[df.isnull().all()].tolist()
    df = df.drop(columns=empty_cols)
    id_vars = ['COUNTRY', 'INDICATOR', 'FREQUENCY', 'SCALE',
               'OBS_MEASURE', 'TYPE_OF_TRANSFORMATION',
               'DEPARTMENT', 'METHODOLOGY_NOTES']
    id_vars = [c for c in id_vars if c in df.columns]
    time_cols = [c for c in df.columns if c not in id_vars]
    indicator = "Assets, Total assets, All sectors (CB1SR)"
    df_assets = df[df['INDICATOR'] == indicator].copy()
    df_assets = df_assets.loc[:, ~df_assets.columns.duplicated()]
    df_long = pd.melt(
        df_assets,
        id_vars=[c for c in id_vars if c in df_assets.columns],
        value_vars=[c for c in time_cols if c in df_assets.columns],
        var_name='DATE',
        value_name='TOTAL_ASSETS'
    )
    df_long['DATE'] = df_long['DATE'].astype(str)
    df_long = df_long.dropna(subset=['TOTAL_ASSETS'])
    df_long['DATE_PARSED'] = df_long['DATE'].apply(parse_date)
    df_long['TOTAL_ASSETS'] = pd.to_numeric(df_long['TOTAL_ASSETS'], errors='coerce')
    df_long = df_long.dropna(subset=['DATE_PARSED', 'TOTAL_ASSETS'])
    df_long = df_long.sort_values(['COUNTRY', 'DATE_PARSED']).reset_index(drop=True)
    return df_long


def build_features(df_country):
    df = df_country.drop_duplicates(subset=['DATE_PARSED'], keep='first').copy()
    df = df[['DATE_PARSED', 'TOTAL_ASSETS']].sort_values('DATE_PARSED').reset_index(drop=True)
    df['LAG_1']  = df['TOTAL_ASSETS'].shift(1)
    df['LAG_3']  = df['TOTAL_ASSETS'].shift(3)
    df['LAG_6']  = df['TOTAL_ASSETS'].shift(6)
    df['LAG_12'] = df['TOTAL_ASSETS'].shift(12)
    df['MA_3']   = df['TOTAL_ASSETS'].rolling(3).mean()
    df['MA_12']  = df['TOTAL_ASSETS'].rolling(12).mean()
    return df.dropna().reset_index(drop=True)


# ── Main app flow ─────────────────────────────────────────────────────────────

if uploaded_file is None:
    st.info("👈 Upload the IMF MFS Central Bank CSV from the sidebar to get started.")
    st.markdown("""
    **How to get the data:**
    1. Go to [IMF Data Portal](https://data.imf.org/)
    2. Search for **Monetary and Financial Statistics — Central Bank Survey**
    3. Download the full CSV
    4. Upload it here
    """)
    st.stop()

# Load data
with st.spinner("Loading and cleaning data..."):
    df_long = load_and_clean(uploaded_file)

st.success(f"✅ Data loaded — {df_long['COUNTRY'].nunique()} countries, "
           f"{len(df_long):,} data points")

# ── Country selector ──────────────────────────────────────────────────────────
country_counts = df_long.groupby('COUNTRY').size().sort_values(ascending=False)
countries = country_counts.index.tolist()

col1, col2 = st.columns([2, 1])
with col1:
    selected_country = st.selectbox(
        "Select Country",
        countries,
        index=countries.index('Chile') if 'Chile' in countries else 0,
        help="Countries sorted by number of data points (more = better for forecasting)"
    )
with col2:
    n_pts = country_counts[selected_country]
    st.metric("Data Points", n_pts)
    st.metric("Recommended?" , "✅ Yes" if n_pts >= 100 else "⚠️ Low data")

st.markdown("---")

# ── Build model data ──────────────────────────────────────────────────────────
df_country = df_long[df_long['COUNTRY'] == selected_country].copy()
df_model   = build_features(df_country)

if len(df_model) < 30:
    st.error("Not enough data to train models for this country. Try one with 100+ data points.")
    st.stop()

feature_cols = ['LAG_1', 'LAG_3', 'LAG_6', 'LAG_12', 'MA_3', 'MA_12']
X = df_model[feature_cols]
y = df_model['TOTAL_ASSETS']
split_idx = int(len(df_model) * 0.8)

X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]
dates_train = df_model['DATE_PARSED'][:split_idx]
dates_test  = df_model['DATE_PARSED'][split_idx:]

# ── Train models ──────────────────────────────────────────────────────────────
with st.spinner("Training models..."):
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_test)

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)

    gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    gb.fit(X_train, y_train)
    gb_preds = gb.predict(X_test)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Historical Data",
    "🤖 Model Comparison",
    "🔍 Feature Importance",
    "🔮 Future Forecast"
])

# ── Tab 1: Historical ─────────────────────────────────────────────────────────
with tab1:
    st.subheader(f"{selected_country} — Central Bank Total Assets")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(dates_train, y_train, color='royalblue', linewidth=2, label='Train Data')
    ax.plot(dates_test,  y_test,  color='crimson',   linewidth=2, label='Test Data',  linestyle='--')
    ax.axvline(x=dates_train.max(), color='black', linestyle=':', linewidth=1.5, label='Train/Test Split')
    ax.set_title(f"{selected_country} — Central Bank Total Assets (Train/Test Split)", fontweight='bold')
    ax.set_ylabel("Total Assets (Millions)")
    ax.legend(); ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    c1, c2, c3 = st.columns(3)
    c1.metric("Train Points", len(X_train))
    c2.metric("Test Points", len(X_test))
    c3.metric("Date Range",
              f"{df_model['DATE_PARSED'].min().year} – {df_model['DATE_PARSED'].max().year}")

# ── Tab 2: Model Comparison ────────────────────────────────────────────────────
with tab2:
    st.subheader("All 3 Models vs Actual")

    results = [
        evaluate_model("Linear Regression", y_test, lr_preds),
        evaluate_model("Random Forest",     y_test, rf_preds),
        evaluate_model("Gradient Boosting", y_test, gb_preds),
    ]
    st.dataframe(pd.DataFrame(results), use_container_width=True)

    st.markdown("---")

    models_info = [
        ("Linear Regression", lr_preds, "green"),
        ("Random Forest",     rf_preds, "orange"),
        ("Gradient Boosting", gb_preds, "purple"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    for ax, (name, preds, color) in zip(axes, models_info):
        r2 = r2_score(y_test, preds)
        ax.plot(dates_train, y_train, color='royalblue', linewidth=1.5,
                label='Train Actual', alpha=0.5)
        ax.plot(dates_test, y_test, color='black', linewidth=2,
                linestyle='--', label='Test Actual')
        ax.plot(dates_test, preds, color=color, linewidth=2,
                label=f'{name} (R²={r2:.4f})', marker='o', markersize=3)
        ax.axvline(x=dates_train.max(), color='red', linestyle=':', linewidth=1)
        ax.set_title(name, fontweight='bold')
        ax.set_ylabel("Total Assets (Millions)")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    st.pyplot(fig)

    with st.expander("🔍 Why did tree models fail?"):
        st.markdown("""
        **The Extrapolation Problem**

        Random Forest and Gradient Boosting are tree-based models that can only predict
        values within the range they saw during training. When Chile's central bank assets
        doubled post-COVID (2020), those values were completely outside the training range.

        Tree models responded by **plateauing** — predicting flat values near the training
        maximum. Linear Regression, which fits a global trend line, followed the upward
        trajectory correctly.

        **This is a real-world ML lesson:** model selection depends on data structure.
        For strongly trending economic series, simpler linear models often outperform
        complex tree-based models.
        """)

# ── Tab 3: Feature Importance ─────────────────────────────────────────────────
with tab3:
    st.subheader("What Drives the Predictions?")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    lr_coefs = pd.Series(np.abs(lr.coef_), index=feature_cols).sort_values()
    axes[0].barh(lr_coefs.index, lr_coefs.values, color='steelblue', edgecolor='black')
    axes[0].set_title("Linear Regression\n|Coefficients|", fontweight='bold')
    axes[0].set_xlabel("Absolute Coefficient")
    axes[0].grid(True, alpha=0.3, axis='x')

    rf_imp = pd.Series(rf.feature_importances_, index=feature_cols).sort_values()
    axes[1].barh(rf_imp.index, rf_imp.values, color='darkorange', edgecolor='black')
    axes[1].set_title("Random Forest\nFeature Importances", fontweight='bold')
    axes[1].set_xlabel("Importance Score")
    axes[1].grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("**Key Insight:** If `LAG_1` dominates, yesterday's value is the strongest predictor — "
                "classic autocorrelation in central bank data.")

# ── Tab 4: Future Forecast ─────────────────────────────────────────────────────
with tab4:
    st.subheader("12-Month Future Forecast")

    n_future = st.slider("Months to forecast", 6, 24, 12)

    df_future = df_model.copy()
    forecast_rows = []

    for _ in range(n_future):
        nf = {
            'LAG_1' : df_future['TOTAL_ASSETS'].iloc[-1],
            'LAG_3' : df_future['TOTAL_ASSETS'].iloc[-3],
            'LAG_6' : df_future['TOTAL_ASSETS'].iloc[-6],
            'LAG_12': df_future['TOTAL_ASSETS'].iloc[-12],
            'MA_3'  : df_future['TOTAL_ASSETS'].values[-3:].mean(),
            'MA_12' : df_future['TOTAL_ASSETS'].values[-12:].mean(),
        }
        y_next    = lr.predict(pd.DataFrame([nf]))[0]
        next_date = df_future['DATE_PARSED'].iloc[-1] + pd.DateOffset(months=1)
        new_row   = {'DATE_PARSED': next_date, 'TOTAL_ASSETS': y_next, **nf}
        df_future = pd.concat([df_future, pd.DataFrame([new_row])], ignore_index=True)
        forecast_rows.append({'Date': next_date, 'Forecasted Total Assets (M)': round(y_next, 2)})

    df_fc = pd.DataFrame(forecast_rows)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df_model['DATE_PARSED'], df_model['TOTAL_ASSETS'],
            color='royalblue', linewidth=1.5, label='Historical Actual')
    ax.plot(dates_test, lr_preds, color='green', linewidth=2,
            linestyle='--', label='LR Test Predictions')
    ax.plot(df_fc['Date'], df_fc['Forecasted Total Assets (M)'],
            color='red', linewidth=2.5, marker='o', markersize=5,
            label=f'{n_future}-Month Forecast')
    ax.axvspan(df_fc['Date'].min(), df_fc['Date'].max(), alpha=0.08, color='red')
    ax.axvline(x=df_model['DATE_PARSED'].iloc[-1], color='black',
               linestyle=':', linewidth=1.5, label='Forecast Start')
    ax.set_title(f"{selected_country} — Total Assets Forecast", fontweight='bold')
    ax.set_ylabel("Total Assets (Millions)")
    ax.legend(); ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    st.markdown("**Forecast Table:**")
    st.dataframe(df_fc, use_container_width=True)

    csv = df_fc.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Forecast CSV", csv,
                       file_name=f"{selected_country}_forecast.csv", mime='text/csv')

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "Built by **Disha** | "
    "[Redefine Apex YouTube](https://youtube.com/@RedefineApex) | "
    "Portfolio Project #3 — IMF Central Bank Forecaster"
)
