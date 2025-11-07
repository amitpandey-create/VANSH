"""
Bank Loan Data Insights - Streamlit app (NO matplotlib)
Dependencies: streamlit, pandas, numpy
Run: streamlit run app.py
"""

import io
import numpy as np
import pandas as pd
import streamlit as st

# -----------------------
# Sample Data
# -----------------------
SAMPLE_DATA = [
    ["C001","Male",35,"Married","Graduate",5400,1500,128,360,1,"Urban","Approved"],
    ["C002","Female",29,"Single","Graduate",3200,0,66,360,1,"Semiurban","Rejected"],
    ["C003","Male",42,"Married","Not Graduate",4300,1200,100,180,0,"Rural","Rejected"],
    ["C004","Male",28,"Single","Graduate",6000,0,141,360,1,"Urban","Approved"],
    ["C005","Female",33,"Married","Graduate",2500,2500,85,240,1,"Semiurban","Approved"],
    ["C006","Male",45,"Married","Not Graduate",4000,0,115,360,0,"Rural","Rejected"],
    ["C007","Female",31,"Single","Graduate",3500,1200,90,360,1,"Urban","Approved"],
    ["C008","Male",50,"Married","Graduate",8000,0,200,360,1,"Urban","Approved"],
    ["C009","Female",27,"Single","Graduate",2500,0,75,180,0,"Rural","Rejected"],
    ["C010","Male",39,"Married","Graduate",4200,1500,130,360,1,"Semiurban","Approved"]
]

DEFAULT_COLS = [
    "Customer_ID","Gender","Age","Marital_Status","Education","ApplicantIncome",
    "CoapplicantIncome","LoanAmount","Loan_Term_months","Credit_History","Property_Area","Loan_Status"
]

# -----------------------
# Load Data
# -----------------------

def load_sample_df():
    return pd.DataFrame(SAMPLE_DATA, columns=DEFAULT_COLS)


def read_uploaded_file(uploaded_file) -> pd.DataFrame:
    try:
        return pd.read_csv(uploaded_file)
    except Exception:
        uploaded_file.seek(0)
        try:
            return pd.read_excel(uploaded_file)
        except Exception as e:
            raise e

# -----------------------
# Data helpers
# -----------------------

def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["ApplicantIncome","CoapplicantIncome","LoanAmount","Age","Loan_Term_months","Credit_History"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["TotalIncome"] = df.get("ApplicantIncome", 0).fillna(0) + df.get("CoapplicantIncome", 0).fillna(0)
    if "Loan_Status" in df.columns:
        df["Loan_Approved"] = df["Loan_Status"].astype(str).str.strip().str.lower().map(lambda x: 1 if x == "approved" else 0)
    return df


def zscore_outlier_mask(series: pd.Series, threshold: float = 2.0) -> pd.Series:
    s = series.dropna()
    mean = s.mean()
    std = s.std(ddof=0)
    if std == 0 or pd.isna(std):
        return pd.Series([False]*len(series), index=series.index)
    z = (series - mean) / std
    return z.abs() > threshold

# -----------------------
# Streamlit UI
# -----------------------

st.set_page_config(page_title="Bank Loan Data Insights (No Matplotlib)", layout="wide")
st.title("🏦 Bank Loan Data Insights — Streamlit (no matplotlib)")

# Sidebar: data source
st.sidebar.header("Data")
source = st.sidebar.radio("Choose data source", ("Sample dataset", "Upload CSV/Excel"))

if source == "Sample dataset":
    df = load_sample_df()
else:
    uploaded = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv","xls","xlsx"])    
    if uploaded:
        try:
            df = read_uploaded_file(uploaded)
        except Exception as e:
            st.sidebar.error(f"Could not read file: {e}")
            df = load_sample_df()
    else:
        st.sidebar.info("No file uploaded — using sample dataset")
        df = load_sample_df()

# Preprocess
df = add_derived_columns(df)

# Display
st.subheader("Data preview")
st.dataframe(df)

# Filters
st.sidebar.header("Filters & Settings")
if "Loan_Status" in df.columns:
    status = st.sidebar.selectbox("Filter by Loan_Status", options=["All"] + sorted(df["Loan_Status"].dropna().unique().tolist()))
    if status != "All":
        df = df[df["Loan_Status"] == status]

# Download
buffer = io.StringIO()
df.to_csv(buffer, index=False)
st.download_button("Download filtered/processed CSV", data=buffer.getvalue(), file_name="loan_data_processed.csv", mime="text/csv")

# Numeric summary
st.subheader("Numeric summary")
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if num_cols:
    st.table(df[num_cols].describe().T.round(3))
else:
    st.info("No numeric columns detected.")

# Categorical summary
st.subheader("Categorical summary")
cat_cols = df.select_dtypes(include=[object, "category"]).columns.tolist()
for c in cat_cols:
    st.write(f"**{c}**")
    st.dataframe(df[c].fillna("Missing").value_counts().rename_axis(c).reset_index(name="Count"))

# Grouping examples
st.subheader("Grouping & Aggregation")
group_cols = st.multiselect("Group by (categorical)", options=cat_cols, default=cat_cols[:1])
agg_numeric = st.multiselect("Numeric columns to aggregate", options=num_cols, default=num_cols[:1])
agg_funcs = st.multiselect("Aggregation functions", options=["mean","sum","median","min","max","count"], default=["mean"])

if group_cols:
    st.write("Count per group")
    st.dataframe(df.groupby(group_cols).size().reset_index(name="Count"))
    if agg_numeric:
        agg_map = {c: agg_funcs for c in agg_numeric}
        try:
            grouped_agg = df.groupby(group_cols).agg(agg_map)
            grouped_agg.columns = ["_".join(col).strip() for col in grouped_agg.columns.values]
            st.write("Numeric aggregations by group")
            st.dataframe(grouped_agg.reset_index())
        except Exception as e:
            st.error(f"Aggregation error: {e}")

# Correlation & simple charts (use Streamlit built-ins)
st.subheader("Correlation & Charts")
if len(num_cols) >= 2:
    x = st.selectbox("X (numeric)", options=num_cols, index=0)
    y = st.selectbox("Y (numeric)", options=num_cols, index=1)
    corr = np.corrcoef(df[x].fillna(0), df[y].fillna(0))[0,1]
    st.write(f"Pearson correlation ({x},{y}): **{corr:.3f}**")
    # Scatter using built-in altair via st.scatter_chart not available; use line/area/bar as alternatives
    st.write("Sample line chart of chosen columns (for scatter-like view, consider adding Altair/Plotly later)")
    st.line_chart(df[[x,y]].dropna().head(200))
else:
    st.info("Need at least two numeric columns for correlation/scatter.")

# Outliers
st.subheader("Outliers (LoanAmount)")
if "LoanAmount" in df.columns:
    thresh = st.slider("Z-score threshold", 1.0, 4.0, 2.0, 0.5)
    mask = zscore_outlier_mask(df["LoanAmount"], threshold=thresh)
    outliers = df[mask]
    st.write(f"Outliers detected: {outliers.shape[0]}")
    if not outliers.empty:
        st.dataframe(outliers)

# Quick charts for categorical and numeric
st.subheader("Quick Visuals")
if cat_cols:
    cat_choice = st.selectbox("Category for bar chart", options=cat_cols)
    counts = df[cat_choice].fillna("Missing").value_counts()
    st.bar_chart(counts)

if num_cols:
    num_choice = st.selectbox("Numeric for histogram-like line", options=num_cols, index=0)
    st.area_chart(df[num_choice].dropna().reset_index(drop=True).head(200))

st.markdown("---")
st.info("This app uses Streamlit built-in charts (no matplotlib). To add interactive charts use Plotly or Altair in requirements.")

st.sidebar.write("App by ChatGPT — lightweight version without matplotlib")