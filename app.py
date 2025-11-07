"""
Bank Loan Data Insights - Streamlit app
Dependencies: streamlit, pandas, numpy, matplotlib
Run: streamlit run app.py
"""

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

# -----------------------
# Streamlit Layout
# -----------------------
st.set_page_config(page_title="Bank Loan Data Insights", layout="wide")
st.title("🏦 Bank Loan Data Insights Dashboard")

st.sidebar.header("Data Source")
data_source = st.sidebar.radio("Choose data source", ("Sample dataset", "Upload CSV"))

if data_source == "Sample dataset":
    df = load_sample_df()
else:
    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
    else:
        st.warning("Please upload a CSV file.")
        st.stop()

# Derived Columns
df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
df["Loan_Approved"] = df["Loan_Status"].map({"Approved": 1, "Rejected": 0})

st.subheader("Dataset Preview")
st.dataframe(df)

# Summary Stats
st.subheader("Summary Statistics")
st.write(df.describe(include='all'))

# Grouping and Aggregation
st.subheader("Group by Categorical Columns")
cat_col = st.selectbox("Select a categorical column", ["Education", "Property_Area", "Gender", "Loan_Status"])
agg_col = st.selectbox("Select a numeric column to aggregate", ["ApplicantIncome", "LoanAmount", "TotalIncome"])
agg_func = st.selectbox("Select aggregation function", ["mean", "sum", "median", "count"])

if cat_col and agg_col:
    grouped = df.groupby(cat_col)[agg_col].agg(agg_func).reset_index()
    st.write(grouped)

# Plotting
st.subheader("Visualizations")
plot_type = st.selectbox("Select Plot Type", ["Bar Chart", "Scatter Plot", "Histogram"])

if plot_type == "Bar Chart":
    fig, ax = plt.subplots()
    grouped.plot(kind="bar", x=cat_col, y=agg_col, ax=ax)
    st.pyplot(fig)

elif plot_type == "Scatter Plot":
    x_col = st.selectbox("Select X-axis", ["ApplicantIncome", "LoanAmount", "TotalIncome"])
    y_col = st.selectbox("Select Y-axis", ["LoanAmount", "ApplicantIncome", "TotalIncome"])
    fig, ax = plt.subplots()
    ax.scatter(df[x_col], df[y_col])
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    st.pyplot(fig)

elif plot_type == "Histogram":
    num_col = st.selectbox("Select numeric column", ["ApplicantIncome", "LoanAmount", "TotalIncome"])
    fig, ax = plt.subplots()
    ax.hist(df[num_col], bins=10, edgecolor='black')
    st.pyplot(fig)

st.sidebar.write("App by ChatGPT - Bank Loan Analysis using Pandas, NumPy, and Matplotlib")