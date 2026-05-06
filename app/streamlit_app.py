import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib

from src.models.huggingface_model import generate_explanation

# -----------------------------------
# PAGE CONFIG
# -----------------------------------
st.set_page_config(
    page_title="Fraud Detection AI Terminal",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -----------------------------------
# CUSTOM STYLING (FUTURISTIC TRADING UI)
# -----------------------------------
st.markdown("""
<style>
body {
    background-color: #0e1117;
}
.block-container {
    padding-top: 1rem;
}
h1, h2, h3 {
    color: #00D9FF;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------
# HEADER
# -----------------------------------
st.markdown("# 🧠 Fraud Detection AI Terminal")
st.markdown("### Real-time AI Risk Analysis • Trading Style Dashboard")

# -----------------------------------
# LOAD MODEL
# -----------------------------------
@st.cache_resource
def load_model():
    return joblib.load("models/lightgbm_model.pkl")

model = load_model()

# -----------------------------------
# LOAD DATASET
# -----------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/raw/transactions.csv")

df = load_data()

# -----------------------------------
# TOP STATS BAR
# -----------------------------------
colA, colB, colC, colD = st.columns(4)

colA.metric("Total Transactions", f"{len(df):,}")
colB.metric("Fraud Cases", f"{df['Class'].sum():,}")
colC.metric("Features", df.shape[1])
colD.metric("Model", "LightGBM")

st.divider()

# -----------------------------------
# MAIN DASHBOARD (3 COLUMNS)
# -----------------------------------
col1, col2, col3 = st.columns([2,3,2])

# -----------------------------------
# LEFT PANEL - DATA CONTROL
# -----------------------------------
with col1:
    st.markdown("## 📂 Data Control")

    txn_id = st.slider("Select Transaction", 0, len(df)-1, 0)
    selected_txn = df.iloc[txn_id]

    st.markdown("### 📄 Transaction Data")
    st.dataframe(selected_txn.to_frame().T, use_container_width=True)

    run_btn = st.button("⚡ Run AI Analysis")

# -----------------------------------
# CENTER PANEL - FRAUD GAUGE
# -----------------------------------
with col2:
    st.markdown("## 📊 Risk Engine")

    if run_btn:
        input_data = selected_txn.values.reshape(1, -1)
        prob = model.predict_proba(input_data)[0][1]

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title={'text': "Fraud Risk %"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#00D9FF"},
                'steps': [
                    {'range': [0, 50], 'color': "#00FF9C"},
                    {'range': [50, 80], 'color': "#FFA500"},
                    {'range': [80, 100], 'color': "#FF3B3B"}
                ]
            }
        ))

        st.plotly_chart(fig, use_container_width=True)

# -----------------------------------
# RIGHT PANEL - VERDICT
# -----------------------------------
with col3:
    st.markdown("## 🧾 Decision")

    if run_btn:
        pred = model.predict(input_data)[0]

        if pred == 1:
            st.error("🚨 FRAUD DETECTED")
        else:
            st.success("✅ SAFE TRANSACTION")

        st.metric("Confidence", f"{prob*100:.2f}%")

# -----------------------------------
# AI EXPLANATION (HUGGING FACE)
# -----------------------------------
st.divider()

if run_btn:
    st.markdown("## 🤖 AI Financial Explanation")

    explanation = generate_explanation(selected_txn, pred, prob)
    st.write(explanation)

# -----------------------------------
# EXTRA VISUALS (PRO LEVEL)
# -----------------------------------
st.divider()

st.markdown("## 📊 Dataset Insights")

col4, col5 = st.columns(2)

# Fraud Distribution Pie Chart
with col4:
    fraud_counts = df['Class'].value_counts()

    fig_pie = go.Figure(data=[go.Pie(
        labels=["Safe", "Fraud"],
        values=fraud_counts.values,
        hole=0.5
    )])

    fig_pie.update_layout(title="Fraud Distribution")
    st.plotly_chart(fig_pie, use_container_width=True)

# Feature Correlation Heatmap
with col5:
    corr = df.corr().values

    fig_heatmap = go.Figure(data=go.Heatmap(
        z=corr,
        colorscale='RdBu'
    ))

    fig_heatmap.update_layout(title="Feature Correlation Heatmap")
    st.plotly_chart(fig_heatmap, use_container_width=True)

# -----------------------------------
# FOOTER
# -----------------------------------
st.divider()
st.markdown("Built with AI • Streamlit • LightGBM • Hugging Face")