# satellite_dashboard.py
import streamlit as st
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
from preprocess import preprocess_df

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Satellite Collision Risk Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# Custom CSS for Background & Styling
# -------------------------------
st.markdown("""
    <style>
        body {
            background: linear-gradient(to bottom right, #ffffff, #ffd700);
            color: black;
        }
        .stApp {
            background: linear-gradient(to bottom right, #ffffff, #ffd700);
        }
        h1, h2, h3, h4, h5, h6, p, div {
            color: black !important;
        }
        /* Sidebar background */
        section[data-testid="stSidebar"] {
            background: linear-gradient(to bottom right, #ffffff, #ffd700);
        }
        /* Dataframe container cleanup */
        .stDataFrame {
            background-color: white !important;
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# Styled Title
# -------------------------------
st.markdown("""
    <h1 style='text-align: center; color: black;'>
        📊 <span style='color:#DAA520;'>Satellite Collision Risk</span> Dashboard
    </h1>
    <p style='text-align: center; color: #111; font-size:16px;'>
        Predict collision risk of LEO satellites with interactive charts and tables
    </p>
""", unsafe_allow_html=True)

# -------------------------------
# Load Model
# -------------------------------
MODEL_PATH = "collision_model.pkl"
if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file not found. Please run train.py first.")
    st.stop()

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# -------------------------------
# Sidebar - File Upload
# -------------------------------
st.sidebar.header("Upload Satellite Data")
uploaded_file = st.sidebar.file_uploader("📂 CSV file", type=["csv"])

sample_file = None
if os.path.exists("data"):
    sample_files = [f for f in os.listdir("data") if f.endswith(".csv")]
    if sample_files:
        sample_file = st.sidebar.selectbox("Or choose a sample CSV", ["(none)"] + sample_files)

# -------------------------------
# Load Data
# -------------------------------
df = None
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
elif sample_file and sample_file != "(none)":
    df = pd.read_csv(os.path.join("data", sample_file))
else:
    st.info("Upload a CSV or choose a sample file to see predictions.")
    st.stop()

st.subheader("📊 Data Preview")
st.dataframe(df.head())

# -------------------------------
# Preprocess Data
# -------------------------------
df_proc, X = preprocess_df(df)

# -------------------------------
# Predictions
# -------------------------------
try:
    if "PredictedRiskProb" in df_proc.columns:
        # Use probabilities from CSV (demo mode)
        df_proc["PredictedRisk"] = (df_proc["PredictedRiskProb"] >= 0.5).astype(int)
    else:
        pred_probs = model.predict_proba(X)[:, 1]
        df_proc["PredictedRiskProb"] = pred_probs
        threshold = 0.5
        df_proc["PredictedRisk"] = (pred_probs >= threshold).astype(int)

    # -------------------------------
    # Styled Table with Risk Badges (HTML version)
    # -------------------------------
    st.subheader("🛰️ Collision Risk Predictions")
    display_cols = ["satellite_id", "Altitude_km", "Inclination_rad", "Eccentricity",
                    "PredictedRiskProb", "PredictedRisk"]

    table_html = "<table style='width:100%; border-collapse:collapse; background-color:white; border-radius:10px;'>"
    # Header
    table_html += "<tr style='background-color:#FFD700; color:black; font-weight:bold;'>"
    for col in display_cols:
        table_html += f"<th style='padding:8px; border:1px solid #ddd; text-align:center;'>{col}</th>"
    table_html += "</tr>"

    # Rows
    for _, row in df_proc[display_cols].head(20).iterrows():
        table_html += "<tr>"
        for col in display_cols:
            val = row[col]
            if col == "PredictedRisk":
                if val == 1:
                    cell = "<span style='background-color:#FF6347; color:white; padding:4px 8px; border-radius:6px;'>High</span>"
                else:
                    cell = "<span style='background-color:#32CD32; color:white; padding:4px 8px; border-radius:6px;'>Low</span>"
                table_html += f"<td style='padding:8px; border:1px solid #ddd; text-align:center;'>{cell}</td>"
            elif col == "PredictedRiskProb":
                table_html += f"<td style='padding:8px; border:1px solid #ddd; text-align:center;'>{val:.3f}</td>"
            else:
                table_html += f"<td style='padding:8px; border:1px solid #ddd; text-align:center;'>{val}</td>"
        table_html += "</tr>"
    table_html += "</table>"

    st.markdown(table_html, unsafe_allow_html=True)

    # -------------------------------
    # Scatter Plot
    # -------------------------------
    st.subheader("📈 Risk Scatter (Altitude vs Inclination)")
    fig, ax = plt.subplots(figsize=(8,5))
    sc = ax.scatter(
        df_proc['Altitude_km'],
        df_proc['Inclination_rad'],
        c=df_proc['PredictedRiskProb'],
        cmap='plasma',
        s=80,
        edgecolors='k'
    )
    ax.set_xlabel('Altitude (km)')
    ax.set_ylabel('Inclination (rad)')
    ax.set_title('Predicted Collision Risk Probability', color="black")
    plt.colorbar(sc, ax=ax, label='Risk probability')
    st.pyplot(fig)

    # -------------------------------
    # Download CSV
    # -------------------------------
    csv = df_proc.to_csv(index=False).encode("utf-8")
    st.download_button(
        "💾 Download Predictions",
        data=csv,
        file_name="collision_predictions.csv",
        mime="text/csv"
    )

except Exception as e:
    st.error(f"⚠️ Prediction failed: {e}")

