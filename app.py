import streamlit as st
import joblib
import os
import traceback

# ---------------- BASIC TEST ---------------- #

st.set_page_config(page_title="Cancer Digital Twin", layout="wide")

st.write("✅ APP STARTED SUCCESSFULLY")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.write("📁 Project folder:", BASE_DIR)


# ---------------- SAFE LOAD FUNCTION ---------------- #

def safe_load(name):
    path = os.path.join(BASE_DIR, name)

    st.write(f"📦 Trying to load: {path}")

    if not os.path.exists(path):
        st.error(f"❌ File NOT FOUND: {name}")
        st.stop()

    try:
        obj = joblib.load(path)
        st.success(f"✅ Loaded: {name}")
        return obj

    except Exception as e:
        st.error(f"❌ Failed to load {name}")
        st.code(traceback.format_exc())
        st.stop()


# ---------------- LOAD ALL FILES ---------------- #

st.write("⏳ Loading files...")

pcr_model = safe_load("pcr_model.pkl")
rec_model = safe_load("rec_model.pkl")

pcr_cols = safe_load("feature_columns_pcr.pkl")
rec_cols = safe_load("feature_columns_rec.pkl")

encoders = safe_load("encoder.pkl")

st.success("🎉 ALL FILES LOADED SUCCESSFULLY")


# ---------------- BASIC UI TEST ---------------- #

st.divider()

st.title("🧬 Cancer Digital Twin Dashboard")

st.subheader("Debug Mode Working")

st.write("If you see this, your app is running correctly.")


# ---------------- SIMPLE INPUT TEST ---------------- #

age = st.slider("Age", 18, 90, 50)

er = st.selectbox("ER Status", ["Positive", "Negative"])

if st.button("Test Prediction"):

    st.info("Preparing dummy input...")

    import pandas as pd

    sample = {}

    for col in pcr_cols:
        sample[col] = 0

    if "Age" in sample:
        sample["Age"] = age

    if "ER_Status" in sample:
        sample["ER_Status"] = 1 if er == "Positive" else 0

    df = pd.DataFrame([sample])

    st.write("Input Data:")
    st.dataframe(df)

    try:
        pcr = pcr_model.predict_proba(df)[0][1]
        rec = rec_model.predict_proba(df)[0][1]

        st.success("Prediction Successful")

        st.metric("pCR Probability", f"{pcr*100:.2f}%")
        st.metric("Recurrence Risk", f"{rec*100:.2f}%")

    except Exception as e:

        st.error("❌ Prediction Failed")
        st.code(traceback.format_exc())


