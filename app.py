import streamlit as st
import pandas as pd
import joblib
import os


# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="Cancer Digital Twin",
    layout="wide"
)

st.title("🧬 Cancer Digital Twin Dashboard")
st.markdown("AI-based pCR & Recurrence Prediction System")

st.divider()


# ---------------- BASE PATH ---------------- #

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_file(name):
    return joblib.load(os.path.join(BASE_DIR, name))


# ---------------- LOAD FILES ---------------- #

@st.cache_resource
def load_models():

    pcr_model = load_file("pcr_model.pkl")
    rec_model = load_file("rec_model.pkl")

    pcr_cols = load_file("feature_columns_pcr.pkl")
    rec_cols = load_file("feature_columns_rec.pkl")

    encoder = load_file("encoder.pkl")

    return pcr_model, rec_model, pcr_cols, rec_cols, encoder


pcr_model, rec_model, pcr_cols, rec_cols, encoder = load_models()


# ---------------- SIDEBAR INPUT ---------------- #

st.sidebar.header("Patient Information")

age = st.sidebar.slider("Age", 18, 90, 50)
bmi = st.sidebar.number_input("BMI", 15.0, 45.0, 24.0)

menopause = st.sidebar.selectbox("Menopause Status", ["Pre", "Post"])
family = st.sidebar.selectbox("Family History", ["Yes", "No"])
symptoms = st.sidebar.selectbox(
    "Symptoms", ["Discharge", "Lump", "Pain", "Screening"]
)


st.sidebar.header("Tumor Staging")

t_stage = st.sidebar.selectbox("T Stage", ["T1", "T2", "T3", "T4"])
n_stage = st.sidebar.selectbox("N Stage", ["N0", "N1", "N2", "N3"])
m_stage = st.sidebar.selectbox("M Stage", ["M0", "M1"])

clinical_stage = st.sidebar.selectbox("Clinical Stage", ["I", "II", "III", "IV"])


st.sidebar.header("Pathology")

grade = st.sidebar.selectbox("Tumor Grade", ["1", "2", "3"])

histology = st.sidebar.selectbox(
    "Histology Type", ["IDC", "ILC", "Mixed", "Other"]
)

er = st.sidebar.selectbox("ER Status", ["Positive", "Negative"])
pr = st.sidebar.selectbox("PR Status", ["Positive", "Negative"])
her2 = st.sidebar.selectbox("HER2 Status", ["Positive", "Negative"])

ki67 = st.sidebar.slider("Ki67 (%)", 0, 100, 30)


st.sidebar.header("Treatment")

surgery = st.sidebar.selectbox(
    "Surgery Type", ["Lumpectomy", "Mastectomy", "None"]
)

chemo = st.sidebar.selectbox("Chemotherapy", ["Yes", "No"])

regimen = st.sidebar.selectbox(
    "Chemo Regimen", ["AC-T", "TC", "FEC", "None"]
)

cycles = st.sidebar.slider("Chemo Cycles", 0, 12, 6)

radiation = st.sidebar.selectbox("Radiation Therapy", ["Yes", "No"])

hormone = st.sidebar.selectbox("Hormone Therapy", ["Yes", "No"])

targeted = st.sidebar.selectbox("Targeted Therapy", ["Yes", "No"])

targeted_drug = st.sidebar.selectbox(
    "Targeted Drug", ["Trastuzumab", "Pertuzumab", "None"]
)


# ---------------- INPUT DATA ---------------- #

input_df = pd.DataFrame([{

    "Age": age,
    "BMI": bmi,
    "Menopause_Status": menopause,
    "Family_History": family,
    "Symptoms": symptoms,

    "Clinical_T_Stage": t_stage,
    "Clinical_N_Stage": n_stage,
    "Clinical_M_Stage": m_stage,
    "Clinical_Stage": clinical_stage,

    "Tumor_Grade_Biopsy": grade,
    "Histology_Type": histology,

    "ER_Status": er,
    "PR_Status": pr,
    "HER2_Status": her2,

    "Ki67_Percent": ki67,

    "Surgery_Type": surgery,

    "Chemo_Given": chemo,
    "Chemo_Regimen": regimen,
    "Chemo_Cycles": cycles,

    "Radiation_Given": radiation,
    "Hormone_Therapy_Given": hormone,

    "Targeted_Therapy_Given": targeted,
    "Targeted_Drug": targeted_drug
}])



# ---------------- ENCODE (FIXED) ---------------- #

encoder_cols = list(encoder.feature_names_in_)

for col in encoder_cols:
    if col not in input_df.columns:
        input_df[col] = "Unknown"

input_df[encoder_cols] = encoder.transform(input_df[encoder_cols])


# ---------------- ALIGN FEATURES ---------------- #

X_pcr = input_df.reindex(columns=pcr_cols, fill_value=0)
X_rec = input_df.reindex(columns=rec_cols, fill_value=0)


# ---------------- PREDICT ---------------- #

st.subheader("🔍 Prediction Results")

if st.button("🔮 Predict Outcome"):

    with st.spinner("Running AI Model..."):

        pcr_prob = pcr_model.predict_proba(X_pcr)[0][1]
        rec_prob = rec_model.predict_proba(X_rec)[0][1]


    col1, col2 = st.columns(2)

    col1.metric(
        "pCR Probability",
        f"{pcr_prob*100:.2f}%"
    )

    col2.metric(
        "Recurrence Risk",
        f"{rec_prob*100:.2f}%"
    )


    st.markdown("### 📌 Risk Interpretation")

    if rec_prob < 0.25:
        st.success("🟢 Low Recurrence Risk")

    elif rec_prob < 0.5:
        st.warning("🟡 Moderate Recurrence Risk")

    else:
        st.error("🔴 High Recurrence Risk")


