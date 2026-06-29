import streamlit as st
import os

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Breast Cancer Predictor",
    page_icon="🎗️",
    layout="wide",
)

# ── Load model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    from model import load_artifacts, train, preprocess, load_data, save_artifacts
    from sklearn.model_selection import train_test_split

    model_path  = "model/knn_model.pkl"
    scaler_path = "model/scaler.pkl"

    if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
        df = load_data()
        X, y = preprocess(df)
        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        model, scaler = train(X_train, y_train)
        save_artifacts(model, scaler)
    else:
        model, scaler = load_artifacts(model_path, scaler_path)

    return model, scaler


model, scaler = load_model()

# ── Constants ──────────────────────────────────────────────────────────────────
FEATURES = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
    "smoothness_mean", "compactness_mean", "concavity_mean",
    "concave points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se",
    "smoothness_se", "compactness_se", "concavity_se",
    "concave points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
    "smoothness_worst", "compactness_worst", "concavity_worst",
    "concave points_worst", "symmetry_worst", "fractal_dimension_worst",
]

RANGES = {
    "radius_mean":              (14.1, 6.9,   28.1),
    "texture_mean":             (19.3, 9.7,   39.3),
    "perimeter_mean":           (91.9, 43.8,  188.5),
    "area_mean":                (654.9, 143.5, 2501.0),
    "smoothness_mean":          (0.096, 0.05,  0.163),
    "compactness_mean":         (0.104, 0.02,  0.345),
    "concavity_mean":           (0.089, 0.0,   0.427),
    "concave points_mean":      (0.049, 0.0,   0.201),
    "symmetry_mean":            (0.181, 0.11,  0.304),
    "fractal_dimension_mean":   (0.063, 0.05,  0.097),
    "radius_se":                (0.405, 0.11,  2.87),
    "texture_se":               (1.217, 0.36,  4.88),
    "perimeter_se":             (2.866, 0.76,  21.98),
    "area_se":                  (40.34, 6.8,   542.2),
    "smoothness_se":            (0.007, 0.0,   0.031),
    "compactness_se":           (0.025, 0.0,   0.135),
    "concavity_se":             (0.032, 0.0,   0.396),
    "concave points_se":        (0.012, 0.0,   0.053),
    "symmetry_se":              (0.021, 0.008, 0.079),
    "fractal_dimension_se":     (0.004, 0.001, 0.03),
    "radius_worst":             (16.3,  7.9,   36.0),
    "texture_worst":            (25.7,  12.0,  49.5),
    "perimeter_worst":          (107.3, 50.4,  251.2),
    "area_worst":               (880.6, 185.2, 4254.0),
    "smoothness_worst":         (0.132, 0.07,  0.223),
    "compactness_worst":        (0.254, 0.03,  1.058),
    "concavity_worst":          (0.272, 0.0,   1.252),
    "concave points_worst":     (0.115, 0.0,   0.291),
    "symmetry_worst":           (0.290, 0.16,  0.664),
    "fractal_dimension_worst":  (0.084, 0.06,  0.208),
}

# ── Session state defaults ─────────────────────────────────────────────────────
if "step" not in st.session_state:
    st.session_state.step = 1          # 1 = patient info, 2 = sliders, 3 = result
if "patient_name" not in st.session_state:
    st.session_state.patient_name = ""
if "patient_age" not in st.session_state:
    st.session_state.patient_age = 0
if "input_values" not in st.session_state:
    st.session_state.input_values = {}

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown(
        """
        **Model:** K-Nearest Neighbors (KNN)  
        **Dataset:** Wisconsin Breast Cancer Dataset  
        **Accuracy:** ~94%  
        **Features:** 30 numeric FNA biopsy measurements  
        """
    )
    st.markdown("---")
    st.markdown("Built by **Durga Charan Mallick**")

    # Allow restart from anywhere
    if st.session_state.step > 1:
        st.markdown("---")
        if st.button("🔄 Start Over", use_container_width=True):
            st.session_state.step = 1
            st.session_state.patient_name = ""
            st.session_state.patient_age  = 0
            st.session_state.input_values = {}
            st.rerun()

# ── Step indicator ─────────────────────────────────────────────────────────────
st.header("🎗️ Breast Cancer Prediction")

step_labels = ["Patient Info", "Tumor Parameters", "Result"]
cols_steps  = st.columns(3)
for i, label in enumerate(step_labels):
    step_num = i + 1
    if step_num < st.session_state.step:
        cols_steps[i].success(f"✅ Step {step_num}: {label}")
    elif step_num == st.session_state.step:
        cols_steps[i].info(f"▶️ Step {step_num}: {label}")
    else:
        cols_steps[i].empty()

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Patient Info
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.step == 1:
    st.subheader("👤 Patient Information")
    st.caption("Please enter the patient's basic details to begin.")

    with st.container():
        name = st.text_input("Patient Name", placeholder="e.g. Priya Sharma")
        age  = st.number_input("Patient Age", min_value=1, max_value=120, value=30, step=1)

        st.markdown(" ")
        if st.button("Submit & Continue →", type="primary", use_container_width=True):
            if not name.strip():
                st.warning("⚠️ Please enter the patient's name before continuing.")
            else:
                st.session_state.patient_name = name.strip()
                st.session_state.patient_age  = int(age)
                st.session_state.step = 2
                st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Tumor Feature Sliders
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 2:
    st.subheader(f"📋 Tumor Parameters — {st.session_state.patient_name}, Age {st.session_state.patient_age}")
    st.caption("Adjust the sliders to match the patient's biopsy report values, then click Predict.")

    input_values = {}
    slider_cols  = st.columns(3)

    for i, feat in enumerate(FEATURES):
        mean_val, min_val, max_val = RANGES[feat]
        col = slider_cols[i % 3]
        input_values[feat] = col.slider(
            label=feat.replace("_", " ").title(),
            min_value=float(min_val),
            max_value=float(max_val),
            value=float(mean_val),
            step=float(round((max_val - min_val) / 200, 5)),
            format="%.4f",
        )

    st.markdown("---")
    if st.button("🔍 Predict Now", type="primary", use_container_width=True):
        st.session_state.input_values = input_values
        st.session_state.step = 3
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Result
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 3:
    from model import predict_single

    label, proba = predict_single(
        st.session_state.input_values, model=model, scaler=scaler
    )
    benign_pct    = round(proba[0] * 100, 1)
    malignant_pct = round(proba[1] * 100, 1)

    name = st.session_state.patient_name
    age  = st.session_state.patient_age

    st.subheader("🧬 Diagnosis Report")

    # Patient card
    st.markdown(
        f"""
        <div style="background:#1e2a3a;padding:16px 24px;border-radius:10px;margin-bottom:20px;">
            <b style="font-size:1.1rem;">👤 Patient:</b>
            <span style="font-size:1.1rem;"> {name} &nbsp;|&nbsp; Age: {age}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Main result banner
    if label == "Malignant":
        st.error(
            f"### 🔴 {name} is diagnosed as **Malignant (Cancerous)**\n\n"
            f"Confidence: **{malignant_pct}%**"
        )
    else:
        st.success(
            f"### 🟢 {name} is diagnosed as **Benign (Non-Cancerous)**\n\n"
            f"Confidence: **{benign_pct}%**"
        )

    # Probability metrics
    col1, col2 = st.columns(2)
    col1.metric("🟢 Benign Probability",    f"{benign_pct}%")
    col2.metric("🔴 Malignant Probability", f"{malignant_pct}%")

    st.progress(int(malignant_pct), text=f"Malignant risk level: {malignant_pct}%")

    st.markdown("---")
    st.info(
        "⚠️ **Disclaimer:** This tool is for educational purposes only. "
        "Always consult a qualified medical professional for a confirmed diagnosis."
    )

    # Start new patient
    st.markdown(" ")
    if st.button("➕ New Patient", use_container_width=True):
        st.session_state.step = 1
        st.session_state.patient_name = ""
        st.session_state.patient_age  = 0
        st.session_state.input_values = {}
        st.rerun()
