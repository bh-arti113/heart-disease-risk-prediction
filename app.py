
"""
CARDIO INSIGHT
Heart Disease Prediction Research Dashboard

Educational research demonstration only.
Not intended for clinical diagnosis.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# 1. CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Cardio Insight | Heart Disease Research",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

ROOT = Path(__file__).resolve().parent

MODEL_PATH = ROOT / "models" / "final_model.joblib"
METADATA_PATH = ROOT / "models" / "model_metadata.json"

DATA_PATH = (
    ROOT / "data" / "processed"
    / "heart_disease_model_input.csv"
)

REPORTS_DIR = ROOT / "reports"

IMPORTANCE_PATH = (
    REPORTS_DIR / "shap_original_feature_importance.csv"
)

st.session_state.setdefault("prediction", None)
st.session_state.setdefault("patient_input", None)
st.session_state.setdefault("shap_values", None)



# ============================================================
# 2. CUSTOM STYLING - HIGH CONTRAST LIGHT THEME
# ============================================================

st.markdown(
    """
    <style>

    @import url(
        'https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap'
    );

    /* -----------------------------------------------------
       GLOBAL APPLICATION
    ----------------------------------------------------- */

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif !important;
    }

    .stApp {
        background-color: #F4F7FB !important;
        color: #173A56 !important;
    }

    [data-testid="stAppViewContainer"] {
        background-color: #F4F7FB !important;
    }

    [data-testid="stMain"] {
        background-color: #F4F7FB !important;
    }

    [data-testid="stHeader"] {
        background-color: transparent !important;
    }

    .main .block-container {
        padding-top: 2rem !important;
        padding-bottom: 3rem !important;
        max-width: 1450px;
    }

    /* -----------------------------------------------------
       MAIN CONTENT TEXT
    ----------------------------------------------------- */

    [data-testid="stMain"] .stMarkdown,
    [data-testid="stMain"] .stMarkdown p,
    [data-testid="stMain"] .stMarkdown li,
    [data-testid="stMain"] .stMarkdown span {
        color: #334E68 !important;
    }

    [data-testid="stMain"] h1,
    [data-testid="stMain"] h2,
    [data-testid="stMain"] h3,
    [data-testid="stMain"] h4 {
        color: #173A56 !important;
        font-weight: 750 !important;
    }

    [data-testid="stMain"] .stCaption,
    [data-testid="stMain"] [data-testid="stCaptionContainer"] {
        color: #526B80 !important;
    }

    /* -----------------------------------------------------
       SIDEBAR
    ----------------------------------------------------- */

    [data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #102B46 0%,
            #153D5B 100%
        ) !important;
    }

    [data-testid="stSidebar"] * {
        color: #FFFFFF !important;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #D6E3ED !important;
    }

    [data-testid="stSidebar"] hr {
        border-color: #35516A !important;
    }

    [data-testid="stSidebar"] label {
        color: #FFFFFF !important;
    }

    /* -----------------------------------------------------
       HERO HEADER
    ----------------------------------------------------- */

    .hero {
        background: linear-gradient(
            120deg,
            #102B46 0%,
            #174B68 65%,
            #147C83 100%
        );

        padding: 34px;
        border-radius: 20px;
        color: #FFFFFF !important;
        margin-bottom: 25px;
        box-shadow: 0 8px 25px rgba(16, 43, 70, 0.12);
    }

    .hero h1 {
        color: #FFFFFF !important;
        font-size: 34px !important;
        font-weight: 800 !important;
        margin-bottom: 10px !important;
    }

    .hero p {
        color: #E0EDF5 !important;
        font-size: 15px;
    }

    .hero .small-label {
        color: #A9DCD9 !important;
    }

    /* -----------------------------------------------------
       SECTION HEADINGS
    ----------------------------------------------------- */

    .section-heading {
        color: #173A56 !important;
        font-size: 24px !important;
        font-weight: 800 !important;
        margin-top: 28px;
        margin-bottom: 18px;
    }

    /* -----------------------------------------------------
       METRIC CARDS
    ----------------------------------------------------- */

    .metric-card {
        background: #FFFFFF !important;
        padding: 22px;
        border: 1px solid #DCE6EF;
        border-radius: 15px;
        min-height: 125px;
        box-shadow: 0 4px 14px rgba(20, 50, 75, 0.05);
    }

    .metric-value {
        font-size: 29px;
        font-weight: 800;
        color: #173A56 !important;
        overflow-wrap: anywhere;
    }

    .metric-description {
        font-size: 13px;
        color: #526B80 !important;
        margin-top: 5px;
    }

    .small-label {
        color: #526B80 !important;
        font-size: 12px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* -----------------------------------------------------
       INFORMATION CARDS
    ----------------------------------------------------- */

    .info-card {
        background: #FFFFFF !important;
        border: 1px solid #DCE6EF;
        padding: 24px;
        border-radius: 16px;
        margin-bottom: 16px;
        box-shadow: 0 4px 14px rgba(20, 50, 75, 0.04);
    }

    .info-card h3 {
        color: #173A56 !important;
        font-size: 20px !important;
        font-weight: 750 !important;
    }

    .info-card p {
        color: #526B80 !important;
        line-height: 1.7;
    }

    /* -----------------------------------------------------
       RESEARCH DISCLAIMER
    ----------------------------------------------------- */

    .disclaimer {
        background: #FFF7E8 !important;
        border: 1px solid #F0D8A8;
        border-left: 5px solid #D99B30;
        padding: 17px;
        border-radius: 9px;
        color: #684B19 !important;
        font-size: 14px;
        line-height: 1.6;
    }

    .disclaimer b {
        color: #684B19 !important;
    }

    /* -----------------------------------------------------
       INPUT LABELS - IMPORTANT FIX
    ----------------------------------------------------- */

    [data-testid="stMain"] [data-testid="stWidgetLabel"],
    [data-testid="stMain"] [data-testid="stWidgetLabel"] p,
    [data-testid="stMain"] [data-testid="stWidgetLabel"] label,
    [data-testid="stMain"] .stNumberInput label,
    [data-testid="stMain"] .stSelectbox label,
    [data-testid="stMain"] .stTextInput label,
    [data-testid="stMain"] .stSlider label,
    [data-testid="stMain"] label {
        color: #173A56 !important;
        font-weight: 700 !important;
        font-size: 14px !important;
    }

    /* -----------------------------------------------------
       FORM HEADINGS
    ----------------------------------------------------- */

    [data-testid="stMain"] [data-testid="stForm"] h1,
    [data-testid="stMain"] [data-testid="stForm"] h2,
    [data-testid="stMain"] [data-testid="stForm"] h3 {
        color: #173A56 !important;
        font-weight: 800 !important;
    }

    /* -----------------------------------------------------
       INPUT FIELDS
    ----------------------------------------------------- */

    [data-testid="stMain"] input {
        background-color: #FFFFFF !important;
        color: #173A56 !important;
        border: 1px solid #C5D3DF !important;
        border-radius: 9px !important;
        -webkit-text-fill-color: #173A56 !important;
    }

    [data-testid="stMain"] input::placeholder {
        color: #72869A !important;
    }

    [data-testid="stMain"] [data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        border: 1px solid #C5D3DF !important;
        border-radius: 9px !important;
        color: #173A56 !important;
    }

    [data-testid="stMain"] [data-baseweb="select"] span {
        color: #173A56 !important;
    }

    [data-testid="stMain"] [data-baseweb="select"] input {
        color: #173A56 !important;
    }

    [data-testid="stMain"] [data-baseweb="select"] svg {
        fill: #173A56 !important;
    }

    [data-testid="stMain"] [data-baseweb="input"] > div {
        background-color: #FFFFFF !important;
        border-color: #C5D3DF !important;
    }

    /* Dropdown popup options */

    [data-baseweb="popover"],
    [data-baseweb="menu"],
    [role="listbox"] {
        background-color: #FFFFFF !important;
    }

    [role="option"] {
        color: #173A56 !important;
        background-color: #FFFFFF !important;
    }

    [role="option"]:hover {
        background-color: #EAF4F5 !important;
    }

    /* -----------------------------------------------------
       FORM CONTAINER
    ----------------------------------------------------- */

    [data-testid="stForm"] {
        background: #FFFFFF !important;
        padding: 25px !important;
        border: 1px solid #DCE6EF !important;
        border-radius: 17px !important;
        box-shadow: 0 4px 14px rgba(20, 50, 75, 0.04);
    }

    /* -----------------------------------------------------
       EXPANDER - VIEW SUBMITTED INFORMATION
    ----------------------------------------------------- */

    [data-testid="stExpander"] {
        background: #FFFFFF !important;
        border: 1px solid #C5D3DF !important;
        border-radius: 12px !important;
    }

    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary span,
    [data-testid="stExpander"] summary p {
        color: #173A56 !important;
        font-weight: 700 !important;
    }

    [data-testid="stExpander"] [data-testid="stMarkdownContainer"] p {
        color: #334E68 !important;
    }

    /* -----------------------------------------------------
       STREAMLIT METRICS
    ----------------------------------------------------- */

    [data-testid="stMetric"] {
        background: #FFFFFF !important;
        padding: 16px;
        border: 1px solid #DCE6EF;
        border-radius: 12px;
    }

    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] p {
        color: #526B80 !important;
        font-weight: 700 !important;
    }

    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] div {
        color: #173A56 !important;
        font-weight: 800 !important;
    }

    /* -----------------------------------------------------
       DATAFRAMES AND TABLES
    ----------------------------------------------------- */

    [data-testid="stDataFrame"] {
        border: 1px solid #DCE6EF !important;
        border-radius: 12px !important;
        overflow: hidden;
    }

    [data-testid="stTable"] {
        color: #173A56 !important;
    }

    /* -----------------------------------------------------
       BUTTONS
    ----------------------------------------------------- */

    div.stButton > button,
    div.stFormSubmitButton > button {
        min-height: 45px;
        border-radius: 10px;
        font-weight: 750 !important;
        transition: 0.2s;
    }

    div.stButton > button[kind="primary"],
    div.stFormSubmitButton > button {
        background: #147C83 !important;
        color: #FFFFFF !important;
        border: 1px solid #147C83 !important;
    }

    div.stButton > button[kind="primary"]:hover,
    div.stFormSubmitButton > button:hover {
        background: #105F66 !important;
        border-color: #105F66 !important;
    }

    div.stButton > button[kind="secondary"] {
        background: #FFFFFF !important;
        color: #173A56 !important;
        border: 1px solid #C5D3DF !important;
    }

    /* -----------------------------------------------------
       ALERTS
    ----------------------------------------------------- */

    [data-testid="stAlert"] {
        border-radius: 12px !important;
    }

    /* -----------------------------------------------------
       FOOTER
    ----------------------------------------------------- */

    footer {
        visibility: hidden;
    }

    .custom-footer {
        text-align: center;
        padding: 22px;
        color: #526B80 !important;
        font-size: 13px;
    }

    .custom-footer b {
        color: #173A56 !important;
    }

    /* -----------------------------------------------------
       RESPONSIVE
    ----------------------------------------------------- */

    @media (max-width: 768px) {

        .hero {
            padding: 23px;
        }

        .hero h1 {
            font-size: 26px !important;
        }

        .section-heading {
            font-size: 21px !important;
        }

        [data-testid="stForm"] {
            padding: 15px !important;
        }

    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 3. LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = joblib.load(MODEL_PATH)

    with open(METADATA_PATH, "r") as file:
        metadata = json.load(file)

    return model, metadata


try:

    if not MODEL_PATH.exists():
        st.error("Final model file is missing.")
        st.stop()

    if not METADATA_PATH.exists():
        st.error("Model metadata file is missing.")
        st.stop()

    model, metadata = load_model()

except Exception as error:

    st.error(f"Model loading failed: {error}")
    st.stop()


# ============================================================
# 4. SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="padding: 12px 4px 25px 4px;">
            <div style="font-size:32px;">❤️</div>
            <div style="
                font-size:22px;
                font-weight:800;
                color:white;
            ">
                CARDIO INSIGHT
            </div>
            <div style="
                font-size:12px;
                color:#AFC5D5;
            ">
                HEART DISEASE RESEARCH
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Prediction",
            "Explainability",
            "About Project"
        ],
        label_visibility="visible"
    )

    st.markdown("---")

    st.markdown(
        """
        **MODEL STATUS**

        🟢 Model loaded

        **DATASET**

        Cleveland Heart Disease

        **MODE**

        Research demonstration
        """
    )

    st.markdown("---")

    st.caption("Developed using Python, Scikit-learn, SHAP and Streamlit.")


# ============================================================
# 5. REUSABLE HEADER
# ============================================================

def render_header(title, subtitle):

    st.markdown(
        f"""
        <div class="hero">
            <div class="small-label" style="color:#A9DCD9;">
                HEART DISEASE RESEARCH PLATFORM
            </div>
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_disclaimer():

    st.markdown(
        """
        <div class="disclaimer">
        <b>Research demonstration only.</b>
        This application is not a medical diagnostic tool.
        Model outputs are experimental and must not be used
        to make medical decisions or replace professional care.
        </div>
        """,
        unsafe_allow_html=True
    )


def render_metric(value, label, description):

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="small-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-description">
                {description}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# 6. OVERVIEW PAGE
# ============================================================

if page == "Overview":

    render_header(
        "Heart Disease Research Dashboard",
        "Explore an experimental machine learning system "
        "for binary heart disease classification."
    )

    render_disclaimer()

    st.markdown(
        '<div class="section-heading">Project at a glance</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        render_metric(
            "303",
            "DATA RECORDS",
            "Cleveland dataset"
        )

    with c2:
        render_metric(
            "13",
            "INPUT FEATURES",
            "Patient measurements"
        )

    with c3:
        render_metric(
            "5-fold",
            "OUTER CV",
            "Nested validation"
        )

    with c4:
        render_metric(
            metadata["model_name"],
            "FINAL MODEL",
            "Selected candidate"
        )

    st.markdown(
        '<div class="section-heading">About this project</div>',
        unsafe_allow_html=True
    )

    left, right = st.columns(2)

    with left:

        st.markdown(
            """
            <div class="info-card">
                <h3 style="color:#173A56;">
                    🎯 Project objective
                </h3>
                <p style="color:#647C90;">
                    Develop and evaluate machine learning
                    classification models using the Cleveland
                    Heart Disease dataset.
                </p>
            </div>

            <div class="info-card">
                <h3 style="color:#173A56;">
                    🧠 Machine learning
                </h3>
                <p style="color:#647C90;">
                    Compare multiple classification algorithms,
                    perform hyperparameter tuning and evaluate
                    the model-selection process.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with right:

        st.markdown(
            """
            <div class="info-card">
                <h3 style="color:#173A56;">
                    🔍 Explainability
                </h3>
                <p style="color:#647C90;">
                    Use permutation importance and SHAP to
                    explore global and individual model behaviour.
                </p>
            </div>

            <div class="info-card">
                <h3 style="color:#173A56;">
                    📊 Evaluation
                </h3>
                <p style="color:#647C90;">
                    Examine ROC-AUC, precision, recall,
                    specificity and other classification metrics.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        '<div class="section-heading">Model information</div>',
        unsafe_allow_html=True
    )

    
    m1, m2, m3 = st.columns(3)

    with m1:
        render_metric(
            f"{metadata['best_cv_roc_auc']:.4f}",
            "CROSS-VALIDATION ROC-AUC",
            "Model tuning CV score"
            )

    with m2:
        render_metric(
            str(metadata["training_records"]),
            "TRAINING RECORDS",
            "Development dataset"
            )

    with m3:
        render_metric(
            f"{metadata.get('prediction_threshold', 0.5):.2f}",
            "PREDICTION THRESHOLD",
            "Classification decision threshold"
            )


    st.caption(
        "The displayed ROC-AUC is the final candidate's "
        "full-development-data tuning CV score, not the "
        "nested CV outer-fold performance estimate."
    )


# ============================================================
# 7. PREDICTION PAGE
# ============================================================

elif page == "Prediction":

    render_header(
        "Patient Prediction",
        "Enter the dataset-coded measurements to explore "
        "the model's output."
    )

    render_disclaimer()

    st.markdown(
        '<div class="section-heading">Patient measurements</div>',
        unsafe_allow_html=True
    )

    with st.form("patient_form"):

        left, right = st.columns(2)

        with left:

            st.markdown("### Demographic and clinical")

            age = st.number_input(
                "Age (years)",
                1, 120, 50
            )

            sex = st.selectbox(
                "Sex",
                [1, 0],
                format_func=lambda x: (
                    "Male (1)" if x == 1 else "Female (0)"
                )
            )

            cp = st.selectbox(
                "Chest pain type",
                [1, 2, 3, 4],
                format_func=lambda x: {
                    1: "Typical angina (1)",
                    2: "Atypical angina (2)",
                    3: "Non-anginal pain (3)",
                    4: "Asymptomatic (4)"
                }[x]
            )

            trestbps = st.number_input(
                "Resting blood pressure (mm Hg)",
                50, 250, 120
            )

            chol = st.number_input(
                "Serum cholesterol (mg/dl)",
                50, 700, 200
            )

            fbs = st.selectbox(
                "Fasting blood sugar > 120 mg/dl",
                [0, 1],
                format_func=lambda x: (
                    "False (0)" if x == 0 else "True (1)"
                )
            )

            restecg = st.selectbox(
                "Resting ECG",
                [0, 1, 2],
                format_func=lambda x: {
                    0: "Normal (0)",
                    1: "ST-T abnormality (1)",
                    2: "LV hypertrophy (2)"
                }[x]
            )

        with right:

            st.markdown("### Exercise and cardiac measurements")

            thalach = st.number_input(
                "Maximum heart rate achieved",
                50, 250, 150
            )

            exang = st.selectbox(
                "Exercise-induced angina",
                [0, 1],
                format_func=lambda x: (
                    "No (0)" if x == 0 else "Yes (1)"
                )
            )

            oldpeak = st.number_input(
                "ST depression (oldpeak)",
                0.0, 10.0, 1.0, step=0.1
            )

            slope = st.selectbox(
                "Slope of peak exercise ST segment",
                [1, 2, 3],
                format_func=lambda x: {
                    1: "Upsloping (1)",
                    2: "Flat (2)",
                    3: "Downsloping (3)"
                }[x]
            )

            ca = st.selectbox(
                "Number of major vessels (ca)",
                [0, 1, 2, 3]
            )

            thal = st.selectbox(
                "Thalassemia (thal)",
                [3, 6, 7],
                format_func=lambda x: {
                    3: "Normal (3)",
                    6: "Fixed defect (6)",
                    7: "Reversible defect (7)"
                }[x]
            )

        st.markdown("---")

        submitted = st.form_submit_button(
            "Generate prediction",
            type="primary",
            use_container_width=True
        )

    if submitted:

        input_data = pd.DataFrame([{
            "age": age,
            "sex": sex,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs,
            "restecg": restecg,
            "thalach": thalach,
            "exang": exang,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "thal": thal
        }])

        if list(input_data.columns) != metadata["feature_columns"]:

            st.error("Input feature order mismatch.")

        else:

            try:

                prediction = int(
                    model.predict(input_data)[0]
                )

                probabilities = model.predict_proba(
                    input_data
                )[0]

                positive_index = list(
                    model.classes_
                ).index(1)

                probability = float(
                    probabilities[positive_index]
                )

                st.session_state.prediction = {
                    "class": prediction,
                    "probability": probability
                }

                st.session_state.patient_input = input_data
                st.session_state.shap_values = None

            except Exception as error:

                st.error(f"Prediction failed: {error}")

    # Results remain visible after Streamlit reruns.
    if st.session_state.prediction is not None:

        result = st.session_state.prediction

        st.markdown(
            '<div class="section-heading">Prediction result</div>',
            unsafe_allow_html=True
        )

        r1, r2 = st.columns([1.3, 1])

        with r1:

            if result["class"] == 1:

                st.warning(
                    "The model predicts the positive class."
                )

            else:

                st.success(
                    "The model predicts the negative class."
                )

            st.markdown(
                """
                <div class="info-card">
                    <b>Interpretation</b>
                    <p style="color:#647C90;">
                    This is an experimental classification output.
                    It is not a clinical diagnosis.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

        with r2:

            st.metric(
                "Positive-class model output",
                f"{result['probability']:.1%}"
            )

            st.progress(result["probability"])

        with st.expander("View submitted measurements"):

            st.dataframe(
                st.session_state.patient_input,
                use_container_width=True
            )

        st.markdown(
            '<div class="section-heading">Individual explanation</div>',
            unsafe_allow_html=True
        )

        if st.button(
            "Generate SHAP explanation",
            type="primary"
        ):

            try:

                import shap

                if not DATA_PATH.exists():
                    raise FileNotFoundError(
                        "Processed dataset not found."
                    )

                dataset = pd.read_csv(DATA_PATH)

                X_background = dataset.drop(
                    columns=["target"]
                )

                preprocessor = model.named_steps[
                    "preprocessing"
                ]

                classifier = model.named_steps[
                    "classifier"
                ]

                background = X_background.sample(
                    n=min(30, len(X_background)),
                    random_state=42
                )

                background_t = preprocessor.transform(
                    background
                )

                patient_t = preprocessor.transform(
                    st.session_state.patient_input
                )

                if hasattr(background_t, "toarray"):
                    background_t = background_t.toarray()

                if hasattr(patient_t, "toarray"):
                    patient_t = patient_t.toarray()

                names = preprocessor.get_feature_names_out()

                background_df = pd.DataFrame(
                    background_t,
                    columns=names
                )

                patient_df = pd.DataFrame(
                    patient_t,
                    columns=names
                )

                explainer = shap.PermutationExplainer(
                    classifier.predict_proba,
                    background_df
                )

                explanation = explainer(
                    patient_df,
                    max_evals=2 * patient_df.shape[1] + 1
                )

                positive_index = list(
                    model.classes_
                ).index(1)

                if explanation.values.ndim == 3:
                    explanation = explanation[
                        :, :, positive_index
                    ]

                st.session_state.shap_values = explanation

            except Exception as error:

                st.error(
                    f"SHAP explanation failed: {error}"
                )

        if st.session_state.shap_values is not None:

            fig = plt.figure(figsize=(10, 7))

            import shap

            shap.plots.waterfall(
                st.session_state.shap_values[0],
                max_display=15,
                show=False
            )

            fig = plt.gcf()

            st.pyplot(fig, use_container_width=True)

            plt.close(fig)


# ============================================================
# 8. EXPLAINABILITY PAGE
# ============================================================

elif page == "Explainability":

    render_header(
        "Model Explainability",
        "Explore which features influence the model's outputs."
    )

    render_disclaimer()

    st.markdown(
        '<div class="section-heading">Global feature importance</div>',
        unsafe_allow_html=True
    )

    if IMPORTANCE_PATH.exists():

        importance = pd.read_csv(IMPORTANCE_PATH)

        importance = importance.sort_values(
            "Mean Absolute Grouped SHAP",
            ascending=True
        )

        fig, ax = plt.subplots(figsize=(10, 7))

        ax.barh(
            importance["Feature"],
            importance["Mean Absolute Grouped SHAP"],
            color="#168A8A"
        )

        ax.set_xlabel("Mean absolute grouped SHAP")
        ax.set_ylabel("Original feature")
        ax.set_title(
            "SHAP Importance by Original Feature",
            fontsize=14,
            fontweight="bold"
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig.tight_layout()

        st.pyplot(fig, use_container_width=True)

        plt.close(fig)

        st.markdown(
            '<div class="section-heading">Importance values</div>',
            unsafe_allow_html=True
        )

        st.dataframe(
            importance.sort_values(
                "Mean Absolute Grouped SHAP",
                ascending=False
            ).round(4),
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "Generate the grouped SHAP report in the "
            "explainability notebook first."
        )

    st.markdown(
        """
        <div class="info-card">
            <h3 style="color:#173A56;">
                Understanding SHAP
            </h3>
            <p style="color:#647C90;">
                SHAP values describe how features contribute
                to the model's output relative to a background
                dataset. They do not establish causation or
                medical importance.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# 9. ABOUT PROJECT PAGE
# ============================================================

elif page == "About Project":

    render_header(
        "About the Research Project",
        "Methodology, technologies and research limitations."
    )

    st.markdown(
        '<div class="section-heading">Technology stack</div>',
        unsafe_allow_html=True
    )

    technologies = pd.DataFrame({
        "Technology": [
            "Python",
            "Pandas and NumPy",
            "Scikit-learn",
            "SHAP",
            "Matplotlib",
            "Streamlit"
        ],
        "Purpose": [
            "Core programming language",
            "Data processing",
            "Machine learning and evaluation",
            "Model explainability",
            "Visualizations",
            "Interactive web application"
        ]
    })

    st.dataframe(
        technologies,
        use_container_width=True,
        hide_index=True
    )

    st.markdown(
        '<div class="section-heading">Research methodology</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        1. Dataset auditing and validation.
        2. Data preprocessing within machine learning pipelines.
        3. Baseline model comparison.
        4. Nested stratified cross-validation.
        5. Hyperparameter tuning.
        6. Global and local explainability.
        7. Interactive application development.
        """
    )

    st.markdown(
        '<div class="section-heading">Research limitations</div>',
        unsafe_allow_html=True
    )

    st.warning(
        """
        The model uses a small historical dataset of 303 records.
        Its performance has not established clinical validity.
        Independent external validation and further research
        are required before any clinical use could be considered.
        """
    )


# ============================================================
# 10. FOOTER
# ============================================================


st.markdown(
    """
    <div class="custom-footer">
        <b>CARDIO INSIGHT</b><br>
        Heart Disease Risk Prediction Research Project<br>
        Built with Python · Scikit-learn · SHAP · Streamlit<br><br>
        Educational research demonstration only.
    </div>
    """,
    unsafe_allow_html=True
)