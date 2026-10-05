import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# ============================================================
# 1. PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 2. CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
        .block-container {
            max-width: 1200px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        .main-title {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }

        .subtitle {
            color: #6b7280;
            font-size: 1.05rem;
            margin-bottom: 1.5rem;
        }

        .status-card {
            padding: 1.4rem;
            border-radius: 14px;
            border: 1px solid rgba(128,128,128,0.25);
            margin-top: 1rem;
        }

        .small-text {
            color: #6b7280;
            font-size: 0.9rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3. MODEL LOADING
# ============================================================

MODEL_PATH = Path(__file__).parent / "machine_failure_model.pkl"


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH.name}"
        )

    return joblib.load(MODEL_PATH)


try:
    model = load_model()

except Exception as e:
    st.error("Unable to load the machine learning model.")
    st.exception(e)
    st.stop()


# ============================================================
# 4. MODEL INFORMATION
# ============================================================

EXPECTED_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Type",
]


# Check model classes
if not hasattr(model, "classes_"):
    st.error("The loaded model is not a valid fitted classifier.")
    st.stop()


if 1 not in model.classes_:
    st.error(
        "The model does not contain class 1 "
        "(machine failure)."
    )
    st.stop()


# Find the probability column corresponding to class 1
failure_class_index = list(model.classes_).index(1)


# ============================================================
# 5. HEADER
# ============================================================

st.markdown(
    '<div class="main-title">⚙️ Predictive Maintenance</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
        Machine failure risk estimation using a trained
        Random Forest classifier.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 6. SIDEBAR
# ============================================================

with st.sidebar:

    st.header("About")

    st.write(
        """
        This application estimates machine failure risk
        from operating conditions.
        """
    )

    st.divider()

    st.subheader("Model")

    st.write("**Algorithm:** Random Forest")

    st.write(
        f"**Number of classes:** {len(model.classes_)}"
    )

    if hasattr(model, "n_estimators"):
        st.write(
            f"**Trees:** {model.n_estimators}"
        )

    st.divider()

    st.caption(
        "The probability displayed by this application is "
        "a model estimate, not a guaranteed physical failure probability."
    )


# ============================================================
# 7. INPUT SECTION
# ============================================================

st.subheader("Machine operating conditions")

st.caption(
    "Enter the current measurements of the machine."
)


col1, col2, col3 = st.columns(3)


with col1:

    air_temp = st.number_input(
        "Air temperature (K)",
        min_value=250.0,
        max_value=350.0,
        value=300.0,
        step=0.1,
        help="Ambient air temperature around the machine.",
    )

    process_temp = st.number_input(
        "Process temperature (K)",
        min_value=250.0,
        max_value=400.0,
        value=310.0,
        step=0.1,
        help="Temperature measured during the industrial process.",
    )


with col2:

    rpm = st.number_input(
        "Rotational speed (rpm)",
        min_value=100,
        max_value=5000,
        value=1500,
        step=10,
        help="Rotational speed of the machine.",
    )

    torque = st.number_input(
        "Torque (Nm)",
        min_value=0.0,
        max_value=150.0,
        value=40.0,
        step=0.1,
        help="Mechanical torque measured at the machine.",
    )


with col3:

    tool_wear = st.number_input(
        "Tool wear (min)",
        min_value=0,
        max_value=500,
        value=100,
        step=1,
        help="Accumulated tool usage time.",
    )

    machine_type = st.selectbox(
        "Machine type",
        ["L", "M", "H"],
        help="Machine/product quality type.",
    )


# ============================================================
# 8. ENCODING
# ============================================================

TYPE_MAPPING = {
    "L": 0,
    "M": 1,
    "H": 2,
}

machine_type_encoded = TYPE_MAPPING[machine_type]


# ============================================================
# 9. CREATE MODEL INPUT
# ============================================================

input_data = pd.DataFrame(
    [
        {
            "Air temperature [K]": air_temp,
            "Process temperature [K]": process_temp,
            "Rotational speed [rpm]": rpm,
            "Torque [Nm]": torque,
            "Tool wear [min]": tool_wear,
            "Type": machine_type_encoded,
        }
    ]
)


# Force the same column order as training
input_data = input_data[EXPECTED_FEATURES]


# ============================================================
# 10. CHECK FEATURES
# ============================================================

if hasattr(model, "feature_names_in_"):

    trained_features = list(model.feature_names_in_)

    if trained_features != EXPECTED_FEATURES:

        st.error(
            "The application features do not match "
            "the features used during model training."
        )

        st.write("Model expects:")

        st.code(str(trained_features))

        st.write("Application provides:")

        st.code(str(EXPECTED_FEATURES))

        st.stop()


# ============================================================
# 11. PREDICTION FUNCTION
# ============================================================

def predict_failure(data):

    probabilities = model.predict_proba(data)[0]

    failure_probability = float(
        probabilities[failure_class_index]
    )

    prediction = int(
        model.predict(data)[0]
    )

    return prediction, failure_probability, probabilities


# ============================================================
# 12. PREDICT
# ============================================================

st.divider()


if st.button(
    "Analyze machine",
    type="primary",
    use_container_width=True,
):

    try:

        prediction, failure_probability, raw_probabilities = (
            predict_failure(input_data)
        )

    except Exception as e:

        st.error(
            "An error occurred while making the prediction."
        )

        st.exception(e)

        st.stop()


    # ========================================================
    # RESULTS
    # ========================================================

    st.subheader("Analysis result")


    metric1, metric2, metric3 = st.columns(3)


    with metric1:

        st.metric(
            "Failure probability",
            f"{failure_probability:.2%}",
        )


    with metric2:

        st.metric(
            "Normal probability",
            f"{1 - failure_probability:.2%}",
        )


    with metric3:

        if prediction == 1:
            st.metric(
                "Model classification",
                "Failure risk",
            )
        else:
            st.metric(
                "Model classification",
                "Normal",
            )


    # ========================================================
    # PROBABILITY BAR
    # ========================================================

    st.write("#### Estimated failure risk")

    st.progress(
        min(
            max(failure_probability, 0.0),
            1.0
        )
    )


    # ========================================================
    # STATUS
    # ========================================================

    if prediction == 1:

        st.error(
            "⚠️ The model detected an elevated risk of machine failure."
        )

        st.write(
            "The current operating conditions were classified "
            "as **machine failure (class 1)**."
        )

    else:

        st.success(
            "✓ The machine was classified as operating normally."
        )

        st.write(
            "The current operating conditions were classified "
            "as **normal operation (class 0)**."
        )


    # ========================================================
    # RISK INTERPRETATION
    # ========================================================

    if failure_probability < 0.20:

        risk_level = "Low"

    elif failure_probability < 0.50:

        risk_level = "Moderate"

    elif failure_probability < 0.75:

        risk_level = "High"

    else:

        risk_level = "Very high"


    st.info(
        f"Risk indicator: **{risk_level}**"
    )


    # ========================================================
    # DETAILS
    # ========================================================

    with st.expander(
        "View prediction details"
    ):

        st.write("##### Input sent to the model")

        display_data = input_data.copy()

        display_data["Type"] = machine_type

        st.dataframe(
            display_data,
            use_container_width=True,
            hide_index=True,
        )


        st.write("##### Raw model output")

        probability_table = pd.DataFrame(
            {
                "Class": model.classes_,
                "Probability": raw_probabilities,
            }
        )

        st.dataframe(
            probability_table,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# 13. FOOTER
# ============================================================

st.divider()

st.caption(
    "Machine Failure Prediction • Random Forest • "
    "Scikit-learn • Streamlit"
)