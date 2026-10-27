import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Diabetes Risk Prediction",
    page_icon="🩺",
    layout="wide"
)

# Load the trained model and feature names
model = joblib.load("model.joblib")
feature_names = joblib.load("feature_names.joblib")

st.title("🩺 Diabetes Risk Prediction Application")
st.write(
    "Enter the patient's age, gender, and symptoms "
    "to predict the diabetes risk."
)

# Sidebar
st.sidebar.header("Patient Details")

age = st.sidebar.number_input(
    "Age",
    min_value=1,
    max_value=120,
    value=30
)

gender = st.sidebar.selectbox(
    "Gender",
    ["Male", "Female"]
)

st.sidebar.header("Symptoms")

polyuria = st.sidebar.selectbox(
    "Polyuria (Frequent Urination)",
    ["No", "Yes"]
)

polydipsia = st.sidebar.selectbox(
    "Polydipsia (Excessive Thirst)",
    ["No", "Yes"]
)

sudden_weight_loss = st.sidebar.selectbox(
    "Sudden Weight Loss",
    ["No", "Yes"]
)

weakness = st.sidebar.selectbox(
    "Weakness",
    ["No", "Yes"]
)

polyphagia = st.sidebar.selectbox(
    "Polyphagia (Excessive Hunger)",
    ["No", "Yes"]
)

genital_thrush = st.sidebar.selectbox(
    "Genital Thrush",
    ["No", "Yes"]
)

visual_blurring = st.sidebar.selectbox(
    "Visual Blurring",
    ["No", "Yes"]
)

itching = st.sidebar.selectbox(
    "Itching",
    ["No", "Yes"]
)

irritability = st.sidebar.selectbox(
    "Irritability",
    ["No", "Yes"]
)

delayed_healing = st.sidebar.selectbox(
    "Delayed Healing",
    ["No", "Yes"]
)

partial_paresis = st.sidebar.selectbox(
    "Partial Paresis",
    ["No", "Yes"]
)

muscle_stiffness = st.sidebar.selectbox(
    "Muscle Stiffness",
    ["No", "Yes"]
)

alopecia = st.sidebar.selectbox(
    "Alopecia (Hair Loss)",
    ["No", "Yes"]
)

obesity = st.sidebar.selectbox(
    "Obesity",
    ["No", "Yes"]
)

# Collect inputs using the dataset's exact column names
input_data = {
    "Age": age,
    "Gender": gender,
    "Polyuria": polyuria,
    "Polydipsia": polydipsia,
    "sudden weight loss": sudden_weight_loss,
    "weakness": weakness,
    "Polyphagia": polyphagia,
    "Genital thrush": genital_thrush,
    "visual blurring": visual_blurring,
    "Itching": itching,
    "Irritability": irritability,
    "delayed healing": delayed_healing,
    "partial paresis": partial_paresis,
    "muscle stiffness": muscle_stiffness,
    "Alopecia": alopecia,
    "Obesity": obesity
}

input_df = pd.DataFrame([input_data])

# Arrange columns in the same order used during training
input_df = input_df.reindex(
    columns=feature_names
)

# Display patient details
st.subheader("📋 Patient Details")
st.dataframe(input_df, use_container_width=True)

# Predict only when the button is clicked
if st.button("Predict Diabetes Risk", type="primary"):

    try:
        prediction = model.predict(input_df)[0]

        st.subheader("🔍 Prediction Result")

        if str(prediction).strip().lower() in [
            "positive", "1", "yes"
        ]:
            st.error("Prediction: Positive")
            st.write(
                "The model predicts a positive diabetes "
                "classification. Please consult a qualified "
                "healthcare professional for evaluation."
            )

        else:
            st.success("Prediction: Negative")
            st.write(
                "The model predicts a negative diabetes "
                "classification. This does not rule out diabetes."
            )

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.info(
            "Check that your model uses the same feature "
            "names, column order, and input preprocessing."
        )

st.caption(
    "This application is for educational purposes only "
    "and is not a substitute for medical diagnosis."
)