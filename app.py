
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# --------------------------------------------------
# 1. Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Student Marks Prediction",
    page_icon="🎓",
    layout="centered"
)

# --------------------------------------------------
# 2. Load the Trained Model
# --------------------------------------------------
MODEL_PATH = Path(__file__).parent / "students_model.pkl"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except FileNotFoundError:
    st.error(
        "Model file not found! Please run model.py first "
        "to create students_model.pkl."
    )
    st.stop()
except Exception as error:
    st.error(f"Unable to load the model: {error}")
    st.stop()


# --------------------------------------------------
# 3. Application Heading
# --------------------------------------------------
st.title("🎓 Student Performance Prediction")

st.markdown(
    "### Machine Learning-Based Exam Score Prediction"
)

st.write(
    "Enter the student's academic details below "
    "to estimate their final examination marks."
)

st.divider()


# --------------------------------------------------
# 4. Student Input Form
# --------------------------------------------------
with st.form("student_prediction_form"):

    st.subheader("Student Academic Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        study_hours = st.number_input(
            "Daily Study Hours",
            min_value=0.0,
            max_value=24.0,
            value=6.0,
            step=0.5
        )

    with col2:
        attendance = st.number_input(
            "Attendance (%)",
            min_value=0.0,
            max_value=100.0,
            value=80.0,
            step=1.0
        )

    with col3:
        previous_marks = st.number_input(
            "Previous Exam Marks",
            min_value=0.0,
            max_value=100.0,
            value=65.0,
            step=1.0
        )

    submitted = st.form_submit_button(
        "Predict Final Marks",
        width="stretch"
    )


# --------------------------------------------------
# 5. Predict Final Marks
# --------------------------------------------------
if submitted:

    # Feature names and order must match model.py
    input_data = pd.DataFrame([{
        "study_hours": study_hours,
        "attendance": attendance,
        "previous_marks": previous_marks
    }])

    try:
        # Generate prediction
        prediction = float(model.predict(input_data)[0])

        # Keep displayed score within 0–100
        prediction = max(0.0, min(100.0, prediction))

        st.divider()
        st.subheader("Predicted Examination Result")

        # Display prediction and study hours
        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                label="Predicted Final Marks",
                value=f"{prediction:.2f} / 100"
            )

        with col2:
            st.metric(
                label="Daily Study Hours",
                value=f"{study_hours:.1f} hours"
            )

        # Visual score indicator
        st.write("Predicted Score")
        st.progress(int(round(prediction)))

        # Performance interpretation
        if prediction >= 75:
            st.success(
                "Strong predicted performance! "
                "Continue your consistent study habits."
            )

        elif prediction >= 50:
            st.info(
                "Moderate predicted performance. "
                "Regular revision and practice may help."
            )

        else:
            st.warning(
                "The model predicts a low score. "
                "Additional practice and revision may help."
            )

        # Display input details
        st.subheader("Student Input Summary")

        st.dataframe(
            input_data,
            width="stretch"
        )

        # Disclaimer
        st.caption(
            "This is an educational machine-learning estimate, "
            "not a guaranteed examination result. "
            "Prediction quality depends on the training data."
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")