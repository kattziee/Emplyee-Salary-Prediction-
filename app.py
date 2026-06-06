import streamlit as st
import pandas as pd
import joblib

model = joblib.load(
    "models/salary_model.pkl"
)

st.title(
    "Employee Salary Prediction"
)

age = st.number_input(
    "Age",
    18,
    65,
    25
)

gender = st.selectbox(
    "Gender",
    [
        "Male",
        "Female"
    ]
)

education = st.selectbox(
    "Education",
    [
        "Bachelor's",
        "Master's",
        "PhD"
    ]
)

job = st.text_input(
    "Job Title"
)

experience = st.number_input(
    "Years of Experience",
    0,
    40,
    1
)

if st.button("Predict Salary"):

    data = pd.DataFrame(
        {
            "Age":[age],
            "Gender":[gender],
            "Education Level":[education],
            "Job Title":[job],
            "Years of Experience":[experience]
        }
    )

    prediction = model.predict(data)

    st.success(
        f"Predicted Salary: ₹ {prediction[0]:,.2f}"
    )