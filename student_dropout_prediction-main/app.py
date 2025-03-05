import streamlit as st
import pickle
import pandas as pd

# Load the trained model and label encoder
with open("logistic_regression_pipeline.pkl", "rb") as model_file:
    loaded_pipeline = pickle.load(model_file)

with open("label_encoder.pkl", "rb") as le_file:
    label_encoder = pickle.load(le_file)

# Streamlit UI
st.title("Student Dropout Risk Predictor")
st.write("Enter student details to predict dropout risk.")

# User Input Form
education_board = st.selectbox("Education Board", ["CBSE", "ICSE", "State Board"])
gender = st.selectbox("Gender", ["Male", "Female"])
family_structure = st.selectbox("Family Structure", ["Nuclear", "Joint", "Single Parent"])
parent_education = st.selectbox("Parent Education", ["Graduate", "Postgraduate", "High School"])
residential_area = st.selectbox("Residential Area", ["Urban", "Rural"])
academic_trend = st.selectbox("Academic Trend", ["Good", "Average", "Poor"])
access_to_tutoring = st.selectbox("Access to Tutoring", ["Yes", "No"])
monthly_absences = st.number_input("Monthly Absences", min_value=0, max_value=30, value=2)
last_year_marks = st.number_input("Last Year Marks (%)", min_value=0, max_value=100, value=75)
family_income = st.number_input("Family Income (INR)", min_value=0, step=1000, value=50000)

# Predict button
if st.button("Predict Dropout Risk"):
    try:
        # Convert input into DataFrame
        input_data = pd.DataFrame([{ 
            "education_board": education_board,
            "gender": gender,
            "family_structure": family_structure,
            "parent_education": parent_education,
            "residential_area": residential_area,
            "academic_trend": academic_trend,
            "access_to_tutoring": access_to_tutoring,
            "monthly_absences": monthly_absences,
            "last_year_marks": last_year_marks,
            "family_income": family_income
        }])
        
        # Ensure columns match the trained model
        expected_features = loaded_pipeline["preprocessor"].feature_names_in_
        input_data = input_data[expected_features]
        
        # Make prediction
        prediction_encoded = loaded_pipeline.predict(input_data)
        prediction = label_encoder.inverse_transform(prediction_encoded)[0]
        
        st.success(f"Predicted Dropout Risk: {prediction}")
    except Exception as e:
        st.error(f"Error: {str(e)}")