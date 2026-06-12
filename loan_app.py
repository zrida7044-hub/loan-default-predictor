import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("loan_model.pkl", "rb"))

st.title("Loan Default Risk Prediction App")

st.write("Enter applicant details below")

age = st.number_input("Age", 18, 100, 30)
income = st.number_input("Income", 0)
loan_amount = st.number_input("Loan Amount", 0)
credit_score = st.number_input("Credit Score", 300, 900)
employment_years = st.number_input("Employment Years", 0)

if st.button("Predict"):
    input_data = np.array([[age, income, loan_amount, credit_score, employment_years]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("High Risk: May default")
    else:
        st.success("Low Risk: Likely to repay")
