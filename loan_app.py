import streamlit as st
import pickle
import numpy as np

st.set_page_config(page_title="Loan Default Predictor", layout="wide")

st.title("🏦 Alpha Dreamers Banking Consortium")
st.subheader("Personal Loan Default Risk Assessment")

uploaded_model = st.file_uploader("Upload your loan_model.pkl file", type=["pkl"])

if uploaded_model is not None:
    try:
        model = pickle.load(uploaded_model)
        st.success("✅ Model loaded successfully!")
        
        st.sidebar.header("Enter Customer Details")
        
        # === REPLACE THESE WITH YOUR ACTUAL COLUMN NAMES ===
        # Get from the code above
        age = st.sidebar.slider("Age", 18, 80, 35)
        income = st.sidebar.number_input("Annual Income", 0, 500000, 50000)
        loan_amount = st.sidebar.number_input("Loan Amount", 0, 200000, 20000)
        credit_score = st.sidebar.slider("Credit Score", 300, 850, 680)
        employment_years = st.sidebar.slider("Years Employed", 0, 40, 5)
        # ... add 10 more features here
        # Total must be 15
        
        # Create array with ALL features in the correct order
        features = np.array([[
            age,
            income,
            loan_amount,
            credit_score,
            employment_years,
            # ... add all other values in the same order as your CSV
        ]])
        
        if st.sidebar.button("Predict Default Risk"):
            prediction = model.predict(features)
            probability = model.predict_proba(features)[0][1]
            
            col1, col2 = st.columns(2)
            with col1:
                if prediction[0] == 1:
                    st.error("⚠️ HIGH RISK")
                else:
                    st.success("✅ LOW RISK")
            with col2:
                st.metric("Default Probability", f"{probability:.1%}")
                
    except Exception as e:
        st.error(f"Error: {str(e)}")
