import streamlit as st
import pickle
import numpy as np

st.set_page_config(page_title="Loan Default Predictor", layout="wide")

st.title("🏦 Alpha Dreamers Banking Consortium")
st.subheader("Personal Loan Default Risk Assessment")

st.info("📁 Please upload your loan_model.pkl file to get started")

# Let user upload the model file
uploaded_model = st.file_uploader("Choose loan_model.pkl file", type=["pkl"])

if uploaded_model is not None:
    try:
        # Load the model from the uploaded file
        model = pickle.load(uploaded_model)
        st.success("✅ Model loaded successfully!")
        
        # Sidebar inputs
        st.sidebar.header("Customer Information")
        age = st.sidebar.slider("Age", 18, 80, 35)
        income = st.sidebar.number_input("Annual Income ($)", 0, 500000, 50000)
        loan_amount = st.sidebar.number_input("Loan Amount ($)", 0, 200000, 20000)
        credit_score = st.sidebar.slider("Credit Score", 300, 850, 680)
        
        if st.sidebar.button("Predict Default Risk"):
            # Create feature array
            features = np.array([[age, income, loan_amount, credit_score]])
            
            try:
                prediction = model.predict(features)
                probability = model.predict_proba(features)[0][1]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if prediction[0] == 1:
                        st.error("⚠️ HIGH RISK: Customer likely to default")
                        st.metric("Decision", "REJECT LOAN")
                    else:
                        st.success("✅ LOW RISK: Customer likely to repay")
                        st.metric("Decision", "APPROVE LOAN")
                
                with col2:
                    st.metric("Default Probability", f"{probability:.1%}")
                    if probability < 0.3:
                        st.info("🟢 Low Risk")
                    elif probability < 0.6:
                        st.warning("🟡 Medium Risk")
                    else:
                        st.error("🔴 High Risk")
                        
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
                st.info("Make sure the input features match your model's training data")
                
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Please make sure you're uploading a valid .pkl file")

else:
    st.info("👆 Upload your loan_model.pkl file above to start predicting")
    
    # Show file info
    st.markdown("""
    ### How to use this app:
    1. Upload your `loan_model.pkl` file
    2. Enter customer details in the sidebar
    3. Click 'Predict Default Risk'
    4. View the results
    """)
