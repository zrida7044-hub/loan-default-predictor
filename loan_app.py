import streamlit as st
import pickle
import numpy as np
import pandas as pd

st.set_page_config(page_title="Loan Default Predictor", layout="wide")

st.title("🏦 Alpha Dreamers Banking Consortium")
st.subheader("Personal Loan Default Risk Assessment")

# Let user upload the model
uploaded_model = st.file_uploader("Upload your loan_model.pkl file", type=["pkl"])

if uploaded_model is not None:
    try:
        model = pickle.load(uploaded_model)
        st.success("✅ Model loaded successfully!")
        
        # Show how many features the model expects
        n_features = model.n_features_in_
        st.info(f"📊 This model requires {n_features} customer details")
        
        st.sidebar.header("Enter All Customer Details")
        
        # Create 15 input fields (all numbers for simplicity)
        input_values = []
        for i in range(n_features):
            val = st.sidebar.number_input(
                f"Feature {i+1}", 
                value=0.0,
                step=1.0,
                key=f"feature_{i}"
            )
            input_values.append(val)
        
        if st.sidebar.button("Predict Default Risk"):
            # Convert to numpy array
            features = np.array([input_values])
            
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
                    
                    # Risk level indicator
                    if probability < 0.3:
                        st.success("🟢 Low Risk")
                    elif probability < 0.6:
                        st.warning("🟡 Medium Risk")
                    else:
                        st.error("🔴 High Risk")
                        
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
                st.info(f"Make sure you entered exactly {n_features} values")
                
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Please upload a valid .pkl model file")

else:
    st.info("📁 Upload your loan_model.pkl file to get started")
    
    st.markdown("""
    ### How to use:
    1. Upload your `loan_model.pkl` file above
    2. Enter all customer details in the sidebar
    3. Click 'Predict Default Risk'
    4. View the prediction results
    """)
