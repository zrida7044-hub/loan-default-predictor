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
        st.success(f"✅ Model loaded successfully! Expects {model.n_features_in_} features")
        
        st.sidebar.header("Enter Customer Details")
        st.sidebar.warning(f"⚠️ Please fill in all {model.n_features_in_} fields")
        
        # Create 15 input fields (numbered for clarity)
        input_values = []
        for i in range(model.n_features_in_):
            val = st.sidebar.number_input(
                f"Feature {i+1}", 
                value=0.0,
                step=1.0,
                key=f"feature_{i}"
            )
            input_values.append(val)
        
        if st.sidebar.button("Predict Default Risk"):
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
                    
                    if probability < 0.3:
                        st.success("🟢 Low Risk")
                    elif probability < 0.6:
                        st.warning("🟡 Medium Risk")
                    else:
                        st.error("🔴 High Risk")
                        
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
                st.info(f"Model expects {model.n_features_in_} features")
                
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")

else:
    st.info("📁 Upload your loan_model.pkl file to get started")
    
    st.markdown("""
    ### How to use:
    1. Upload your `loan_model.pkl` file
    2. Enter all 15 features in the sidebar
    3. Click 'Predict Default Risk'
    4. View the prediction results
    """)
