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
        
        # ============================================
        # FEATURE 1: Income
        # ============================================
        income = st.sidebar.number_input("💰 Annual Income ($)", min_value=0, max_value=500000, value=50000, step=1000)
        
        # ============================================
        # FEATURE 2: Age
        # ============================================
        age = st.sidebar.slider("🎂 Age", min_value=18, max_value=80, value=35)
        
        # ============================================
        # FEATURE 3: Experience
        # ============================================
        experience = st.sidebar.slider("💼 Years of Experience", min_value=0, max_value=50, value=5)
        
        # ============================================
        # FEATURE 4: Married/Single
        # ============================================
        married = st.sidebar.selectbox("💍 Marital Status", ["Single", "Married"])
        married_encoded = 1 if married == "Married" else 0
        
        # ============================================
        # FEATURE 5: House_Ownership
        # ============================================
        house = st.sidebar.selectbox("🏠 House Ownership", ["Rented", "Owned", "Mortgage"])
        house_map = {"Rented": 0, "Owned": 1, "Mortgage": 2}
        house_encoded = house_map[house]
        
        # ============================================
        # FEATURE 6: Car_Ownership
        # ============================================
        car = st.sidebar.selectbox("🚗 Car Ownership", ["No", "Yes"])
        car_encoded = 1 if car == "Yes" else 0
        
        # ============================================
        # FEATURE 7: Profession
        # ============================================
        profession = st.sidebar.selectbox(
            "👔 Profession",
            ["Salaried", "Self-Employed", "Business", "Student", "Unemployed"]
        )
        profession_map = {
            "Salaried": 0,
            "Self-Employed": 1,
            "Business": 2,
            "Student": 3,
            "Unemployed": 4
        }
        profession_encoded = profession_map[profession]
        
        # ============================================
        # FEATURE 8: CITY
        # ============================================
        city_options = [
            "Rewa", "Parbhani", "Alappuzha", "Bhubaneswar", "Tiruchirappalli",
            "Jalgaon", "Tiruppur", "Jamnagar", "Kota", "Karimnagar",
            "Hajipur", "Adoni", "Erode", "Kollam", "Madurai",
            "Anantapuram", "Kamarhati", "Bhusawal", "Sirsa", "Amaravati",
            "Secunderabad", "Ahmedabad", "Ajmer", "Ongole", "Miryalaguda",
            "Ambattur", "Indore", "Pondicherry", "Shimoga", "Chennai",
            "Gulbarga", "Khammam", "Saharanpur", "Gopalpur", "Amravati",
            "Udupi", "Howrah", "Aurangabad", "Hospet", "Shimla"
        ]
        city = st.sidebar.selectbox("🏙️ City", city_options)
        city_map = {city: i for i, city in enumerate(city_options)}
        city_encoded = city_map[city]
        
        # ============================================
        # FEATURE 9: STATE
        # ============================================
        state_options = [
            "Madhya_Pradesh", "Maharashtra", "Kerala", "Odisha", "Tamil_Nadu",
            "Gujarat", "Rajasthan", "Telangana", "Bihar", "Andhra_Pradesh",
            "West_Bengal", "Haryana", "Puducherry", "Karnataka", "Uttar_Pradesh",
            "Himachal_Pradesh", "Punjab", "Tripura", "Uttarakhand", "Jharkhand",
            "Mizoram", "Assam", "Jammu_and_Kashmir", "Delhi", "Chhattisgarh",
            "Chandigarh", "Manipur", "Sikkim"
        ]
        state = st.sidebar.selectbox("📍 State", state_options)
        state_map = {state: i for i, state in enumerate(state_options)}
        state_encoded = state_map[state]
        
        # ============================================
        # FEATURE 10: CURRENT_JOB_YRS
        # ============================================
        job_years = st.sidebar.slider("📅 Years at Current Job", min_value=0, max_value=30, value=3)
        
        # ============================================
        # FEATURE 11: CURRENT_HOUSE_YRS
        # ============================================
        house_years = st.sidebar.slider("🏡 Years in Current House", min_value=0, max_value=40, value=5)
        
        # ============================================
        # PREDICT BUTTON
        # ============================================
        if st.sidebar.button("🔮 Predict Default Risk", use_container_width=True):
            # Create array with ALL 11 features in the EXACT order
            features = np.array([[
                income,
                age,
                experience,
                married_encoded,
                house_encoded,
                car_encoded,
                profession_encoded,
                city_encoded,
                state_encoded,
                job_years,
                house_years
            ]])
            
            try:
                prediction = model.predict(features)
                probability = model.predict_proba(features)[0][1]
                
                # Display results
                st.subheader("📊 Prediction Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if prediction[0] == 1:
                        st.error("⚠️ HIGH RISK")
                        st.metric("Decision", "REJECT LOAN", delta="⚠️ High Risk")
                    else:
                        st.success("✅ LOW RISK")
                        st.metric("Decision", "APPROVE LOAN", delta="✅ Low Risk")
                
                with col2:
                    st.metric("Default Probability", f"{probability:.1%}")
                
                with col3:
                    if probability < 0.3:
                        st.success("🟢 Low Risk")
                        st.progress(probability)
                    elif probability < 0.6:
                        st.warning("🟡 Medium Risk")
                        st.progress(probability)
                    else:
                        st.error("🔴 High Risk")
                        st.progress(probability)
                        
            except Exception as e:
                st.error(f"❌ Prediction error: {str(e)}")
                st.info("💡 Make sure all inputs are filled correctly")
                
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.info("💡 Please make sure you're uploading a valid .pkl file")

else:
    st.info("📁 Upload your loan_model.pkl file to get started")
    
    st.markdown("""
    ### 📋 How to use this app:
    1. Upload your `loan_model.pkl` file
    2. Enter customer details in the sidebar
    3. Click 'Predict Default Risk'
    4. View the prediction results
    
    ### 📊 Features used:
    | # | Feature | Type |
    |---|---------|------|
    | 1 | Income | Numeric |
    | 2 | Age | Numeric |
    | 3 | Experience | Numeric |
    | 4 | Marital Status | Categorical |
    | 5 | House Ownership | Categorical |
    | 6 | Car Ownership | Categorical |
    | 7 | Profession | Categorical |
    | 8 | City | Categorical |
    | 9 | State | Categorical |
    | 10 | Current Job Years | Numeric |
    | 11 | Current House Years | Numeric |
    """)
