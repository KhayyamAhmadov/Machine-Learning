# bmw_streamlit_app.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np
from PIL import Image
import sklearn.compose._column_transformer
import types

# Fix for sklearn compatibility issue
if not hasattr(sklearn.compose._column_transformer, '_RemainderColsList'):
    class _RemainderColsList(list):
        """
        Remainder column list that will be passed to the transformer.
        """
        pass
    
    sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList

# App configuration
st.set_page_config(
    page_title="BMW Sales Classification",
    page_icon="🚗",
    layout="wide"
)

# Load model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('bmw_class.pkl')
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Feature columns (same as your training)
feature_columns = ['Model', 'Region', 'Color', 'Fuel_Type', 'Transmission', 
                   'Engine_Size_L', 'Mileage_KM', 'Price_USD', 'Sales_Volume', 
                   'Car_Age', 'Price_Per_KM', 'Engine_Power_Ratio']

# Header
st.title("🚗 BMW Sales Classification Predictor")
st.markdown("""
This application predicts the sales potential of BMW cars based on their features.
Enter the car details below to predict whether it has high or low sales potential.
""")

# Load model
model_pipeline = load_model()

if model_pipeline is not None:
    # Sidebar - Data input
    with st.sidebar:
        st.header("📋 Car Details")
        
        # Basic car information with correct categories
        model = st.selectbox("Model", [
            '5 Series', 'i8', 'X3', '7 Series', 'M5', '3 Series', 
            'X1', 'M3', 'X5', 'i3', 'X6'
        ])
        
        region = st.selectbox("Region", [
            'Asia', 'North America', 'Middle East', 'South America', 'Europe', 'Africa'
        ])
        
        color = st.selectbox("Color", [
            'Red', 'Blue', 'Black', 'Silver', 'White', 'Grey'
        ])
        
        fuel_type = st.selectbox("Fuel Type", [
            'Petrol', 'Hybrid', 'Diesel', 'Electric'
        ])
        
        transmission = st.selectbox("Transmission", [
            'Manual', 'Automatic'
        ])
        
        # Technical specifications
        st.subheader("📊 Technical Specifications")
        engine_size = st.slider("Engine Size (L)", 1.0, 5.0, 2.0, 0.1)
        mileage = st.slider("Mileage (KM)", 0, 200000, 50000, 1000)
        price = st.slider("Price (USD)", 10000, 150000, 45000, 1000)
        sales_volume = st.slider("Sales Volume", 0, 200, 100, 5)
        year = st.slider("Manufacturing Year", 2000, 2025, 2020, 1)
        
        # Calculate derived features
        current_year = 2025
        car_age = current_year - year
        price_per_km = price / (mileage + 1)
        engine_power_ratio = engine_size * sales_volume

    # Display input data
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📋 Input Data")
        data = {
            "Feature": ["Model", "Region", "Color", "Fuel Type", "Transmission", 
                      "Engine Size (L)", "Mileage (KM)", "Price (USD)", "Sales Volume",
                      "Year", "Car Age", "Price Per KM", "Engine Power Ratio"],
            "Value": [model, region, color, fuel_type, transmission, 
                     f"{engine_size}L", f"{mileage:,} KM", f"{price:,} USD", sales_volume,
                     year, car_age, f"${price_per_km:.2f}", f"{engine_power_ratio:.1f}"]
        }
        df_display = pd.DataFrame(data)
        st.table(df_display)

    # Prediction button
    if st.button("🎯 Predict Sales Potential", type="primary"):
        # Prepare input data
        input_data = pd.DataFrame({
            'Model': [model],
            'Region': [region],
            'Color': [color],
            'Fuel_Type': [fuel_type],
            'Transmission': [transmission],
            'Engine_Size_L': [engine_size],
            'Mileage_KM': [mileage],
            'Price_USD': [price],
            'Sales_Volume': [sales_volume],
            'Car_Age': [car_age],
            'Price_Per_KM': [price_per_km],
            'Engine_Power_Ratio': [engine_power_ratio]
        })
        
        input_data = input_data[feature_columns]
        
        # Make prediction
        try:
            prediction = model_pipeline.predict(input_data)
            prediction_proba = model_pipeline.predict_proba(input_data)
            
            # Get the class labels from the model
            classes = model_pipeline.named_steps['classifier'].classes_
            
            # Create a mapping between class labels and probabilities
            prob_mapping = dict(zip(classes, prediction_proba[0]))
            
            with col2:
                st.subheader("📊 Prediction Results")
                
                # Prediction result
                result = "High Sales Potential" if prediction[0] == 'High' else "Low Sales Potential"
                result_color = "green" if prediction[0] == 'High' else "red"
                
                st.markdown(f"<h2 style='text-align: center; color: {result_color};'>{result}</h2>", 
                           unsafe_allow_html=True)
                
                # Display probabilities correctly
                if 'High' in prob_mapping and 'Low' in prob_mapping:
                    high_prob = prob_mapping['High']
                    low_prob = prob_mapping['Low']
                else:
                    # Fallback if class names are different
                    # Assuming the first class is 'Low' and second is 'High'
                    low_prob = prediction_proba[0][0]
                    high_prob = prediction_proba[0][1] if len(prediction_proba[0]) > 1 else 0
                
                # Display metrics
                col_high, col_low = st.columns(2)
                with col_high:
                    st.metric("High Sales Probability", f"{high_prob:.2%}")
                with col_low:
                    st.metric("Low Sales Probability", f"{low_prob:.2%}")
                
                # Visual indicator for the predicted class
                if prediction[0] == 'High':
                    st.progress(high_prob, text="High Sales Confidence")
                else:
                    st.progress(low_prob, text="Low Sales Confidence")
                
                # Analysis and recommendations
                st.subheader("💡 Analysis & Recommendations")
                
                if prediction[0] == 'High':
                    st.success("""
                    🎉 This car has high sales potential!
                    **Recommendations:**
                    - Strengthen advertising campaigns
                    - Offer customer test drives
                    - Provide trial packages
                    - Highlight premium features in marketing materials
                    """)
                else:
                    st.warning("""
                    ⚠️ This car has low sales potential.
                    **Suggestions:**
                    - Review pricing strategy
                    - Offer additional services to customers
                    - Expand target audience
                    - Consider promotional discounts
                    """)
                    
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            st.error("Please check if the model was trained with 'High' and 'Low' classes.")

    # Information section
    with st.expander("ℹ️ About This Application"):
        st.markdown("""
        **About this application:**
        
        This application uses machine learning to predict BMW car sales classification based on 
        vehicle features. The model uses the following algorithms:
        
        - Random Forest
        - Gradient Boosting
        - Logistic Regression
        - Support Vector Machines (SVM)
        
        **Features used:**
        - Model, Region, Color, Fuel Type, Transmission
        - Engine Size, Mileage, Price, Sales Volume
        - Car Age, Price Per KM, Engine Power Ratio
        
        **Prediction results:** Cars are classified as having either 'High' or 'Low' sales potential.
        
        **Note:** The probabilities shown correspond to the confidence level for each prediction.
        """)

    # Footer
    st.markdown("---")
    st.caption("© 2025 BMW Sales Classification Predictor | Machine Learning Model")
else:
    st.error("Model could not be loaded. Please ensure the model file exists.")