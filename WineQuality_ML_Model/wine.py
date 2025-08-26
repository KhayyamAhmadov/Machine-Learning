import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Modeli yüklə
@st.cache_resource
def load_model():
    try:
        with open('wine_quality_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except FileNotFoundError:
        st.error("Model faylı tapılmadı. Əvvəlcə modeli öyrədin və yadda saxlayın.")
        return None

# Əsas tətbiq
def main():
    st.title("🍷 Şərab Keyfiyyəti Proqnoz Modeli")
    st.markdown("---")
    
    # Modeli yüklə
    model_data = load_model()
    
    if model_data is None:
        return
    
    model = model_data['model']
    scaler = model_data['scaler']
    feature_names = model_data['feature_names']
    
    # İstifadəçi interfeysi
    st.sidebar.header("Şərab Xüsusiyyətlərini Daxil Edin")
    
    # Xüsusiyyət inputları
    inputs = {}
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        inputs['fixed acidity'] = st.number_input("Fixed Acidity", min_value=4.0, max_value=16.0, value=7.0, step=0.1)
        inputs['volatile acidity'] = st.number_input("Volatile Acidity", min_value=0.1, max_value=2.0, value=0.5, step=0.01)
        inputs['citric acid'] = st.number_input("Citric Acid", min_value=0.0, max_value=1.0, value=0.25, step=0.01)
        inputs['residual sugar'] = st.number_input("Residual Sugar", min_value=0.5, max_value=16.0, value=2.0, step=0.1)
        inputs['chlorides'] = st.number_input("Chlorides", min_value=0.01, max_value=0.2, value=0.08, step=0.01)
        inputs['free sulfur dioxide'] = st.number_input("Free Sulfur Dioxide", min_value=1.0, max_value=100.0, value=30.0, step=1.0)
    
    with col2:
        inputs['total sulfur dioxide'] = st.number_input("Total Sulfur Dioxide", min_value=5.0, max_value=300.0, value=100.0, step=5.0)
        inputs['density'] = st.number_input("Density", min_value=0.98, max_value=1.04, value=0.996, step=0.001, format="%.3f")
        inputs['pH'] = st.number_input("pH", min_value=2.5, max_value=4.5, value=3.2, step=0.1)
        inputs['sulphates'] = st.number_input("Sulphates", min_value=0.3, max_value=2.0, value=0.6, step=0.1)
        inputs['alcohol'] = st.number_input("Alcohol", min_value=8.0, max_value=15.0, value=10.5, step=0.1)
    
    # Proqnoz et
    if st.sidebar.button("Keyfiyyəti Proqnoz Et"):
        # Xüsusiyyətləri sırala (modelin gözlədiyi kimi)
        input_values = [inputs[feature] for feature in feature_names]
        
        # Miqyasla
        input_scaled = scaler.transform([input_values])
        
        # Proqnoz et
        prediction = model.predict(input_scaled)[0]
        
        # Ehtimalları hesabla (əgər mövcuddursa)
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(input_scaled)[0]
            confidence = np.max(probabilities)
        
        # Nəticələri göstər
        st.success(f"**Proqnoz Edilən Keyfiyyət:** {prediction}")
        
        if hasattr(model, 'predict_proba'):
            st.info(f"**Etibar Səviyyəsi:** {confidence:.1%}")
        
        # Keyfiyyət şərhi
        if prediction <= 4:
            st.warning("Aşağı keyfiyyətli şərab")
        elif prediction == 5 or prediction == 6:
            st.info("Orta keyfiyyətli şərab")
        else:
            st.success("Yüksək keyfiyyətli şərab")
    
    # Model məlumatları
    st.markdown("---")
    st.subheader("Model Məlumatları")
    st.write(f"**Model növü:** {model_data['model_name']}")
    st.write(f"**Test dəqiqliyi:** {model_data['accuracy']:.2%}")
    
    # Xüsusiyyət əhəmiyyətləri
    if hasattr(model, 'feature_importances_'):
        st.subheader("Ən Əhəmiyyətli Xüsusiyyətlər")
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:5]  # Ən əhəmiyyətli 5 xüsusiyyət
        
        for i, idx in enumerate(indices, 1):
            st.write(f"{i}. {feature_names[idx]} ({importances[idx]:.3f})")
    
    # Nümunə verilənlər
    st.markdown("---")
    st.subheader("Nümunə Dəyərlər")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Aşağı keyfiyyət üçün:**")
        st.code("""
Fixed Acidity: 8.5
Volatile Acidity: 1.2
Citric Acid: 0.02
Residual Sugar: 3.5
Chlorides: 0.12
Free Sulfur Dioxide: 35.0
Total Sulfur Dioxide: 120.0
Density: 0.998
pH: 3.1
Sulphates: 0.45
Alcohol: 8.5
        """)
    
    with col2:
        st.write("**Yüksək keyfiyyət üçün:**")
        st.code("""
Fixed Acidity: 5.5
Volatile Acidity: 0.3
Citric Acid: 0.35
Residual Sugar: 1.2
Chlorides: 0.045
Free Sulfur Dioxide: 15.0
Total Sulfur Dioxide: 40.0
Density: 0.992
pH: 3.25
Sulphates: 0.75
Alcohol: 12.8
        """)

if __name__ == "__main__":
    main()