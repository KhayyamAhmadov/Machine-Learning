# BMW Sales Classification Predictor 🚗

## 📖 Overview
The **BMW Sales Classification Predictor** is a machine learning-based web application that predicts the sales potential of BMW vehicles as **High** or **Low**.  
It helps dealerships and sales teams quickly assess market performance of different BMW models across regions.

---

## ✨ Features
- 🔹 **Predictive Analysis**: Classifies sales potential (High / Low)  
- 🔹 **Interactive Interface**: Streamlit-powered user-friendly input forms  
- 🔹 **Comprehensive Feature Set**: Model, region, color, fuel type, transmission, engine size, mileage, price, sales volume  
- 🔹 **Detailed Results**: Probability scores and actionable recommendations  
- 🔹 **Data Visualization**: Organized display of inputs & prediction results  

---

## 🛠️ Tech Stack
- **Python** – Core language  
- **Scikit-learn** – Machine learning models  
- **Streamlit** – Web application framework  
- **Pandas / NumPy** – Data manipulation and numerical computing  
- **Joblib** – Model saving/loading  
- **Matplotlib & Seaborn** – Visualization  

---

## 📊 Model
The predictor uses an **ensemble approach** combining:  
- Random Forest  
- Gradient Boosting  
- Logistic Regression  
- Support Vector Machines (SVM)  

### Features Considered:
- **Model**: (5 Series, i8, X3, 7 Series, M5, 3 Series, X1, M3, X5, i3, X6)  
- **Region**: (Asia, North America, Middle East, South America, Europe, Africa)  
- **Color**: (Red, Blue, Black, Silver, White, Grey)  
- **Fuel Type**: (Petrol, Hybrid, Diesel, Electric)  
- **Transmission**: (Manual, Automatic)  
- **Engine Size (L)**  
- **Mileage (KM)**  
- **Price (USD)**  
- **Sales Volume**  
- **Derived Features**: Car Age, Price per KM, Engine Power Ratio  

---
## 📊 Dataset

The dataset used in this project can be found here:  
[📂 BMW Car Sales Dataset on Kaggle](https://www.kaggle.com/datasets/sumedh1507/bmw-car-sales-dataset)
