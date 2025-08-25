# 🍷 Wine Quality Prediction

**Wine Quality Prediction** project is designed to predict the quality of wine based on various physicochemical properties. This project leverages **Machine Learning** techniques to provide practical insights for winemakers, sellers, and wine enthusiasts.

---

## 🎯 Project Goal
- Predict the quality of wine samples automatically.
- Compare different machine learning models for prediction.
- Identify the best-performing model for wine quality prediction.
- Provide analytical and visual insights to support decision-making.

---

## 📊 Dataset
- Source: [📂 Red Wine Quality Dataset on Kaggle](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009)
- Features include:
  - **Fixed acidity**, **Volatile acidity**, **Citric acid**, **Residual sugar**
  - **Chlorides**, **Free sulfur dioxide**, **Total sulfur dioxide**
  - **Density**, **pH**, **Sulphates**, **Alcohol**
- Target: `quality` 

---

## ⚙️ Methodology
1. **Data Preprocessing**  
   - Cleaning and checking the dataset
   - Scaling and normalization of features
2. **Exploratory Data Analysis (EDA)**  
   - Distribution of target variable
   - Correlation heatmap and feature visualizations
3. **Feature Selection**  
   - Identifying important features for better prediction
4. **Model Training & Evaluation**  
   - Testing multiple machine learning algorithms
   - Cross-validation and evaluation on a test set
   - Metrics: `accuracy`, `f1-score`, `confusion matrix`
5. **Model Deployment**  
   - Saving the trained model for future predictions
   - Optional integration into applications or APIs

---

## 🏆 Results
- The model provides accurate predictions of wine quality.
- High performance on both training and test datasets.
- Predictions are close to actual quality scores, ensuring reliability.

---

## 💡 Benefits
- Quality control support for winemakers.
- Product assessment for sellers.
- Informed wine selection for enthusiasts.
- Practical machine learning example for learners.

---

## 📂 Future Improvements
- Train on larger and more diverse datasets.
- Optimize models with hyperparameter tuning.

---

## 🛠️ Tech Stack
- **Python** – Core language  
- **Scikit-learn** – Machine learning models  
- **Streamlit** – Web application framework  
- **Pandas / NumPy** – Data manipulation and numerical computing  
- **Pickle** – Model saving/loading  
- **Matplotlib & Seaborn** – Visualization  
