# 🧠 MRI Prediction System using Machine Learning

This project is an **end-to-end Machine Learning application** that predicts MRI-related outcomes using clinical input data.  
After evaluating multiple algorithms, **Gradient Boosting** was selected as the **best-performing model**.

The project demonstrates a complete ML workflow — from data preprocessing to model deployment with a Flask web interface.

---

## 📌 Project Overview

Medical data often contains missing values, outliers, and imbalanced classes.  
This project addresses these challenges and builds a reliable prediction system using supervised machine learning.

**Best Model Selected:** Gradient Boosting Classifier

---

## 🧠 Machine Learning Workflow

1. Data Loading  
2. Missing Value Handling  
3. Outlier Detection & Variable Transformation  
4. Train–Test Split  
5. Data Balancing (SMOTE when applicable)  
6. Feature Scaling  
7. Model Training  
8. Model Evaluation  
9. Best Model Selection  
10. Model Deployment using Flask

---

## ⚙️ Algorithms Used

The following models were trained and evaluated:

- Logistic Regression  
- Decision Tree Classifier  
- Random Forest Classifier  
- XGBoost Classifier  
- **Gradient Boosting Classifier (Selected Best Model)**

---

## 📊 Model Evaluation

Models were compared using the following metrics:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC–AUC Curve

The **Gradient Boosting model** achieved the highest overall performance and was selected for deployment.

---

## 🏆 Best Model: Gradient Boosting

**Why Gradient Boosting?**

- Handles non-linear relationships effectively  
- Robust to outliers  
- High predictive accuracy  
- Performs well on structured medical datasets  

---

## 🚀 Deployment

- Backend: Flask  
- Frontend: HTML, CSS, JavaScript  
- Model Serialization: Pickle  
- Scaler saved for inference consistency  

The trained Gradient Boosting model is loaded in the Flask app to provide real-time predictions.

---

## 🛠️ Technologies Used

- Python  
- Pandas, NumPy  
- Scikit-learn  
- Imbalanced-learn (SMOTE)  
- Flask  
- HTML, CSS, JavaScript
  
