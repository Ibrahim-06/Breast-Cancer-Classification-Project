# 🎗️ Breast Cancer Prediction

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Libraries](https://img.shields.io/badge/Libraries-pandas%2C%20numpy%2C%20matplotlib%2C%20seaborn%2C%20scikit--learn%2C%20xgboost%2C%20imbalanced--learn%2C%20joblib%2C%20streamlit-orange)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

## 📌 Overview
This project focuses on predicting **breast cancer diagnosis (benign vs malignant)** using various machine learning algorithms. 
The aim is to help in early detection by analyzing medical diagnostic data.

The dataset used is from **Kaggle Breast Cancer dataset** containing tumor characteristics.

---

## 📂 Project Structure
```
├── Data/
│   └── breast_cancer.csv
├── Model/
│   └── Logistic_model.pkl
├── Scaler/
│   └── scaler.pkl
├── Features/
│   └── features.pkl
├── App/
│   └── app.py         
├── Notebook/
│   └── breast_cancer_analysis.ipynb
├── README.md
└── Streamlit.png
```

---

## 🛠️ Features
- **Data Cleaning** (handling missing values, removing duplicates, converting data types)
- **Exploratory Data Analysis (EDA)** with Seaborn and Matplotlib
- **Feature Engineering** (Scaling, Label Encoding)
- **Model Training** with:
  - Logistic Regression (Best Model with Recall = **95%**)
  - Decision Tree
  - Random Forest
  - SVM
  - KNN
  - XGBoost
  - AdaBoost
- **Model Comparison** (Accuracy, Precision, Recall, F1-score)
- **Model Saving** using Joblib
- **Interactive Streamlit App** for prediction

---

## 📊 Data Preprocessing
- Converted categorical labels into numerical form
- Standardized numerical features using `StandardScaler`
- Split dataset into training/testing

---

## 📈 Model Evaluation
Metrics used:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

✅ Best Model: **Logistic Regression with Recall = 95%**

---

## 🚀 Installation & Usage
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/ibrahim-06/Breast-Cancer-Classification-Project.git
cd Breast-Cancer-Classification-Project
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run Jupyter Notebook (for training and analysis)
```bash
jupyter notebook Notebook/Breast_Cancer.ipynb
```

### 4️⃣ Run Streamlit App
```bash
streamlit run App/app.py
```

---

## 📹 Streamlit (UI Demo)
<img width="3812" height="2128" alt="Streamlit" src="https://github.com/user-attachments/assets/c39d383b-98c5-4a85-bea4-5efaf73d84b0" />


---

## 📊 Results Snapshot
| Model                 | Accuracy | Precision | Recall | F1-score |
|-----------------------|----------|-----------|--------|----------|
| Logistic Regression   | 0.92     | 0.90      | 0.95   | 0.92     |
| Random Forest         | 0.94     | 0.91      | 0.92   | 0.91     |
| XGBoost               | 0.95     | 0.92      | 0.93   | 0.93     |
| AdaBoost              | 0.93     | 0.90      | 0.91   | 0.91     |

---

## 📦 Saved Artifacts
- **Logistic_model.pkl** → Trained Logistic Regression model
- **scaler.pkl** → StandardScaler fitted on training data
- **features.pkl** → List of feature names after preprocessing

---

## 📚 Requirements
```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
imbalanced-learn
joblib
streamlit
```

---

## ✨ Project by: **Ibrahim Mohamed**  

📅 **Year:** 2025  
📍 **Field:** Machine Learning & Data Science  

> 💡 *"Turning data into decisions — one model at a time."*
