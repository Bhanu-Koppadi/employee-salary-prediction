# 💼 Employee Salary Prediction using Machine Learning

This project is a Machine Learning-powered Streamlit web application that predicts whether an employee earns more than £50,000 annually using classification models. It leverages demographic and work-related attributes for training and includes interactive visualizations.
---

## 🚀 Live Demo

🔗 **Streamlit App**: [Click here to try the app](https://employee-salary-prediction-b3rdmsdcfi3pyxvogexhka.streamlit.app/)  
📂 **GitHub Repository**: [GitHub Repo Link](https://github.com/Bhanu-Koppadi/employee-salary-prediction.git)

---

## 📌 Project Overview

This project leverages various machine learning algorithms to analyze employee attributes and predict their salary category (`<=£50K` or `>£50K`).  
The app provides:

- 🔮 **Single Prediction**
- 📊 **Batch Prediction via CSV**
- 📈 **Model Comparison Dashboard**
- 🎯 **Feature Importance Analysis**

---

## 🧠 ML Models Used

| Model                  | Accuracy |
|------------------------|----------|
| ✅ Gradient Boosting   | 86.5%    |
| Random Forest          | 84.1%    |
| SVM                    | 81.7%    |
| Logistic Regression    | 79.3%    |
| K-Nearest Neighbors    | 76.8%    |

🔍 **Gradient Boosting** was selected as the best-performing model for final deployment.

---

## 📊 Dataset

- Source: [UCI Adult Income Dataset](https://archive.ics.uci.edu/ml/datasets/adult)
- Records: 48,842 entries
- Features:
  - Age, Workclass, Education, Marital Status, Occupation, Relationship
  - Race, Sex, Hours per Week, Native Country
  - Capital Gain/Loss, etc.
- Target: `Salary` — `>50K` or `<=50K`

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit (Python)
- **Backend / ML**: Scikit-learn, Pandas, NumPy, Joblib
- **Visualization**: Plotly, Matplotlib
- **Deployment**: [Streamlit Cloud](https://streamlit.io/cloud)

---



