# 🏥 HealthPredict: Diabetes Risk Analysis Dashboard

A professional-grade Machine Learning web application that predicts diabetes risk using clinical patient data. Built with **Python**, **Streamlit**, and **Scikit-Learn**.

![Streamlit Version](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=for-the-badge&logo=Streamlit)
![ML Model](https://img.shields.io/badge/Model-Random_Forest-blue?style=for-the-badge)

## 🌟 Overview
This tool serves as a clinical decision support system. It uses a **Random Forest Classifier** trained on the Pima Indians Diabetes Dataset to estimate the probability of a patient having diabetes based on various health metrics.

### Key Features:
* **Real-time Prediction:** Adjust patient vitals via sliders and see results instantly.
* **Risk Visualization:** Uses a Plotly Gauge chart to show probability instead of a simple "Yes/No."
* **Model Explainability:** A dynamic feature importance chart shows which metrics (e.g., Glucose, BMI) are driving the prediction.
* **Data Exploration:** Interactive histograms to compare user input against the general population distribution.

---

## 🛠️ Technical Stack
* **Frontend:** [Streamlit](https://streamlit.io/)
* **Data Science:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (Random Forest)
* **Visualizations:** Plotly Express & Graph Objects

---

## 🚀 How to Run Locally

### 1. Clone the repository
```bash
git clone [https://github.com/YOUR_USERNAME/diabetes-risk-predictor.git](https://github.com/YOUR_USERNAME/diabetes-risk-predictor.git)
cd diabetes-risk-predictor
