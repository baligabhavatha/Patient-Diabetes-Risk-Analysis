import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# --- PAGE CONFIG ---
st.set_page_config(page_title="HealthPredict: Diabetes Risk", page_icon="🏥", layout="wide")

# --- DATA LOADING & MODELING ---
@st.cache_data
def get_data():
    # Load dataset directly from source
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DPF', 'Age', 'Outcome']
    df = pd.read_csv(url, names=columns)
    return df

df = get_data()

# Train a simple model
X = df.drop('Outcome', axis=1)
y = df['Outcome']
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# --- SIDEBAR: USER INPUTS ---
st.sidebar.header("📋 Patient Vitals")
def user_input():
    preg = st.sidebar.number_input("Pregnancies", 0, 20, 1)
    glu = st.sidebar.slider("Glucose Level", 0, 200, 120)
    bp = st.sidebar.slider("Blood Pressure (mm Hg)", 0, 122, 70)
    bmi = st.sidebar.slider("BMI", 0.0, 67.0, 25.0)
    dpf = st.sidebar.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
    age = st.sidebar.slider("Age", 21, 81, 33)
    
    # Matching the structure of training data (others as mean for simplicity)
    data = {
        'Pregnancies': preg, 'Glucose': glu, 'BloodPressure': bp,
        'SkinThickness': 20, 'Insulin': 79, 'BMI': bmi, 'DPF': dpf, 'Age': age
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input()

# --- MAIN PAGE UI ---
st.title("🏥 Patient Diabetes Risk Analysis")
st.markdown("This tool uses a Random Forest Classifier to estimate the probability of diabetes risk based on clinical measurements.")

# Layout: Two Columns
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Analysis Results")
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0][1]

    # Display Result Gauge
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = prediction_proba * 100,
        title = {'text': "Risk Percentage"},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "#ff4b4b" if prediction_proba > 0.5 else "#00cc96"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 100], 'color': "gray"}]
        }
    ))
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

    if prediction == 1:
        st.error("⚠️ High Risk Detected: Consult a healthcare professional.")
    else:
        st.success("✅ Low Risk Detected: Results within standard range.")

with col2:
    st.subheader("Model Insights")
    # Feature Importance Plot
    importances = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=True)
    
    fig_imp = px.bar(importances, x='Importance', y='Feature', orientation='h', 
                     title="What drives this prediction?")
    st.plotly_chart(fig_imp, use_container_width=True)

# --- DATA EXPLORER ---
with st.expander("🔍 View Training Data Distribution"):
    feat_choice = st.selectbox("Select metric to view distribution", X.columns)
    fig_hist = px.histogram(df, x=feat_choice, color="Outcome", barmode="overlay",
                            color_discrete_map={0: "green", 1: "red"})
    st.plotly_chart(fig_hist, use_container_width=True)