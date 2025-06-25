# Streamlit App: Employee Attrition Prediction

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import ssl

# SSL fix for certain platforms
ssl._create_default_https_context = ssl._create_unverified_context

# Page configuration
st.set_page_config(page_title="Employee Attrition Dashboard", layout="wide")

# Custom styling
st.markdown("""
    <style>
        html, body, .main {
            background-color: #f5f5f7;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell;
            color: #1c1c1e;
        }
        h1, h2, h3 {
            font-weight: 600;
        }
        .stButton > button {
            background-color: #007aff;
            color: white;
            border-radius: 8px;
            padding: 0.4em 1em;
            font-size: 0.9em;
        }
        .metric-label, .metric-value {
            font-size: 1.1em;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("Employee Attrition Prediction Dashboard")

# Sidebar Overview
with st.sidebar:
    st.header("About the Project")
    st.write("""
    - Explore employee data visually
    - Analyze attrition factors
    - Train and evaluate ML models
    - Understand key feature impacts
    """)
    st.markdown("---")
    st.caption("Made by Mohit Methi")

# Load dataset
@st.cache_data

def load_data():
    url = 'https://raw.githubusercontent.com/mohitmethi1000/Employee_attrition_-prediction/refs/heads/main/WA_Fn-UseC_-HR-Employee-Attrition.csv'
    return pd.read_csv(url)

attrition = load_data()

# Optional raw view
if st.checkbox("Show raw dataset"):
    st.dataframe(attrition.head(20), use_container_width=True)

# Target encoding
attrition["Attrition_numerical"] = attrition["Attrition"].map({"Yes": 1, "No": 0})
categorical = [col for col in attrition.columns if attrition[col].dtype == 'object']
numerical = attrition.columns.difference(categorical + ['Attrition_numerical'])

# KDE plots
st.subheader("KDE Plots: Key Variable Distributions")
st.markdown("Visualizing feature interactions through Kernel Density Estimation (KDE) plots.")
fig_kde, axes = plt.subplots(3, 3, figsize=(9, 6))
plot_settings = [
    ('Age', 'TotalWorkingYears', 'Age vs Total Working Years', 0, 0),
    ('Age', 'DailyRate', 'Age vs Daily Rate', 0, 1),
    ('YearsInCurrentRole', 'Age', 'Years in Role vs Age', 0, 2),
    ('DailyRate', 'DistanceFromHome', 'Daily Rate vs Distance From Home', 1, 0),
    ('DailyRate', 'JobSatisfaction', 'Daily Rate vs Job Satisfaction', 1, 1),
    ('YearsAtCompany', 'JobSatisfaction', 'Years At Company vs Job Satisfaction', 1, 2),
    ('YearsAtCompany', 'DailyRate', 'Years At Company vs Daily Rate', 2, 0),
    ('RelationshipSatisfaction', 'YearsWithCurrManager', 'Relationship Satisfaction vs Years With Manager', 2, 1),
    ('WorkLifeBalance', 'JobSatisfaction', 'Work Life Balance vs Job Satisfaction', 2, 2),
]
for x_col, y_col, title, row, col in plot_settings:
    cmap = sns.cubehelix_palette(start=np.random.rand(), light=1, as_cmap=True)
    sns.kdeplot(x=attrition[x_col], y=attrition[y_col], cmap=cmap, fill=True, ax=axes[row, col])
    axes[row, col].set_title(title, fontsize=8)
    axes[row, col].tick_params(labelsize=6)
fig_kde.tight_layout(pad=2)
st.pyplot(fig_kde, clear_figure=True)

# Correlation heatmap
st.subheader("Correlation Heatmap")
st.markdown("Explore correlation between numeric features.")
corr_df = attrition[numerical].astype(float).corr()
fig_corr = px.imshow(
    corr_df,
    labels=dict(x="Features", y="Features", color="Correlation"),
    color_continuous_scale='Viridis',
    width=700, height=600
)
st.plotly_chart(fig_corr, use_container_width=True)

# Feature engineering
attrition_cat = pd.get_dummies(attrition[categorical].drop(['Attrition'], axis=1))
attrition_num = attrition[numerical]
attrition_final = pd.concat([attrition_num, attrition_cat], axis=1)
target = attrition["Attrition_numerical"]

# Train/test split
train, test, target_train, target_val = train_test_split(attrition_final, target, train_size=0.8, random_state=0)

# SMOTE oversampling (FIXED)
train = train.select_dtypes(include=[np.number])  # Ensure numeric features only
target_train = target_train.astype(int)           # Ensure target is int
oversampler = SMOTE(random_state=0)
smote_train, smote_target = oversampler.fit_resample(train, target_train)

# Random Forest model training
st.subheader("Random Forest Model")
rf_params = {
    'n_jobs': -1,
    'n_estimators': 1000,
    'max_features': 'sqrt',
    'max_depth': 4,
    'min_samples_leaf': 2,
    'random_state': 0
}
rf = RandomForestClassifier(**rf_params)
rf.fit(smote_train, smote_target)
rf_predictions = rf.predict(test)

# Evaluation metrics
st.subheader("Evaluation Metrics")
col1, col2 = st.columns(2)
with col1:
    accuracy = accuracy_score(target_val, rf_predictions)
    st.metric(label="Accuracy", value=f"{accuracy:.2f}")
    st.caption("Classification Report")
    st.code(classification_report(target_val, rf_predictions), language="text")
    st.write(f"Model achieved an accuracy of {accuracy * 100:.2f}% on the test set.")

with col2:
    cm = confusion_matrix(target_val, rf_predictions)
    fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm, cbar=False)
    ax_cm.set_title('Confusion Matrix', fontsize=10)
    st.pyplot(fig_cm)

# Feature importance
st.subheader("Feature Importance")
st.markdown("Features contributing most to the model's predictions.")
imp_df = pd.DataFrame({
    'Feature': attrition_final.columns,
    'Importance': rf.feature_importances_
}).sort_values(by='Importance', ascending=False)
fig_imp = px.bar(imp_df.head(15), x='Importance', y='Feature', orientation='h', color='Importance',
                 color_continuous_scale='Teal', height=500)
st.plotly_chart(fig_imp, use_container_width=True)
