import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

# Load the trained model
try:
    with open("xgb_income_model.pkl", "rb") as f:
        model = pickle.load(f)
except Exception as e:
    import traceback
    error_message = f"Error loading model: {e}\n{traceback.format_exc()}"
    print(error_message)
    st.error(error_message)
    model = None

# Define the prediction function
def predict_income(input_data):
    # Feature Engineering (similar to your original code)
    input_data['capital_gain_flag'] = (input_data['capital-gain'] > 0).astype(int)
    input_data['capital_loss_flag'] = (input_data['capital-loss'] > 0).astype(int)
    input_data['age_bin'] = pd.cut(input_data['age'], bins=[0, 30, 50, 100], labels=[0, 1, 2]).astype(int)
    input_data['capital-gain'] = np.log1p(input_data['capital-gain'])
    input_data['capital-loss'] = np.log1p(input_data['capital-loss'])
    # ... (apply other feature engineering steps as needed) ...

    # One-hot encoding for categorical columns
    categorical_cols = input_data.select_dtypes(include=['object']).columns.tolist()
    input_data = pd.get_dummies(input_data, columns=categorical_cols, drop_first=True)

    # Feature Scaling
    scaler = StandardScaler()
    numeric_cols = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    input_data[numeric_cols] = scaler.fit_transform(input_data[numeric_cols])

    # Align input features with model's expected features
    model_features = model.get_booster().feature_names
    for col in model_features:
        if col not in input_data.columns:
            input_data[col] = 0
    input_data = input_data[model_features]

    # Make prediction
    prediction = model.predict(input_data)
    return "<=50K" if prediction[0] == 0 else ">50K"

# Streamlit app UI with improved visual layout and styling

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main {
        background-color: #f0f2f6;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #333333;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        padding: 10px 24px;
        border-radius: 8px;
        border: none;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .section-header {
        font-size: 22px;
        font-weight: 600;
        color: #2c3e50;
        margin-top: 20px;
        margin-bottom: 10px;
        border-bottom: 2px solid #4CAF50;
        padding-bottom: 4px;
    }
    .input-label {
        font-weight: 500;
        color: #34495e;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Income Prediction App")
st.markdown("### Predict whether an individual's income exceeds $50K/year based on census data.")

# Group inputs into sections and columns for better layout
with st.container():
    st.markdown('<div class="section-header">Demographic Information</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", min_value=16, max_value=90, value=30)
        education_num = st.number_input("Education Number", min_value=1, max_value=16, value=10)
        sex = st.selectbox("Sex", ["Female", "Male"])
        race = st.selectbox("Race", ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"])
    with col2:
        marital_status = st.selectbox("Marital Status", ["Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"])
        relationship = st.selectbox("Relationship", ["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unrelated"])
        native_country = st.selectbox("Native Country", ["United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany", "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece", "South", "China", "Cuba", "Iran", "Honduras", "Philippines", "Italy", "Poland", "Jamaica", "Vietnam", "Mexico", "Portugal", "Ireland", "France", "Dominican-Republic", "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia", "Hungary", "Guatemala", "Nicaragua", "Scotland", "Thailand", "Yugoslavia", "El-Salvador", "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands"])
    with col3:
        workclass = st.selectbox("Workclass", ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked"])
        education = st.selectbox("Education", ["Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th", "Masters", "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool"])
        occupation = st.selectbox("Occupation", ["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"])

st.markdown('<div class="section-header">Financial Information</div>', unsafe_allow_html=True)
col4, col5 = st.columns(2)
with col4:
    capital_gain = st.number_input("Capital Gain", min_value=0, value=0)
    capital_loss = st.number_input("Capital Loss", min_value=0, value=0)
with col5:
    hours_per_week = st.number_input("Hours per Week", min_value=1, max_value=99, value=40)

# Create input data dictionary
input_data = {
    "age": age,
    "workclass": workclass,
    "education": education,
    "education-num": education_num,
    "marital-status": marital_status,
    "occupation": occupation,
    "relationship": relationship,
    "race": race,
    "sex": sex,
    "capital-gain": capital_gain,
    "capital-loss": capital_loss,
    "hours-per-week": hours_per_week,
    "native-country": native_country
}

# Convert input data to DataFrame
input_df = pd.DataFrame([input_data])

# Make prediction
if st.button("Predict"):
    if model is None:
        st.error("Model is not loaded. Cannot make predictions.")
    else:
        prediction = predict_income(input_df)
        st.success(f"Predicted Income: **{prediction}**")
