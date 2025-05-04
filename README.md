Income Prediction App
A machine learning web application that predicts whether a person earns more than $50K/year based on demographic information. Built using Streamlit and trained on the UCI Adult Income dataset. This project demonstrates the full ML pipeline from data preprocessing to model deployment.

🔍 Overview
This app uses a classification model to predict income category (<=50K or >50K) based on features like age, education, occupation, hours worked per week, etc. It includes:

Data cleaning and preprocessing

Feature encoding and transformation

Model training, evaluation, and tuning

Saving/loading models with Pickle

A web UI with Streamlit

🚀 Features
✅ Clean and modular codebase

🧠 Multiple ML models (Logistic Regression, Random Forest, etc.)

📊 Evaluation metrics: Accuracy, F1-score, ROC-AUC

🖥️ Easy-to-use UI with Streamlit

💾 Model persistence using pickle

📦 Lightweight and ready to deploy

🗂️ Dataset
Source: UCI Adult Income Dataset

Target Variable: income (<=50K or >50K)

Attributes: age, education, occupation, hours-per-week, etc.

🛠️ Tech Stack
Python (pandas, scikit-learn, pickle)

Streamlit (for frontend deployment)

Matplotlib / seaborn (for optional visualizations)

Jupyter Notebook (for EDA and prototyping)

🧪 How to Run
Clone the repo

bash
Copy
Edit
git clone https://github.com/AlivelliAshwanth/income-prediction-app.git
cd income-prediction-app
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Run the app

bash
Copy
Edit
streamlit run app.py


🧠 Model Evaluation
XG BOOST: F1-score = 0.92, ROC-AUC = 0.92

📁 Project Structure
kotlin
Copy
Edit
├── data/
│   └── adult.csv
├── models/
│   └── model.pkl
├── notebooks/
│   └── eda.ipynb
├── app.py
├── pipeline.py
├── requirements.txt
└── README.md

🤝 Contributing
Feel free to fork the repo and submit a pull request. Issues and suggestions are welcome!
