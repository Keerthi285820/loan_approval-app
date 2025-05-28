import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import os

DATA_FILE = "train.csv"
MODEL_FILE = "loan_model.pkl"

# Function to train and save model
def train_and_save_model():
    df = pd.read_csv(DATA_FILE)
    df.dropna(inplace=True)

    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
    df['Married'] = df['Married'].map({'Yes': 1, 'No': 0})
    df['Education'] = df['Education'].map({'Graduate': 1, 'Not Graduate': 0})
    df['Self_Employed'] = df['Self_Employed'].map({'Yes': 1, 'No': 0})
    df['Property_Area'] = df['Property_Area'].map({'Rural': 0, 'Semiurban': 1, 'Urban': 2})
    df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})

    X = df[['Gender', 'Married', 'Education', 'Self_Employed',
            'ApplicantIncome', 'CoapplicantIncome',
            'LoanAmount', 'Loan_Amount_Term',
            'Credit_History', 'Property_Area']]
    y = df['Loan_Status']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(model, f)

    acc = accuracy_score(y_test, model.predict(X_test))
    return model, acc

# Load or train model
if not os.path.exists(MODEL_FILE):
    model, accuracy = train_and_save_model()
else:
    with open(MODEL_FILE, 'rb') as f:
        model = pickle.load(f)
    accuracy = None

# Streamlit UI
st.title("🏦 Loan Approval Prediction App")
st.markdown("Enter the details below to check if your loan will be approved.")

# Input form
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
loan_amount_term = st.number_input("Loan Amount Term", min_value=0)
credit_history = st.selectbox("Credit History (1=Good, 0=Bad)", [1, 0])
property_area = st.selectbox("Property Area", ["Rural", "Semiurban", "Urban"])

if st.button("Predict"):
    input_data = pd.DataFrame([[
        1 if gender == "Male" else 0,
        1 if married == "Yes" else 0,
        1 if education == "Graduate" else 0,
        1 if self_employed == "Yes" else 0,
        applicant_income,
        coapplicant_income,
        loan_amount,
        loan_amount_term,
        credit_history,
        {"Rural": 0, "Semiurban": 1, "Urban": 2}[property_area]
    ]], columns=['Gender', 'Married', 'Education', 'Self_Employed',
                 'ApplicantIncome', 'CoapplicantIncome',
                 'LoanAmount', 'Loan_Amount_Term',
                 'Credit_History', 'Property_Area'])

    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.success("✅ Loan Approved!")
    else:
        st.error("❌ Loan Not Approved.")

if accuracy:
    st.sidebar.info(f"Model Accuracy: {accuracy:.2f}")
