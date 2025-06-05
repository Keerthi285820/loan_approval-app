🏦 Loan Approval Prediction Web App

This is a machine learning-based web application that predicts whether a loan application will be approved or not based on input features like income, credit history, education, and more. The app is built using Python, trained with Logistic Regression, and deployed using Streamlit.

🔍 Project Overview

Loan approval is a crucial step in the banking process. This project simplifies that process using a machine learning model that predicts outcomes based on historical loan application data.

👇 Key Features:

Clean data preprocessing and feature engineering
Logistic Regression model for binary classification
Model accuracy evaluation on test data
Web app with interactive user input
Hosted using Streamlit Cloud

🛠️ Technologies Used

Programming	Python 🐍
Data Handling	Pandas, NumPy
Model Building	Scikit-learn (Logistic Regression)
Model Saving	Pickle
Web App Development	Streamlit
Deployment	GitHub + Streamlit

📁 Project Structure

loan-approval-prediction/
│
├── train.csv                 # Training dataset
├── loan_model.pkl            # Saved model (generated after training)
├── app.py                    # Streamlit web app code
├── model_train.py            # Script to train and save the ML model
├── README.md                 # Project overview

🔄 Workflow

1. Data Preprocessing
Drop missing values
Encode categorical variables
2. Feature Selection
Key features: Gender, Education, Credit History, Income, Loan Amount, etc.
3. Model Training
Used Logistic Regression with Scikit-learn
4. Model Evaluation
5. Evaluated using accuracy on test data
6. Model Saving
7. Saved with Pickle for reuse
6. Web App Development
Created a user interface with Streamlit for real-time predictions

🎯 Sample Prediction Criteria
The model predicts loan approval based on:
Gender
Marital status
Education
Employment status
Applicant and Co-applicant income
Loan amount & term
Credit history
Property area

💡 What I Learned
Real-world implementation of end-to-end ML pipeline
Feature encoding and dataset cleaning
Working with Scikit-learn models and performance evaluation
Streamlit-based app creation and deployment
Gained insights into how AI can support financial decision-making

📌 How to Run Locally

1. Clone the repository:
git clone https://github.com/Keerthi285820/loan-approval-prediction.git
cd loan-approval-prediction
2. Install required packages:
pip install -r requirements.txt
3. Run the app:
streamlit run app.py

📂 Dataset Source

The dataset used in this project is a common public dataset used in ML learning environments. You can find similar datasets on Kaggle - Loan Prediction Dataset.

🙌 Acknowledgements

Thanks to the open-source community and ML learning platforms for providing accessible datasets and tools to build practical projects.

