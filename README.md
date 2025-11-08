# myprogram
new project
elco Customer Churn Predictor (Streamlit + Random Forest)
This project is an interactive Streamlit web app that predicts whether a telecom customer is likely to churn (leave) or stay, based on their demographic and service information.
The model is trained on the Telco Customer Churn dataset, using a Random Forest Classifier wrapped in a scikit-learn pipeline for preprocessing and modeling.

 Features
•	End-to-end machine learning pipeline using scikit-learn
•	Automatic model training and caching (only trains once)
•	Streamlit UI for easy customer input and prediction
•	One-click churn prediction with probability scores
•	Interactive data visualization of customer details

Technologies Used
•	Python 3.x
•	Streamlit
•	pandas
•	scikit-learn
•	joblib

Project Structure
text
.
├── churn_model.joblib                 # Saved trained model (auto-generated)
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── app.py                             # Main Streamlit app
├── requirements.txt                   # Dependencies list
├── README.md                          # Project documentation

 How It Works
1.	The script checks if a saved model (churn_model.joblib) exists.
2.	If not found, it loads the dataset, preprocesses it, trains a Random Forest Classifier, and saves the model.
3.	The Streamlit interface collects new customer inputs via the sidebar form.
4.	The trained model predicts whether the customer will churn or stay, along with the probability score.

 Installation
1.	Clone the repository:
bash
git clone https://github.com/rohitcharak123/myprogram.git
cd telco-churn-predictor
2.	Install dependencies:
bash
pip install -r requirements.txt
3.	Ensure the dataset file WA_Fn-UseC_-Telco-Customer-Churn.csv is in the same directory.

 Run the App
bash
streamlit run app.py
The app will start a local server (typically at http://localhost:8501).

 Dataset Information
Source: IBM Telco Customer Churn dataset
Target Variable: Churn (Yes / No)
Key Features:
•	Demographics (Gender, SeniorCitizen, Partner, Dependents)
•	Service details (PhoneService, InternetService, OnlineSecurity, etc.)
•	Account information (Contract, PaymentMethod, Tenure, Charges)

 Model Overview
•	Algorithm: Random Forest Classifier
•	Preprocessing:
•	Numeric scaling (StandardScaler)
•	Categorical encoding (OneHotEncoder with drop-first)
•	Pipeline: Handles all preprocessing and inference in one step

 Example Output
When you enter customer details, the app shows:
•	Prediction: Likely to Stay or Likely to Churn
•	Probability of Churn in percentage
•	Customer details summary table

 Model Persistence
The model is saved automatically as churn_model.joblib after the first training session.
Subsequent runs will reuse this model to save time.

 Future Improvements
•	Add feature importance visualization
•	Include SHAP/LIME explanations
•	Deploy on Streamlit Cloud or Hugging Face Spaces
