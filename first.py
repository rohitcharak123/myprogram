import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# --- Constants ---
DATA_PATH = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_PATH = "churn_model.joblib"

# --- Model Training Function ---
def train_and_save_model(data_path):
    """
    Loads data, trains a Random Forest model, and saves it to a .joblib file.
    This function only runs if the model file doesn't already exist.
    """
    st.write("Training a new model... (This only happens once)")

    # Load and clean data
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        st.error(f"Error: Data file '{data_path}' not found.")
        st.stop()
        return  # Extra safety

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna(subset=['TotalCharges'])
    df = df.drop('customerID', axis=1)
    df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

    # Define features and target
    y = df['Churn']
    X = df.drop('Churn', axis=1)

    # Define feature types
    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    
    # We must explicitly list all categorical features for the input form
    categorical_features = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaperlessBilling', 'PaymentMethod'
    ]

    # Handle 'SeniorCitizen' which is 0 or 1 but read as int
    # We'll treat it as categorical for one-hot encoding
    X['SeniorCitizen'] = X['SeniorCitizen'].astype(str)

    # Create pre-processing pipeline
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Create the full model pipeline (Pre-process -> Classify)
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', RandomForestClassifier(random_state=42))])

    # Split and train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model.fit(X_train, y_train)

    # Test and print accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model trained with accuracy: {accuracy:.4f}")

    # Save the pipeline
    joblib.dump(model, MODEL_PATH)
    st.write(f"Model saved to {MODEL_PATH} with accuracy: {accuracy:.4f}")
    
    return model

# --- Model Loading Function ---
@st.cache_resource
def load_model(model_path):
    """
    Loads the trained model from disk. If not found, trains a new one.
    """
    if not os.path.exists(model_path):
        model = train_and_save_model(DATA_PATH)
    else:
        print("Loading existing model.")
        model = joblib.load(model_path)
    return model

# --- Streamlit App UI ---
def run_app(model):
    st.title("\U0001F4C8 Telco Customer Churn Predictor")
    st.markdown("""
    This app predicts whether a customer is likely to churn (leave) or stay.
    Use the controls in the sidebar to enter customer details and click **Predict**.
    """)

    st.sidebar.header("Customer Details")

    # --- Create Input Fields in Sidebar ---
    
    # Numeric Inputs
    tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
    MonthlyCharges = st.sidebar.slider("Monthly Charges ($)", 0.0, 120.0, 50.0, 1.0)
    TotalCharges = st.sidebar.slider("Total Charges ($)", 0.0, 9000.0, 1000.0, 10.0)

    # Categorical Inputs
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    SeniorCitizen = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
    Partner = st.sidebar.selectbox("Has Partner?", ["No", "Yes"])
    Dependents = st.sidebar.selectbox("Has Dependents?", ["No", "Yes"])
    PhoneService = st.sidebar.selectbox("Has Phone Service?", ["No", "Yes"])
    
    MultipleLines_options = ["No", "Yes", "No phone service"]
    MultipleLines = st.sidebar.selectbox("Multiple Lines?", MultipleLines_options)
    
    InternetService_options = ["DSL", "Fiber optic", "No"]
    InternetService = st.sidebar.selectbox("Internet Service", InternetService_options)

    # Dependent categorical options (based on Internet Service)
    no_internet_service = "No internet service"
    if InternetService == "No":
        OnlineSecurity_options = [no_internet_service]
        OnlineBackup_options = [no_internet_service]
        DeviceProtection_options = [no_internet_service]
        TechSupport_options = [no_internet_service]
        StreamingTV_options = [no_internet_service]
        StreamingMovies_options = [no_internet_service]
    else:
        OnlineSecurity_options = ["No", "Yes"]
        OnlineBackup_options = ["No", "Yes"]
        DeviceProtection_options = ["No", "Yes"]
        TechSupport_options = ["No", "Yes"]
        StreamingTV_options = ["No", "Yes"]
        StreamingMovies_options = ["No", "Yes"]

    OnlineSecurity = st.sidebar.selectbox("Online Security", OnlineSecurity_options)
    OnlineBackup = st.sidebar.selectbox("Online Backup", OnlineBackup_options)
    DeviceProtection = st.sidebar.selectbox("Device Protection", DeviceProtection_options)
    TechSupport = st.sidebar.selectbox("Tech Support", TechSupport_options)
    StreamingTV = st.sidebar.selectbox("Streaming TV", StreamingTV_options)
    StreamingMovies = st.sidebar.selectbox("Streaming Movies", StreamingMovies_options)

    Contract_options = ["Month-to-month", "One year", "Two year"]
    Contract = st.sidebar.selectbox("Contract", Contract_options)
    
    PaperlessBilling = st.sidebar.selectbox("Paperless Billing?", ["No", "Yes"])
    
    PaymentMethod_options = ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    PaymentMethod = st.sidebar.selectbox("Payment Method", PaymentMethod_options)

    # --- Prediction Logic ---
    if st.sidebar.button("Predict Churn"):
        # 1. Convert "Yes"/"No" to 1/0 for SeniorCitizen
        # The pipeline expects a string '0' or '1' as it was trained
        senior_citizen_str = '1' if SeniorCitizen == 'Yes' else '0'

        # 2. Create a dictionary of all inputs
        # The keys MUST match the column names in the original DataFrame
        input_data = {
            'tenure': tenure,
            'MonthlyCharges': MonthlyCharges,
            'TotalCharges': TotalCharges,
            'gender': gender,
            'SeniorCitizen': senior_citizen_str,
            'Partner': Partner,
            'Dependents': Dependents,
            'PhoneService': PhoneService,
            'MultipleLines': MultipleLines,
            'InternetService': InternetService,
            'OnlineSecurity': OnlineSecurity,
            'OnlineBackup': OnlineBackup,
            'DeviceProtection': DeviceProtection,
            'TechSupport': TechSupport,
            'StreamingTV': StreamingTV,
            'StreamingMovies': StreamingMovies,
            'Contract': Contract,
            'PaperlessBilling': PaperlessBilling,
            'PaymentMethod': PaymentMethod
        }

        # 3. Convert to DataFrame
        input_df = pd.DataFrame([input_data])

        # 4. Make Prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0] # [Prob_Stay, Prob_Churn]

        # 5. Display Result
        st.subheader("Prediction Result")
        churn_prob = probability[1]
        
        if prediction == 1:
            st.error(f"This customer is **LIKELY TO CHURN** (Probability: {churn_prob*100:.2f}%)", icon="🚨")
        else:
            st.success(f"This customer is **LIKELY TO STAY** (Churn Probability: {churn_prob*100:.2f}%)", icon="✅")

        st.write("---")
        st.subheader("Input Customer Data:")
        st.dataframe(input_df.T.rename(columns={0: 'Customer Details'}))

# --- Main execution ---
if __name__ == "__main__":
    model_pipeline = load_model(MODEL_PATH)
    run_app(model_pipeline)
