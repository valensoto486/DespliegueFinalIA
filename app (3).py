
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the pre-trained XGBoost model
model = joblib.load('xgb_churn_model_tuned.joblib')

# Load the MinMaxScaler
saler = joblib.load('minmax_scaler.joblib')

# Load the LabelEncoders for categorical features
label_encoders = joblib.load('label_encoders_categorical.joblib')

st.write("Models and preprocessing objects loaded successfully!")

st.title('Predicción de Churn de Clientes de Telecomunicaciones')
st.write('Ingrese los detalles del cliente para predecir si abandonará el servicio.')

# Input fields for user data
tenure = st.slider('Antigüedad (meses)', 0, 72, 12)
internet_service = st.selectbox('Servicio de Internet', ['DSL', 'Fiber optic', 'No'])
online_backup = st.selectbox('Copia de seguridad en línea', ['Yes', 'No', 'No internet service'])
tech_support = st.selectbox('Soporte técnico', ['Yes', 'No', 'No internet service'])
contract = st.selectbox('Tipo de Contrato', ['Month-to-month', 'One year', 'Two year'])
payment_method = st.selectbox('Método de Pago', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
total_charges = st.number_input('Cargos Totales', min_value=0.0, value=500.0, step=10.0)
online_security = st.selectbox('Seguridad en línea', ['Yes', 'No', 'No internet service'])

def preprocess_input(tenure, internet_service, online_backup, tech_support, contract, payment_method, total_charges, online_security):
    # Create a DataFrame from the input
    input_df = pd.DataFrame({
        'tenure': [tenure],
        'InternetService': [internet_service],
        'OnlineBackup': [online_backup],
        'TechSupport': [tech_support],
        'Contract': [contract],
        'PaymentMethod': [payment_method],
        'TotalCharges': [total_charges],
        'OnlineSecurity': [online_security]
    })

    # Apply Label Encoding to categorical features
    for col, le in label_encoders.items():
        # Handle cases where the input value might not be in the original categories
        # If a category is new, it will be mapped to -1 or a default value
        input_df[col] = input_df[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

    # Scale numerical features
    cols_to_scale = ['TotalCharges', 'tenure'] # Ensure these match the original scaling columns
    input_df[cols_to_scale] = saler.transform(input_df[cols_to_scale])

    # One-hot encode 'OnlineSecurity' as it was done in training
    # Create all possible columns first to ensure consistent order and presence
    online_security_cols = ['OnlineSecurity_No', 'OnlineSecurity_No internet service', 'OnlineSecurity_Yes']
    for col_name in online_security_cols:
        input_df[col_name] = 0
    
    # Assign 1 to the correct one-hot encoded column
    if online_security == 'No':
        input_df['OnlineSecurity_No'] = 1
    elif online_security == 'No internet service':
        input_df['OnlineSecurity_No internet service'] = 1
    elif online_security == 'Yes':
        input_df['OnlineSecurity_Yes'] = 1

    # Drop the original 'OnlineSecurity' column after one-hot encoding
    input_df = input_df.drop('OnlineSecurity', axis=1)

    # Ensure column order matches training data (X_train had this order originally)
    # Get the columns from X_train or X_resampled
    # For this, we need to know the exact column names from the training set
    # Assuming the column order after preprocessing in the notebook was: 
    # tenure, InternetService, OnlineBackup, TechSupport, Contract, PaymentMethod, TotalCharges, OnlineSecurity_No, OnlineSecurity_No internet service, OnlineSecurity_Yes
    # This order is critical for the model prediction.
    
    # Manually define the expected order of columns based on the training data after preprocessing
    expected_columns = [
        'tenure',
        'InternetService',
        'OnlineBackup',
        'TechSupport',
        'Contract',
        'PaymentMethod',
        'TotalCharges',
        'OnlineSecurity_No',
        'OnlineSecurity_No internet service',
        'OnlineSecurity_Yes'
    ]

    # Reorder columns to match the training data
    processed_input = input_df[expected_columns]

    return processed_input

if st.button('Predecir Churn'):
    processed_data = preprocess_input(tenure, internet_service, online_backup, tech_support, contract, payment_method, total_charges, online_security)
    prediction = model.predict(processed_data)
    prediction_proba = model.predict_proba(processed_data)

    st.subheader('Resultado de la Predicción:')
    if prediction[0] == 1:
        st.error(f'¡El cliente probablemente abandonará el servicio (Churn)! Probabilidad: {prediction_proba[0][1]*100:.2f}%')
    else:
        st.success(f'El cliente probablemente se quedará. Probabilidad: {prediction_proba[0][0]*100:.2f}%')
