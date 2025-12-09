import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="Predicción de Churn de Clientes", layout="wide")

# --- Load Pre-trained Models and Encoders ---
model_filename = 'xgb_churn_model_tuned.joblib'
scaler_filename = 'minmax_scaler.joblib'
label_encoders_filename = 'label_encoders_categorical.joblib'
le_churn_filename = 'label_encoder_churn.joblib'

try:
    model = joblib.load(model_filename)
    scaler = joblib.load(scaler_filename)
    label_encoders = joblib.load(label_encoders_filename)
    le_churn = joblib.load(le_churn_filename)
    st.success("Modelos y preprocesadores cargados correctamente.")
except FileNotFoundError:
    st.error("Error: No se encontraron los archivos del modelo o preprocesadores. Asegúrate de que estén en la misma carpeta.")
    st.stop()
except Exception as e:
    st.error(f"Error al cargar los archivos: {e}")
    st.stop()

# --- Streamlit App Title and Description ---
st.title("\u1F4BB Predicción de Churn de Clientes de Telecomunicaciones")
st.markdown("Esta aplicación predice la probabilidad de que un cliente abandone el servicio (Churn) basándose en sus características.")
st.markdown("--- ")

# Sidebar for user input
st.sidebar.header("Características del Cliente")

# Numerical Inputs
tenure = st.sidebar.slider("Antigüedad del cliente (meses)", 0, 72, 24)
total_charges = st.sidebar.number_input("Cargos totales", min_value=0.0, value=500.0, step=0.1)

# Categorical Inputs (using label_encoders to get original categories)
internet_service_options = list(label_encoders['InternetService'].classes_)
internet_service = st.sidebar.selectbox("Tipo de servicio de Internet", internet_service_options)

online_backup_options = list(label_encoders['OnlineBackup'].classes_)
online_backup = st.sidebar.selectbox("¿Tiene servicio de respaldo en línea?", online_backup_options)

tech_support_options = list(label_encoders['TechSupport'].classes_)
tech_support = st.sidebar.selectbox("¿Tiene servicio de soporte técnico?", tech_support_options)

contract_options = list(label_encoders['Contract'].classes_)
contract = st.sidebar.selectbox("Tipo de contrato", contract_options)

payment_method_options = list(label_encoders['PaymentMethod'].classes_)
payment_method = st.sidebar.selectbox("Método de pago", payment_method_options)

# OnlineSecurity needs special handling because it was One-Hot Encoded
online_security_options = ['No', 'Yes', 'No internet service']
online_security = st.sidebar.selectbox("¿Tiene servicio de seguridad en línea?", online_security_options)

# Preprocessing function
def preprocess_input(tenure, internet_service, online_backup, tech_support, contract, payment_method, total_charges, online_security, scaler, label_encoders):
    # Create a DataFrame for the new input
    input_data = pd.DataFrame({
        'tenure': [tenure],
        'InternetService': [internet_service],
        'OnlineBackup': [online_backup],
        'TechSupport': [tech_support],
        'Contract': [contract],
        'PaymentMethod': [payment_method],
        'TotalCharges': [total_charges],
        'OnlineSecurity': [online_security]
    })

    # Apply Label Encoding for categorical features that were label encoded
    for col in ['InternetService', 'OnlineBackup', 'TechSupport', 'Contract', 'PaymentMethod']:
        if col in label_encoders:
            # Handle potential unseen labels by mapping them to a default or raising an error
            try:
                input_data[col] = label_encoders[col].transform(input_data[col])
            except ValueError:
                st.warning(f"Categoría no vista para {col}: {input_data[col].iloc[0]}. Usando 0 como valor predeterminado.")
                input_data[col] = 0 # Fallback for unseen categories

    # Scale numerical features (tenure and TotalCharges)
    input_data[['tenure', 'TotalCharges']] = scaler.transform(input_data[['tenure', 'TotalCharges']])

    # One-Hot Encode OnlineSecurity
    # Ensure all three OHE columns are present, initialized to 0
    input_data['OnlineSecurity_No'] = 0
    input_data['OnlineSecurity_No internet service'] = 0
    input_data['OnlineSecurity_Yes'] = 0

    if online_security == 'No':
        input_data['OnlineSecurity_No'] = 1
    elif online_security == 'No internet service':
        input_data['OnlineSecurity_No internet service'] = 1
    elif online_security == 'Yes':
        input_data['OnlineSecurity_Yes'] = 1

    input_data = input_data.drop('OnlineSecurity', axis=1)

    # Ensure the order of columns matches the training data (X_train from the notebook)
    # The order is: tenure, InternetService, OnlineBackup, TechSupport, Contract,
    # PaymentMethod, TotalCharges, OnlineSecurity_No, OnlineSecurity_No internet service, OnlineSecurity_Yes
    expected_columns = [
        'tenure', 'InternetService', 'OnlineBackup', 'TechSupport', 'Contract',
        'PaymentMethod', 'TotalCharges', 'OnlineSecurity_No',
        'OnlineSecurity_No internet service', 'OnlineSecurity_Yes'
    ]

    # Reorder columns to match the training data
    processed_input = input_data[expected_columns]

    return processed_input

# Prediction button
if st.sidebar.button("Predecir Churn"):
    # Get raw inputs from widgets
    tenure_val = tenure
    internet_service_val = internet_service
    online_backup_val = online_backup
    tech_support_val = tech_support
    contract_val = contract
    payment_method_val = payment_method
    total_charges_val = total_charges
    online_security_val = online_security

    # Preprocess inputs
    processed_input = preprocess_input(
        tenure_val, internet_service_val, online_backup_val, tech_support_val, contract_val,
        payment_method_val, total_charges_val, online_security_val, scaler, label_encoders
    )

    # Make prediction
    prediction = model.predict(processed_input)
    prediction_proba = model.predict_proba(processed_input)

    st.subheader("Resultados de la Predicción:")
    if prediction[0] == 1:
        st.error(f"El cliente es **PROPENSO AL CHURN** con una probabilidad del **{prediction_proba[0][1]*100:.2f}%**.")
        st.markdown("Se recomienda tomar acciones proactivas para retener a este cliente.")
    else:
        st.success(f"El cliente **NO ES PROPENSO AL CHURN** con una probabilidad del **{(1 - prediction_proba[0][1])*100:.2f}%** de permanecer.")
        st.markdown("Es probable que este cliente continúe con el servicio.")

    st.write("---")
    st.subheader("Detalles del Input Procesado (para depuración):")
    st.write(processed_input)
