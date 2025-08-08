import streamlit as st
import pandas as pd
import joblib

# Load the pre-trained model pipeline
# This pipeline already includes the preprocessor (scaler, one-hot encoder)
try:
    model = joblib.load('engagement_model.pkl')
except FileNotFoundError:
    st.error("File model 'engagement_model.pkl' tidak ditemukan. Pastikan file tersebut berada di direktori yang sama.")
    st.stop()

# --- Page Configuration ---
st.set_page_config(
    page_title="Prediksi Keterlibatan Pelanggan",
    page_icon="🚀",
    layout="centered"
)

# --- UI Design ---
st.title("🚀 Prediksi Keterlibatan Pelanggan")
st.write(
    "Aplikasi ini memprediksi tingkat keterlibatan pelanggan (High, Medium, atau Low) "
    "berdasarkan data demografis dan perilaku mereka. Masukkan data di sidebar untuk memulai."
)

# --- Sidebar for User Inputs ---
st.sidebar.header("Masukkan Data Pelanggan")

def user_input_features():
    """
    Creates sidebar widgets to collect user input.
    """
    age = st.sidebar.slider('Umur', 18, 70, 25)
    gender = st.sidebar.selectbox('Jenis Kelamin', ['Female', 'Male'])
    marital_status = st.sidebar.selectbox('Status Perkawinan', ['Single', 'Married', 'Prefer not to say'])
    occupation = st.sidebar.selectbox('Pekerjaan', ['Student', 'Employee', 'Self Employeed', 'House wife'])
    monthly_income = st.sidebar.selectbox('Pendapatan Bulanan', ['No Income', 'Below Rs.10000', '10001 to 25000', '25001 to 50000', 'More than 50000'])
    edu_qualifications = st.sidebar.selectbox('Kualifikasi Pendidikan', ['Graduate', 'Post Graduate', 'Ph.D', 'School', 'Uneducated'])
    family_size = st.sidebar.slider('Ukuran Keluarga', 1, 10, 3)
    
    # Default values for location data as they are less likely to be changed frequently
    pin_code = 560001
    latitude = 12.9716
    longitude = 77.5946

    # Create a dictionary from the inputs
    data = {
        'Age': age,
        'Gender': gender,
        'Marital Status': marital_status,
        'Occupation': occupation,
        'Monthly Income': monthly_income,
        'Educational Qualifications': edu_qualifications,
        'Family size': family_size,
        'latitude': latitude,
        'longitude': longitude,
        'Pin code': pin_code
    }
    
    # Convert dictionary to a pandas DataFrame
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Display the user input in the main area
st.subheader("Data Pelanggan yang Dimasukkan:")
st.write(input_df)

# --- Prediction Logic ---
if st.sidebar.button('Prediksi Sekarang'):
    # The model pipeline will handle preprocessing automatically
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    st.subheader("Hasil Prediksi:")
    
    # Display the result with styling
    if prediction[0] == 'High':
        st.success(f"**Tingkat Keterlibatan: {prediction[0]}** 🎉")
        st.balloons()
    elif prediction[0] == 'Medium':
        st.warning(f"**Tingkat Keterlibatan: {prediction[0]}** 😐")
    else:
        st.error(f"**Tingkat Keterlibatan: {prediction[0]}** 😟")

    # Display prediction probabilities
    st.subheader("Probabilitas Prediksi:")
    proba_df = pd.DataFrame(prediction_proba, columns=model.classes_, index=["Probabilitas"])
    st.write(proba_df)

st.sidebar.info("Aplikasi ini dibuat sebagai bagian dari proyek UAS MPML.")
