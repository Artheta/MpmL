import streamlit as st
import pandas as pd

# Load the pre-trained model pipeline
# This pipeline already includes the preprocessor (scaler, one-hot encoder)
try:
    model = joblib.load('engagement_model.pkl')
except FileNotFoundError:
    st.error("Model file 'engagement_model.pkl' not found. Please make sure it is in the same directory.")
    st.stop()

# --- Page Configuration ---
st.set_page_config(
    page_title="Customer Engagement Prediction",
    page_icon="🚀",
    layout="centered"
)

# --- UI Design ---
st.title("🚀 Customer Engagement Prediction")
st.write(
    "This application predicts the customer engagement level (High, Medium, or Low) "
    "based on their demographic and behavioral data. Enter the data in the sidebar to get started."
)

# --- Sidebar for User Inputs ---
st.sidebar.header("Enter Customer Data")

def user_input_features():
    """
    Creates sidebar widgets to collect user input.
    """
    age = st.sidebar.slider('Age', 18, 70, 25)
    gender = st.sidebar.selectbox('Gender', ['Female', 'Male'])
    marital_status = st.sidebar.selectbox('Marital Status', ['Single', 'Married', 'Prefer not to say'])
    occupation = st.sidebar.selectbox('Occupation', ['Student', 'Employee', 'Self Employeed', 'House wife'])
    monthly_income = st.sidebar.selectbox('Monthly Income', ['No Income', 'Below Rs.10000', '10001 to 25000', '25001 to 50000', 'More than 50000'])
    edu_qualifications = st.sidebar.selectbox('Educational Qualifications', ['Graduate', 'Post Graduate', 'Ph.D', 'School', 'Uneducated'])
    family_size = st.sidebar.slider('Family Size', 1, 10, 3)
    
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
st.subheader("Customer Data Input:")
st.write(input_df)

# --- Prediction Logic ---
if st.sidebar.button('Predict Now'):
    # The model pipeline will handle preprocessing automatically
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    st.subheader("Prediction Result:")
    
    # Display the result with styling
    if prediction[0] == 'High':
        st.success(f"**Engagement Level: {prediction[0]}** 🎉")
        st.balloons()
    elif prediction[0] == 'Medium':
        st.warning(f"**Engagement Level: {prediction[0]}** 😐")
    else:
        st.error(f"**Engagement Level: {prediction[0]}** 😟")

    # Display prediction probabilities
    st.subheader("Prediction Probabilities:")
    proba_df = pd.DataFrame(prediction_proba, columns=model.classes_, index=["Probability"])
    st.write(proba_df)

st.sidebar.info("This app was created as part of the MPML final project.")

