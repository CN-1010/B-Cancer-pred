import numpy as np
import pickle
import streamlit as st
import requests

# Loading data from github      
url = 'https://github.com/CN-1010/B-Cancer-pred/blob/main/logistic_model.sav'

response = requests.get(url)

if response.status_code == 200:
    with open('logistic_model.sav', 'wb') as f:
        f.write(response.content)
else:
    st.error("Failed to download the model file. Please check the URL.")
    st.stop()

# Load the model from the local file
with open('logistic_model.sav', 'rb') as f:
    try:
        model = pickle.load(f)
    except pickle.UnpicklingError as e:
        st.error(f"Failed to load the model: {e}")
        st.stop()

    
# Define the web app

    st.title("Breast Cancer Prediction Web App")
    st.write("Enter the values for the features below:")
    
    # Input fields for all features
    feature_names = [
        'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness',
        ]
    
    # Collect inputs for all features
    user_input = []
    for feature in feature_names:
        value = st.number_input(f"{feature}", value=0.0)
        user_input.append(value)


# Predict the result when the "Predict" button is clicked
if st.button("Predict"):
    try:
        # Perform the prediction
        prediction = model.predict([user_input])
        result = "Malignant" if prediction[0] == 0 else "Benign"
        
        # Display the prediction result
        st.write(f"The predicted class is: **{result}**")
        st.success(f"Prediction successfully made: **{result}**")  # Highlight success
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")






    
    
