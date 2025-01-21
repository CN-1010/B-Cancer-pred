import numpy as np
import pickle
import streamlit as st
import requests

# Loading data from github      
url = 'https://github.com/CN-1010/B-Cancer-pred/blob/main/logistic_model.sav'

response = requests.get(url)

with open('logistic_model.sav', 'wb') as f:
    pickle.dump(response,f)
    
with open('logistic_model.sav', 'rb') as f:
    model = pickle.load(f)   
    
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






    
    
