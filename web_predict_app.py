import numpy as np
import pickle
import streamlit as st
import pandas as pd
import requests

# Loading data from github      
url = 'https://github.com/CN-1010/B-Cancer-pred/blob/main/logistic_model.sav'

loaded_moldel = requests.get(url)

with open('logistic_model.sav', 'wb') as f:
    pickle.dump(loaded_moldel, f)
    
with open('logistic_model.sav', 'rb') as f:
    loaded_moldel = pickle.load(f)   
    
# Define the feature input
def predict_breast_cancer(input_data):
    # Ensure input data is in the correct shape
    input_array = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_array)
    probability = model.predict_proba(input_array)
    return prediction[0], probability[0]

# Streamlit App
def main():
    st.title("Breast Cancer Prediction Web App")
    st.write("Enter the following feature values to predict if the cancer is Malignant (1) or Benign (0).")
    
    # Collecting user input
    features = [ "mean radius", "mean texture", "mean perimeter", "mean area",
                 "mean smoothness",
                 ]
    input_data = []
    for feature in features:
        try:
            value = st.number_input(f"Enter {feature}:", value=0.0)
            input_data.append(value)
        except ValueError:
                 st.error("Please ensure all inputs are valid numbers.")
    # Predict and display the result
    def predict_breast_cancer(input_data):
        # Check if input_data is valid
        if not input_data or len(input_data) != len(features):
            raise ValueError("Input data is incomplete or invalid.")
            # Reshape the input array for the model
            input_array = np.array(input_data).reshape(1, -1)
            
            
            # Make prediction
            prediction = loaded_model.predict(input_array)
            probability = loaded_model.predict_proba(input_array)
            return prediction[0], probability[0]
        
        
    

# Predict and display the result
if st.button("Predict"):
    result, prob = predict_breast_cancer(input_data)
    if result == 1:
        st.write("The prediction: Malignant (Cancerous)")
    else:
        st.write("The prediction: Benign (Non-Cancerous)")
    st.write(f"Prediction Probability: {prob}")
    
if __name__== '__main__':
        main()
    