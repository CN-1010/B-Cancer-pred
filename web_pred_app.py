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
st.title("Breast Cancer Prediction Web App")
st.write("Enter the following feature values to predict if the cancer is Malignant (1) or Benign (0).")

# Collecting user input
features = [
    "mean radius", "mean texture", "mean perimeter", "mean area",
    "mean smoothness", "mean compactness", "mean concavity",
    "mean concave points", "mean symmetry", "mean fractal dimension",
]

input_data = []
for feature in features:
    value = st.number_input(f"Enter {feature}:", value=0.0)
    input_data.append(value)

# Predict and display the result
if st.button("Predict"):
    result, prob = predict_breast_cancer(input_data)
    if result == 1:
        st.write("The prediction: Malignant (Cancerous)")
    else:
        st.write("The prediction: Benign (Non-Cancerous)")
    st.write(f"Prediction Probability: {prob}")
    