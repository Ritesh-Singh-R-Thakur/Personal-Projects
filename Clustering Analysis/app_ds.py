# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 16:55:08 2024

@author: rites
"""

# app.py
import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open('decision_tree_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Feature names (order matters)
feature_names = ['Income', 'Recency', 'Complain', 'Response', 'Age',
                 'Days_Since_Enrollment', 'Kids', 'Total_expense',
                 'TotalAcceptedCmp', 'TotalNumOfPurchase', 'Education_PG',
                 'Education_UG', 'Marital_Status_In relationship',
                 'Marital_Status_Single']

st.title("Decision Tree Model Deployment")
st.write("Enter feature values:")

# Create input fields for features
user_input = {}
for feat in feature_names:
    user_input[feat] = st.number_input(feat, value=0.0)

# Make predictions
if st.button("Predict"):
    features = [user_input[feat] for feat in feature_names]
    prediction = model.predict([features])[0]
    st.success(f"Predicted class: {prediction}")

# Optional: Add more UI elements (plots, explanations, etc.)
