# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 07:25:40 2024

@author: rites
"""
import pickle
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta

# Load the data
df = pd.read_csv("Gold_data (1).csv")
df.set_index('date', inplace=True)

# Load the model
with open('model_arima.pkl', 'rb') as pkl:
    model_arima = pickle.load(pkl)

# Streamlit app
st.title('Gold Price Prediction for Next 30 Days')

# Date input
start_date = st.date_input("Enter the start date", value=datetime.strptime(df.index[-1], "%Y-%m-%d"))

if st.button('Predict'):
    forecast = model_arima.forecast(steps=30)
    forecast_list = forecast.tolist()

    # Create dates for the next 30 days
    start_date = datetime.combine(start_date, datetime.min.time())
    dates = [start_date + timedelta(days=i) for i in range(1, 31)]

    # Combine dates and forecast into a DataFrame
    prediction_df = pd.DataFrame({'Date': dates, 'Predicted Price': forecast_list})
    st.write(prediction_df)

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(prediction_df['Date'], prediction_df['Predicted Price'], label='Predicted Prices', color='blue')
    plt.xlabel('Date')
    plt.ylabel('Predicted Price')
    plt.title('Gold Price Prediction for Next 30 Days')
    plt.legend()
    st.pyplot(plt)
