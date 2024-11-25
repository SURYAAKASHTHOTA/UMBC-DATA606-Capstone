#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load Dataset (from file uploader)
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    else:
        st.error("Please upload a CSV file.")
        return None

# Preprocess Dataset
def preprocess_data(data):
    # Handle missing values (if any)
    data = data.dropna()

    # Encode categorical variables
    label_encoders = {}
    for column in data.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le
    return data, label_encoders

# Train a basic model (for demonstration purposes)
@st.cache_resource
def train_model(data):
    X = data.drop("income", axis=1)
    y = data["income"]
    model = RandomForestClassifier()
    model.fit(X, y)
    return model

# Streamlit Interface
st.title("Income Prediction App")

# File Upload Section
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

# Load Data
data = load_data(uploaded_file)

if data is not None:
    # Display Raw Data
    if st.checkbox("Show Raw Data"):
        st.write(data.head())

    # Preprocess Data
    processed_data, encoders = preprocess_data(data)

    # Train Model
    model = train_model(processed_data)

    # Input Features for Prediction
    st.sidebar.header("User Input Features")

    def user_input_features():
        user_data = {}
        for column in data.drop("income", axis=1).columns:
            if column in encoders:
                options = encoders[column].classes_
                user_data[column] = st.sidebar.selectbox(f"{column}", options)
            else:
                user_data[column] = st.sidebar.slider(f"{column}", int(data[column].min()), int(data[column].max()))
        return pd.DataFrame([user_data])

    input_df = user_input_features()

    # Convert input to numerical values using the same encoders
    for column in input_df.columns:
        if column in encoders:
            input_df[column] = encoders[column].transform(input_df[column])

    # Make Predictions
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    # Display Results
    st.write("### Prediction Results")
    income_category = encoders["income"].inverse_transform(prediction)[0]
    st.write(f"Predicted Income Category: *{income_category}*")
    st.write(f"Prediction Probabilities: {prediction_proba}")

