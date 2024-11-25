#!/usr/bin/env python
# coding: utf-8

# In[6]:


import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load Dataset
@st.cache
def load_data():
    return pd.read_csv(r'/Users/akashthota/Downloads/adult.csv')

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
@st.cache
def train_model(data):
    X = data.drop("income", axis=1)
    y = data["income"]
    model = RandomForestClassifier()
    model.fit(X, y)
    return model

# Load Data
st.title("Income Prediction App")
data = load_data()

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


# In[ ]:




