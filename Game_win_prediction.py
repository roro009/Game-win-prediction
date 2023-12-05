#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Function to load data
@st.cache  # This decorator will cache the data loading function
def load_data(uploaded_file):
    data = pd.read_csv(uploaded_file)
    return data

# Function to preprocess data
def preprocess_data(data):
    data_cleaned = data.drop(columns=['Unnamed: 21'])  # Example: Dropping a specific column
    data_cleaned = data_cleaned.dropna(subset=['GAME_DATE_EST', 'GAME_ID', 'HOME_TEAM_ID', 'VISITOR_TEAM_ID', 'SEASON', 'PTS_home', 'PTS_away', 'HOME_TEAM_WINS'])
    return data_cleaned

# Function to train the model
def train_model(data):
    features = ['HOME_TEAM_ID', 'VISITOR_TEAM_ID', 'SEASON', 'PTS_home', 'FG_PCT_home', 'FT_PCT_home', 'FG3_PCT_home', 'AST_home', 'REB_home', 'PTS_away', 'FG_PCT_away', 'FT_PCT_away', 'FG3_PCT_away', 'AST_away', 'REB_away']
    X = data[features]
    y = data['HOME_TEAM_WINS']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    return rf, X_test, y_test

# Streamlit application
def main():
    st.title("Game Win Prediction Model")

    # Upload CSV file
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        data = load_data(uploaded_file)
        st.write(data.head())  # Display the first few rows of the dataset

        if st.button('Preprocess and Train Model'):
            data_cleaned = preprocess_data(data)
            model, X_test, y_test = train_model(data_cleaned)
            st.write("Model trained successfully!")

            # Model Evaluation
            accuracy = model.score(X_test, y_test)
            st.write(f"Model Accuracy: {accuracy}")

            # Cross-validation
            if st.checkbox('Perform 5-fold Cross-Validation'):
                cv_scores = cross_val_score(model, X_test, y_test, cv=5)
                st.write(f"Cross-Validation Scores: {cv_scores}")
                st.write(f"Average Score: {np.mean(cv_scores)}")

# Run the Streamlit application
if __name__ == "__main__":
    main()


