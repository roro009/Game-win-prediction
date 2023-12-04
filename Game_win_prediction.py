#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
games = pd.read_csv('games.csv')
games_details = pd.read_csv('games_details.csv')
players = pd.read_csv('players.csv')
ranking = pd.read_csv('ranking.csv')
teams = pd.read_csv('teams.csv')

# Data cleaning steps...
# E.g., Handle missing values, remove unnecessary columns

# EDA
# Visualize distributions of points, assists, etc.
sns.histplot(games_details['PTS'])
plt.show()
# More visualizations...


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Feature engineering and selection...
# Create a feature DataFrame (X) and a target vector (y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Evaluate the model
print(accuracy_score(y_test, predictions))


# In[ ]:


import streamlit as st

# Load the model (assuming it's saved as 'model.pkl')
model = pd.read_pickle('model.pkl')

st.title('NBA Game Win Predictor')

# User inputs for prediction
# team, player stats, etc.

if st.button('Predict'):
    # Process user inputs
    # Make prediction
    # st.write('Prediction: ', prediction)
