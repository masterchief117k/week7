# ====================================================== IMPORTS FOR STREAMLIT APP ==========================================
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# ====================================================== MODEL TRAINING PART ==========================================
# ----- Load and prepare the data -----
data = pd.read_csv('Mall_Customers.csv')
data.columns=['cid','gender','age','annule_income','spending']
encoder = LabelEncoder()
data['gender'] = encoder.fit_transform(data['gender'])

# ----- Feature Scaling -----
x = data.iloc[:, :-1]
scaler = StandardScaler().fit(x)
x = scaler.transform(x)

# ----- Create target variable -----
data['HighSpender'] = (data['spending'] > data['spending'].median()).astype(int)
y = data['HighSpender']

# ----- Split the data -----
xtest, xtrain, ytest, ytrain = train_test_split(x, y, test_size=0.2, random_state=46)

# ----- Train the classifier -----
rf = RandomForestClassifier(n_estimators=100, random_state=46)
rf.fit(xtrain, ytrain)

# ====================================================== STREAMLIT UI START ==========================================
st.title("Mall Customer Spending Predictor")
st.write("Enter customer details to predict whether they are a high spender")

# ----- Create user inputs -----
cid = st.number_input('Customer ID', min_value=1, value=101)
gender_input = st.selectbox('Gender', options=['Male', 'Female'])
gender = 1 if gender_input == 'Male' else 0  # Note: LabelEncoder used 1 for Male, 0 for Female
age = st.slider('Age', min_value=10, max_value=100, value=30)
annule_income = st.slider('Annual Income (K$)', min_value=10, max_value=150, value=60)

# ----- Format the input for prediction -----
input_data = pd.DataFrame([[cid, gender, age, annule_income]],
                          columns=['cid', 'gender', 'age', 'annule_income'])

input_scaled = scaler.transform(input_data)

# ----- Run prediction -----
if st.button("Predict Spending Category"):
 prediction = rf.predict(input_scaled)
 result = "High Spender " if prediction[0] == 1 else "Low Spender "
 st.success(f"The customer is predicted to be a: **{result}**")
