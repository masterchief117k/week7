# ====================================================== IMPORTS FOR STREAMLIT APP ==========================================
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# --------------------- Load & Prepare Data --------------------- #
data = pd.read_csv('Mall_Customers.csv')

# Rename columns for clarity
data.columns = ['cid', 'gender', 'age', 'annual_income', 'spending']

# Encode gender: Female=0, Male=1
encoder = LabelEncoder()
data['gender'] = encoder.fit_transform(data['gender'])

# Create target: 1 if spending > median, else 0
data['HighSpender'] = (data['spending'] > data['spending'].median()).astype(int)

# Select features (drop cid and raw spending)
X = data[['gender', 'age', 'annual_income']]
y = data['HighSpender']

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=46
)

# Scale features
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train classifier
rf = RandomForestClassifier(n_estimators=100, random_state=46)
rf.fit(X_train_scaled, y_train)

# --------------------- Streamlit UI --------------------- #
st.title("Mall Customer Spending Predictor")
st.write("Enter customer details to predict whether they are a high spender")

# Optional: capture an ID just for display (not used by model)
cid = st.number_input("Customer ID (optional)", min_value=1, value=101)

gender_input = st.selectbox("Gender", ["Female", "Male"])
gender = 1 if gender_input == "Male" else 0

age = st.slider("Age", min_value=10, max_value=100, value=30)
annual_income = st.slider("Annual Income (K$)", min_value=10, max_value=150, value=60)

# When button is clicked, run prediction
if st.button("Predict Spending Category"):
    # Prepare and scale the input
    input_df = pd.DataFrame(
        [[gender, age, annual_income]],
        columns=['gender', 'age', 'annual_income']
    )
    input_scaled = scaler.transform(input_df)

    # Make prediction
    pred = rf.predict(input_scaled)[0]
    label = "High Spender" if pred == 1 else "Low Spender"

    st.success(f"Customer {cid} is predicted to be a: **{label}**")
