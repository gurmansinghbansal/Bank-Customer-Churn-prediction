import streamlit as st 
import numpy as np 
import joblib
st.title("Bank Customer Churn Prediction")
credit_score = st.number_input("Credit Score",300,900)
age = st.number_input("Age",18,100)
tenure = st.number_input("Tenure",0,10)
balance = st.number_input("Balance")
num_products = st.number_input("Number of Products", 1,4)
has_card = st.selectbox("Has Credit Card", [0,1])
active_member = st.selectbox("Active Member", [0,1])
salary = st.number_input("Estimated Salary")
germany = st.selectbox("Germany", [0,1])
spain = st.selectbox("Spain", [0,1])
male = st.selectbox("Male", [0,1])
if st.button("Predict Churn"):
 features = np.array([[credit_score, age, tenure, balance, num_products, has_card, active_member, salary, germany, spain, male]])
 model = joblib.load("rf_model.pkl")
 prediction = model.predict(features)
 if prediction[0] == 1:
  st.error("Customer likely to churn")
 else:
  st.success("Customer likely to stay")