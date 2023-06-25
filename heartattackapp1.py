import pickle

import numpy as np
import streamlit as st

st.title("Heart Attack Prediction app")
model = pickle.load(open("model1.pkl", "rb"))

age = st.number_input("Age")
sex = st.number_input("Sex")
cp = st.number_input("Chest Pain Type")
trtbps = st.number_input("Resting Blood Pressure (in mm Hg")
chol = st.number_input("Cholestoral (in mg/dl fetched via BMI sensor)")
fbs = st.number_input("Fasting Blood Sugar >120 mg/dl(1 = true; 0 = False")
restecg = st.number_input("Resting Electrocardiographic Results")
thalach = st.number_input("Maximum Heart Rate Achieved")
exng = st.number_input("exng")
oldpeak = st.number_input("oldpeak")
slp = st.number_input("slp")
caa = st.number_input("caa")
thall = st.number_input("thall")
btn = st.button("predict")

if btn:
    pred = model.predict(np.array([age,sex,cp,trtbps,chol,fbs,restecg,thalach,exng,oldpeak,slp,caa,thall]).reshape(1,-1))
    st.write(f"Chance of heart Attack: {pred}" )