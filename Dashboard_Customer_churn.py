import streamlit as st
import pandas as pd
import joblib 

st.set_page_config(page_title="Marvellous Sysytem Dashborad")

st.title("Project by sakshi")
st.header("Customer Churn Prediction")
st.write("Machine Learning Application Dashboard")

model=joblib.load("Marvellous_ML_Customer_Churn.pkl")

st.subheader("Enter customer details")

age=st.number_input("Customer Age",min_value=1)
tenure=st.number_input("Tenure(months)",min_value=0)
charges=st.number_input("Monthly charges",min_value=0)
calls=st.number_input("Support calls",min_value=0)

if st.button("Predict Churn"):
    new_customers=pd.DataFrame({
    "Age":[age],
    "Tenure":[tenure],
    "MonthlyCharges":[charges],
    "SupportCalls":[calls],
    })

    prediction=model.predict(new_customers)

    if prediction ==1:
        st.error("Customer is likely to churn")
    else:
        st.success("Customer is unlikely to churn")

st.divider()

st.caption("Marvellous Infosysytem Dashboard")


