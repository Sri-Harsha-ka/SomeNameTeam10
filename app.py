import streamlit as st
from src.prediction import model_preds

st.title("Some Model Prediction")
st.write("This is some model")

age = st.number_input("Enter your age : ")
annual_income_lpa = st.number_input("Enter ur annual somehthing : ")
policy_term_year = st.number_input("Enter policy")
Sum_Assured_Lakhs = st.number_input("Enter somthing high val : ")

if st.button("predict"):
    model = model_preds()
    res = model.pred(Age= age, Annual_Income_LPA=annual_income_lpa , Policy_Term_Years=policy_term_year , Sum_Assured_Lakhs=Sum_Assured_Lakhs)
    st.success(res)