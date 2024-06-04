import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import streamlit as st
import joblib

def numerical_preprocessing(d):
    train=pd.read_csv("train.csv")
    train.drop(columns=["loan_id"], inplace=True) # e drop becaue we dont want this numerical column
    for i in train.describe():
        ss = StandardScaler().fit(np.array(train[i]).reshape(-1,1))
        d[i] = ss.transform(np.array(d[i]).reshape(-1,1))[0] #it return the 2-d array

def categorical_preprocessing(d,education,self_employed):
    train=pd.read_csv("train.csv")
    train["education"] = train["education"].str.strip()
    train["self_employed"] = train["self_employed"].str.strip()
    train["loan_status"] = train["loan_status"].map({" Approved": 1, " Rejected": 0})
    if education == "Graduate":
        p_0_education = np.sum((train["education"] == "Graduate") & (train["loan_status"] == 0)) / np.sum(train["education"] == "Graduate")
        p_1_education = np.sum((train["education"] == "Graduate") & (train["loan_status"] == 1)) / np.sum(train["education"] == "Graduate")
        d["p_0_education"]=p_0_education
        d["p_1_education"]=p_1_education
    else:
        p_0_education = np.sum((train["education"] == "Not Graduate") & (train["loan_status"] == 0)) / np.sum(train["education"] == "Not Graduate")
        p_1_education = np.sum((train["education"] == "Not Graduate") & (train["loan_status"] == 1)) / np.sum(train["education"] == "Not Graduate")
        d["p_0_education"]=p_0_education
        d["p_1_education"]=p_1_education
    if self_employed=="Yes":
        p_0_self_employed = np.sum((train["self_employed"] == "Yes") & (train["loan_status"] == 0)) / np.sum(train["self_employed"] == "Yes")
        p_1_self_employed = np.sum((train["self_employed"] == "Yes") & (train["loan_status"] == 1)) / np.sum(train["self_employed"] == "Yes")
        d["p_0_self_employed"]=p_0_self_employed
        d["p_1_self_employed"]=p_1_self_employed
    else:
        p_0_self_employed = np.sum((train["self_employed"] == "No") & (train["loan_status"] == 0)) / np.sum(train["self_employed"] == "No")
        p_1_self_employed = np.sum((train["self_employed"] == "No") & (train["loan_status"] == 1)) / np.sum(train["self_employed"] == "No")
        d["p_0_self_employed"]=p_0_self_employed
        d["p_1_self_employed"]=p_1_self_employed
    d.pop("self_employed")
    d.pop("education")

def preprocessing(d,education,self_employed):
    numerical_preprocessing(d)
    categorical_preprocessing(d,education,self_employed)
    predict(d)

def predict(d):
    model=joblib.load("RFmodel.pkl")
    if model.predict(pd.DataFrame(d))[0] == 1:
        st.write("`The loan for this customer with the given criteria can be Approved")
    else:
        st.write("`The loan for this customer with the given criteria is Rejected")

def taking_inputs():
    st.write("""no_of_dependents,income_annum,loan_amount,loan_term,cibil_score,residential_assets_value,
                    commercial_assets_value,luury_assets_value,bank_assest_value,
                    education,self_employed""")
    d={}
    col1,col2,col3=st.columns(3)
    with col1:
        try:
            no_of_dependents=st.text_input("no_of_dependents 👇","0")
            d["no_of_dependents"]=float(no_of_dependents)
        except:
            st.write("please provide vaild integer input")
    with col2:
        try:
            income_annum=st.text_input("Annual_income 👇","0")
            d["income_annum"]=float(income_annum)
        except:
            st.write("please provide vaild integer input")
    with col3:
        try:
            loan_amount=st.text_input("loan_amount 👇","0")
            d["loan_amount"] =float(loan_amount)
        except:
            st.write("please provide vaild integer input")
    col1, col2, col3 = st.columns(3)
    with col1:
        try:
            loan_term = st.text_input("loan_term 👇", "0")
            d["loan_term"] = float(loan_term)
        except:
            st.write("please provide vaild integer input")
    with col2:
        try:
            cibil_score = st.text_input("cibil_score 👇", "0")
            d["cibil_score"] = float(cibil_score)
        except:
            st.write("please provide vaild integer input")
    with col3:
        try:
            residential_assets_value = st.text_input("residential_assets_value 👇", "0")
            d["residential_assets_value"] = float(residential_assets_value)
        except:
            st.write("please provide vaild integer input")
    col1, col2, col3 = st.columns(3)
    with col1:
        try:
            commercial_assets_value = st.text_input("commercial_assets_value 👇", "0")
            d["commercial_assets_value"] = float(commercial_assets_value)
        except:
            st.write("please provide vaild integer input")
    with col2:
        try:
            luxury_assets_value = st.text_input("luxury_assets_value 👇", "0")
            d["luxury_assets_value"] = float(luxury_assets_value)
        except:
            st.write("please provide vaild integer input")
    with col3:
        try:
            bank_asset_value = st.text_input("bank_asset_value 👇", "0")
            d["bank_asset_value"] = float(bank_asset_value)
        except:
            st.write("please provide vaild integer input")
    try:
        d["Combined_asset_value"]=d["residential_assets_value"]+d["commercial_assets_value"]+d["luxury_assets_value"]+d["bank_asset_value"]
    except:
        st.write("provide the correct inputs")

    col1,col2=st.columns(2)
    with col1:
        education=st.text_input("education 👇","Not Graduate")
        if education not in ["Not Graduate","Graduate"]:
            st.write("please provide vaild string input either Not Graduate or Graduate")
        else:
            d["education"]=education
    with col2:
        self_employed = st.text_input("self_employed 👇", "Yes")
        if self_employed not in ["Yes","No"]:
            st.write("please provide vaild string input either Yes or No")
        else:
            d["self_employed"]=self_employed
    if st.button("submit"):
        preprocessing(d,education,self_employed)
    else:
        st.write("press submit to get the response")




if __name__=="__main__":
    st.title("Loan_Approval_Detection")
    taking_inputs()
