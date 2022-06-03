import streamlit as st
import pandas as pd
import pickle
import make_model,preprocess
import numpy as np
model = pickle.load(open("CarPricePredictorModel.pkl", 'rb'))
# These works need to be done to create a model, as the model is now already created, I need to just use that created model.
df=pd.read_csv('CarPricePredictionDataset.csv')
df = pd.read_pickle("Cleaned_dataframe.pkl")
st.sidebar.title("Car Price Predictor")
names_list=df['name'].unique()
name=st.selectbox("Select Car name:",names_list)
company_list=df['company'].unique()
company=st.selectbox("Select Company name:",company_list)
year = st.number_input('Enter Year of manufacture:')
kms_driven = st.number_input('Enter Kilometres driven:') 
fuel_type_list=df['fuel_type'].unique()
fuel_type=st.selectbox("Select fuel_type:",fuel_type_list)
if st.button("Show Analysis"):
    st.write("Car Name:",name)
    st.write("Company Name:",company)
    st.write("Year Of manufacture:",year)
    st.write("Kilometres Driven:",kms_driven)
    st.title("Expected Price:")
    st.write("Expected Price:",model.predict(pd.DataFrame([[name,company,year,kms_driven,fuel_type]],columns=['name','company','year','kms_driven','fuel_type']))[0])
