import streamlit as st
import pandas as pd
import pickle
import numpy as np

model = pickle.load(open("models/lgbm_model.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))
columns = pickle.load(open("models/columns.pkl", "rb"))

st.title("🏠 House Price Prediction ")

area = st.number_input("Living Area (GrLivArea)", value = 1500)
basement_area = st.number_input("Basement Area (TotalBsmtSF)", value=800)
full_bath = st.number_input("Full Bathrooms", value=2)
half_bath = st.number_input("Half Bathrooms", value=1)
year_built = st.number_input("Year Built", value=2005)
year_sold = st.number_input("YrSold", value=2011)
                            

if st.button("Predict"):

    df = pd.DataFrame({
        'GrLivArea': [float(area)],
        'TotalBsmtSF': [float(basement_area)],
        'FullBath': [float(full_bath)],
        'HalfBath': [float(half_bath)],
        'YearBuilt': [float(year_built)],
        'YrSold': [float(year_sold)]
        
    })

    df['TotalSF'] = df['TotalBsmtSF'] + df['GrLivArea']
    df['TotalBath'] = df['FullBath'] + (0.5 * df['HalfBath'])
    df['HouseAge'] = df['YrSold'] - df['YearBuilt']
  
    df = pd.get_dummies(df)
    df = df.reindex(columns=columns, fill_value=0)

    df = scaler.transform(df)

    pred = model.predict(df)
    pred = np.expm1(pred)

   # USD → INR conversion
    price_inr = pred[0] * 83

    st.success(f"🏠 Estimated Price: ₹ {price_inr:,.0f}")