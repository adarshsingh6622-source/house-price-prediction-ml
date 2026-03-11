import streamlit as st
import pickle
import numpy as np
import pandas as pd

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
features = pickle.load(open("features.pkl", "rb"))
st.title("House Price Prediction App")

area = st.number_input("Living Area")
bedroom = st.number_input("Bedrooms")
bathroom = st.number_input("Bathrooms")

if st.button("Predict"):
     # create empty dataframe with 277 features
    input_data = pd.DataFrame(np.zeros((1,len(features))),columns=features)

    # fill important features
    input_data["GrLivArea"] = area
    input_data["BedroomAbvGr"] = bedroom
    input_data["FullBath"] = bathroom

    # scaling
    input_scaled = scaler.transform(input_data)

    # prediction
    prediction = model.predict(input_scaled)

    st.success(f"Predicted House Price: {prediction[0]}")
   