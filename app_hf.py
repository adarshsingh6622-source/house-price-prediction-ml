import gradio as gr
import pickle
import numpy as np
import pandas as pd

# Load model
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
features = pickle.load(open("features.pkl", "rb"))

def predict_price(area, bedroom, bathroom):
    
    input_data = pd.DataFrame(np.zeros((1, len(features))), columns=features)

    input_data["GrLivArea"] = area
    input_data["BedroomAbvGr"] = bedroom
    input_data["FullBath"] = bathroom

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    return f"Predicted House Price: {prediction[0]}"

iface = gr.Interface(
    fn=predict_price,
    inputs=[
        gr.Number(label="Living Area"),
        gr.Number(label="Bedrooms"),
        gr.Number(label="Bathrooms"),
    ],
    outputs="text",
    title="House Price Prediction",
    description="Enter house details to predict price"
)

iface.launch()