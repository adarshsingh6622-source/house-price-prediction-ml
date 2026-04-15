import gradio as gr
import pandas as pd
import numpy as np
import pickle

# Load model artifacts
model = pickle.load(open("models/lgbm_model.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))
columns = pickle.load(open("models/columns.pkl", "rb"))

# Prediction function
def predict_price(GrLivArea, TotalBsmtSF, FullBath, HalfBath, YearBuilt, YrSold):

    # Create dataframe
    data = pd.DataFrame({
        'GrLivArea': [GrLivArea],
        'TotalBsmtSF': [TotalBsmtSF],
        'FullBath': [FullBath],
        'HalfBath': [HalfBath],
        'YearBuilt': [YearBuilt],
        'YrSold': [YrSold]
    })

    # Feature engineering (same as training)
    data['TotalSF'] = data['TotalBsmtSF'] + data['GrLivArea']
    data['TotalBath'] = data['FullBath'] + (0.5 * data['HalfBath'])
    data['HouseAge'] = data['YrSold'] - data['YearBuilt']

    # One-hot encoding
    data = pd.get_dummies(data)

    # Align columns with training
    data = data.reindex(columns=columns, fill_value=0)

    # Scaling
    data_scaled = scaler.transform(data)

    # Prediction
    prediction = np.expm1(model.predict(data_scaled)[0])
    price_inr = prediction * 83
    return f"🏠 Estimated House Price: ₹ {int(price_inr):,}"

# Gradio UI
interface = gr.Interface(
    fn=predict_price,
    inputs=[
        gr.Number(value=1500, label="Living Area (GrLivArea)"),
        gr.Number(value=800, label="Basement Area (TotalBsmtSF)"),
        gr.Number(value=2, label="Full Bathrooms"),
        gr.Number(value=1, label="Half Bathrooms"),
        gr.Number(value=2005, label="Year Built"),
        gr.Number(value=2011, label="Year Sold")
    ],
    outputs="text",
    title="🏠 House Price Prediction App",
    description="Enter house details to predict price (ML Model: LightGBM)"
)

if __name__ == "__main__":
    interface.launch()