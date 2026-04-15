import pandas as pd
import numpy as np

def load_data(path):
    df = pd.read_csv(path)
    return df

def clean_data(df):
    # Drop ID
    if 'Id' in df.columns:
        df = df.drop('Id', axis=1)

    # Separate numeric & categorical
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = df.select_dtypes(include=['object']).columns

    # Fill numeric with median
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    # Fill categorical with mode
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df