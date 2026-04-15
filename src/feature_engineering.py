import pandas as pd
def feature_engineering(df):

    df['TotalSF'] = df['TotalBsmtSF'] + df['GrLivArea']
    df['TotalBath'] = df['FullBath'] + (0.5 * df['HalfBath'])
    df['HouseAge'] = df['YrSold'] - df['YearBuilt']

    return df

   

    
