# utils/feature_engineering.py
import numpy as np
import pandas as pd

def feature_engineering_fn(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ['PaymentMethod','StreamingTV','StreamingMovies','OnlineSecurity',
                'OnlineBackup','DeviceProtection','TechSupport','MonthlyCharges','TotalCharges','tenure']:
        if col not in df.columns:
            df[col] = np.nan

    df['is_auto_payment'] = df['PaymentMethod'].apply(
        lambda x: 1 if isinstance(x, str) and 'automatic' in x.lower() else 0)
    df['tenure_per_monthly'] = df['tenure'] / (df['MonthlyCharges'].replace(0, np.nan) + 1e-6)
    df['avg_monthly_from_total'] = df['TotalCharges'] / (df['tenure'].replace(0, np.nan) + 1e-6)
    df['has_streaming'] = ((df['StreamingTV'] == 'Yes') | (df['StreamingMovies'] == 'Yes')).astype(int)
    df['has_support_services'] = (
        (df['OnlineSecurity'] == 'Yes') | (df['OnlineBackup'] == 'Yes') |
        (df['DeviceProtection'] == 'Yes') | (df['TechSupport'] == 'Yes')
    ).astype(int)
    return df
