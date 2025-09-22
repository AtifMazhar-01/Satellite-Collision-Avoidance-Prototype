# preprocess.py
import numpy as np
import pandas as pd

def mean_motion_to_altitude(mean_motion):
    """
    Convert mean motion (rev/day) to approximate altitude in km
    """
    mu = 398600.4418  # km^3/s^2
    n = mean_motion * 2 * np.pi / 86400.0
    a = (mu / (n**2)) ** (1/3)
    return a - 6371.0  # subtract Earth radius

def preprocess_df(df):
    """
    Input: df (pandas DataFrame) loaded from CSV
    Output: df (processed) and X (features DataFrame)
    Expected features for the model: Altitude_km, Inclination_rad, Eccentricity
    """
    df = df.copy()
    
    # ---------------- Altitude ----------------
    if 'Altitude_km' not in df.columns:
        if 'altitude' in df.columns:
            df['Altitude_km'] = pd.to_numeric(df['altitude'], errors='coerce')
        elif 'mean_motion' in df.columns:
            df['mean_motion'] = pd.to_numeric(df['mean_motion'], errors='coerce')
            df['Altitude_km'] = df['mean_motion'].apply(mean_motion_to_altitude)
        else:
            df['Altitude_km'] = 0.0  # fallback
    
    # ---------------- Inclination ----------------
    if 'Inclination_rad' not in df.columns:
        if 'Inclination' in df.columns:
            df['Inclination'] = pd.to_numeric(df['Inclination'], errors='coerce')
            df['Inclination_rad'] = np.deg2rad(df['Inclination'])
        elif 'inclination' in df.columns:
            df['inclination'] = pd.to_numeric(df['inclination'], errors='coerce')
            df['Inclination_rad'] = np.deg2rad(df['inclination'])
        else:
            df['Inclination_rad'] = 0.0  # fallback
    
    # ---------------- Eccentricity ----------------
    if 'Eccentricity' not in df.columns:
        if 'eccentricity' in df.columns:
            df['Eccentricity'] = pd.to_numeric(df['eccentricity'], errors='coerce')
        else:
            df['Eccentricity'] = 0.0  # fallback
    else:
        df['Eccentricity'] = pd.to_numeric(df['Eccentricity'], errors='coerce')
    
    # ---------------- Fill numeric NaNs ----------------
    numcols = df.select_dtypes(include='number').columns
    df[numcols] = df[numcols].fillna(df[numcols].mean())
    
    # ---------------- Build feature DataFrame ----------------
    features = ['Altitude_km', 'Inclination_rad', 'Eccentricity']
    for c in features:
        if c not in df.columns:
            df[c] = 0.0
    
    X = df[features].astype(float)
    
    return df, X

