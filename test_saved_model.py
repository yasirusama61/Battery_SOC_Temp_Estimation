#!/usr/bin/env python
# coding: utf-8

"""
Created by Yasir Usama
Script for testing a saved LSTM model on a new dataset.
"""

import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load and preprocess data
def load_and_preprocess_data(file_path):
    df = pd.read_parquet(file_path)
    df['SOC'] = df['SOC'] / 100
    df['Temperature'] = 25
    optimal_window = 10
    df['Voltage_Rolling_Mean'] = df['Voltage'].rolling(window=optimal_window).mean()
    df['Current_Rolling_Mean'] = df['Current'].rolling(window=optimal_window).mean()
    df['Voltage_Rate_of_Change'] = df['Voltage'].diff()
    df['Current_Rate_of_Change'] = df['Current'].diff()
    df = df.interpolate(method='linear', limit_direction='both')
    return df

# Resample data
def resample_data(df, factor):
    num_samples = len(df) // factor
    remainder = len(df) % factor
    resampled_data = {
        'Voltage': [],
        'Current': [],
        'Voltage_Rolling_Mean': [],
        'Current_Rolling_Mean': [],
        'Voltage_Rate_of_Change': [],
        'Current_Rate_of_Change': [],
        'Temperature': [],
        'SOC': []
    }

    for i in range(num_samples):
        start_index = i * factor
        end_index = start_index + factor
        for key in resampled_data.keys():
            resampled_data[key].append(df[key][start_index:end_index].mean())

    if remainder > 0:
        start_index = num_samples * factor
        for key in resampled_data.keys():
            resampled_data[key].append(df[key][start_index:].mean())

    return pd.DataFrame(resampled_data)

# Create sequences for LSTM
def create_sequences(df, features, sequence_length=90):
    X = df[features]
    y = df['SOC']
    X_sequences = []
    y_soc = []

    for i in range(len(X) - sequence_length):
        input_sequence = X.iloc[i:i+sequence_length].values
        target_soc = y.iloc[i+sequence_length]
        X_sequences.append(input_sequence)
        y_soc.append(target_soc)

    return np.array(X_sequences), np.array(y_soc)

# Normalize data
def normalize_data(X, scaler_X):
    num_sequences, sequence_length, num_features = X.shape
    X_reshaped = X.reshape(-1, num_features)
    X_normalized = scaler_X.transform(X_reshaped)
    return X_normalized.reshape(num_sequences, sequence_length, num_features)

# Plot results
def plot_results(y_true, y_pred, title='Comparison of Actual and Predicted SOC Values'):
    plt.figure(figsize=(10, 5))
    plt.plot(y_true, label='Actual SOC')
    plt.plot(y_pred, label='Predicted SOC', linestyle='--')
    plt.title(title)
    plt.xlabel('Sample Index')
    plt.ylabel('SOC')
    plt.legend()
    plt.show()

# Evaluate model
def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"R-squared (R²): {r2}")

# Main execution
if __name__ == "__main__":
    # Load the data
    data_file = 'single_cell_data.parquet'
    df = load_and_preprocess_data(data_file)
    
    # Resample the data
    resampling_factor = 100
    data_resampled = resample_data(df, resampling_factor)
    
    # Define features and create sequences
    features = ['Voltage', 'Current', 'Temperature', 'Voltage_Rolling_Mean', 'Current_Rolling_Mean', 'Voltage_Rate_of_Change', 'Current_Rate_of_Change']
    X_sequences, y_soc = create_sequences(data_resampled, features)
    
    # Load the scaler and normalize data
    scaler_X = joblib.load('scaler_X.pkl')
    X_sequences_normalized = normalize_data(X_sequences, scaler_X)
    
    # Load the model
    model = load_model('lstm_model_new_data_2.h5')
    
    # Predict SOC
    predicted_soc = model.predict(X_sequences_normalized)
    
    # Normalize target SOC
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    y_new = scaler_y.fit_transform(y_soc.reshape(-1, 1)).reshape(-1)
    
    # Plot results
    plot_results(y_new, predicted_soc)
    
    # Evaluate model
    evaluate_model(y_new, predicted_soc)


