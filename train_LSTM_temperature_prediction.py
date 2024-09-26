#!/usr/bin/env python
# coding: utf-8

"""
Project: Temperature Prediction Using LSTM for Immersion-Cooled Battery Packs

Author: Usama Yasir Khan
GitHub Repository: https://github.com/XING-Mobility/lstm-train

Description:
This script is developed for predicting the maximum (MxTmp) and minimum (MinTmp) busbar temperatures in an immersion-cooled battery pack, using key parameters such as voltage, current, SOC, SOH, pump duty cycle, and environmental temperature (Tx). 
The model leverages a sequence-based LSTM neural network to provide accurate temperature predictions that assist in managing the thermal performance of the battery pack in electric vehicles.

Key Components:
- Data preprocessing: Cleaning, filtering, and handling missing values.
- Sequence creation: Sequences are generated based on SOC and time gaps to capture changes effectively.
- Normalization and padding: Ensures consistent input size for the LSTM model.
- Model evaluation: Metrics such as MSE, RMSE, and R² are used to measure model performance.
- Visualization: Plots the comparison of actual vs. predicted temperatures for easier analysis.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, TimeDistributed, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences
import plotly.graph_objs as go

# Function to preprocess the dataset
def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df = df.drop(['N', 'SN', 'Lon', 'Lat'], axis=1)
    df = df.rename(columns={'time': 'Time', 'BtryPckV': 'Voltage', 'BtryPckI': 'Current', 'PmpDty': 'PumpDutyCycle', 'LqdLvl': 'LiquidLevel'})
    df['Date'] = pd.to_datetime(df['Time'], format='%d-%m-%Y %H:%M:%S')
    
    # Handle missing SOH values
    df['SOH'].fillna(method='ffill', inplace=True)
    df['SOH'].fillna(method='bfill', inplace=True)
    
    return df

# Function to load and filter environmental temperature data
def load_environment_data(env_file):
    env_df = pd.read_csv(env_file)
    env_df['Date'] = pd.to_datetime(env_df['Date'], format="%d-%m-%Y %H:%M", errors='coerce')
    return env_df

# Merge datasets based on nearest time
def merge_datasets(df, env_df):
    return pd.merge_asof(df.sort_values('Date'), env_df[['Date', 'Tx']].sort_values('Date'), on='Date', direction='nearest')

# Create sequences based on SOC and time gaps
def create_combined_sequences(df, soc_col='SOC', time_col='Time', soc_threshold=5, time_threshold='1H'):
    sequences = []
    start_idx = 0
    for i in range(1, len(df)):
        time_gap = df[time_col].iloc[i] - df[time_col].iloc[i - 1]
        soc_change = abs(df[soc_col].iloc[i] - df[soc_col].iloc[i - 1])
        if time_gap >= pd.Timedelta(time_threshold) or soc_change >= soc_threshold:
            sequences.append(df.iloc[start_idx:i])
            start_idx = i
    if start_idx < len(df):
        sequences.append(df.iloc[start_idx:])
    return sequences

# Normalize data using MinMaxScaler
def normalize_sequences(sequences, scaler_X, scaler_y, features):
    normalized_sequences = []
    for seq in sequences:
        normalized_features = scaler_X.transform(seq[features])
        normalized_targets = scaler_y.transform(seq[['MxTmp', 'MinTmp']])
        normalized_seq = np.hstack((normalized_features, normalized_targets))
        normalized_sequences.append(normalized_seq)
    return normalized_sequences

# Pad sequences to match the input size for LSTM
def pad_and_split_sequences(sequences, maxlen):
    X = pad_sequences([seq[:, :-2] for seq in sequences], maxlen=maxlen, dtype='float32', padding='post', truncating='post')
    y = pad_sequences([seq[:, -2:] for seq in sequences], maxlen=maxlen, dtype='float32', padding='post', truncating='post')
    return X, y.reshape((y.shape[0], y.shape[1], 2))

# Build LSTM model
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(LSTM(50, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(TimeDistributed(Dense(2)))  # Predicting 2 targets: MxTmp and MinTmp
    model.compile(optimizer='adam', loss='mse')
    return model

# Train the model
def train_model(model, X_train, y_train, X_val, y_val):
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=16, callbacks=[early_stopping], verbose=1)
    return model, history

# Evaluate and calculate metrics
def evaluate_model(model, X_test, y_test, scaler_y):
    test_loss = model.evaluate(X_test, y_test)
    y_pred = model.predict(X_test)
    
    y_pred_reshaped = y_pred.reshape(-1, 2)
    y_test_reshaped = y_test.reshape(-1, 2)
    
    # Inverse transform predictions
    y_pred_original = scaler_y.inverse_transform(y_pred_reshaped)
    y_test_original = scaler_y.inverse_transform(y_test_reshaped)
    
    mse_mx = mean_squared_error(y_test_original[:, 0], y_pred_original[:, 0])
    rmse_mx = np.sqrt(mse_mx)
    r2_mx = r2_score(y_test_original[:, 0], y_pred_original[:, 0])

    mse_min = mean_squared_error(y_test_original[:, 1], y_pred_original[:, 1])
    rmse_min = np.sqrt(mse_min)
    r2_min = r2_score(y_test_original[:, 1], y_pred_original[:, 1])

    print(f"Test Loss: {test_loss}")
    print(f"Metrics for MxTmp: MSE = {mse_mx}, RMSE = {rmse_mx}, R² = {r2_mx}")
    print(f"Metrics for MinTmp: MSE = {mse_min}, RMSE = {rmse_min}, R² = {r2_min}")

    return y_test_original, y_pred_original

# Plot the comparison of actual vs predicted results
def plot_comparison(y_test_original, y_pred_original):
    comparison_df = pd.DataFrame({
        'Actual MxTmp': y_test_original[:, 0], 
        'Predicted MxTmp': y_pred_original[:, 0],
        'Actual MinTmp': y_test_original[:, 1], 
        'Predicted MinTmp': y_pred_original[:, 1]
    })

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=comparison_df['Actual MxTmp'], mode='lines', name='Actual MxTmp', line=dict(color='red')))
    fig.add_trace(go.Scatter(y=comparison_df['Predicted MxTmp'], mode='lines', name='Predicted MxTmp', line=dict(color='blue', dash='dash')))
    fig.add_trace(go.Scatter(y=comparison_df['Actual MinTmp'], mode='lines', name='Actual MinTmp', line=dict(color='green')))
    fig.add_trace(go.Scatter(y=comparison_df['Predicted MinTmp'], mode='lines', name='Predicted MinTmp', line=dict(color='orange', dash='dash')))
    fig.update_layout(title='Comparison of Actual vs Predicted for MxTmp and MinTmp', xaxis_title='Timestep Index', yaxis_title='Temperature (MxTmp / MinTmp)')
    fig.write_html("comparison_plot_fullscreen.html")

# Main function to run the pipeline
if __name__ == "__main__":
    # Preprocess and merge datasets
    df = preprocess_data('filtered_file1.csv')
    env_temp_df = load_environment_data('12J990_2024.csv')
    df = merge_datasets(df, env_temp_df)
    
    # Split data based on months: Jan-July for training, August for validation, May for testing
    train_data = df[df['Date'].dt.month.isin([1, 2, 3, 4, 6, 7])]
    validation_data = df[df['Date'].dt.month == 8]
    test_data = df[df['Date'].dt.month == 5]
    
    # Create sequences
    train_sequences = create_combined_sequences(train_data, soc_col='SOC', time_col='Date')
    validation_sequences = create_combined_sequences(validation_data, soc_col='SOC', time_col='Date')
    test_sequences = create_combined_sequences(test_data, soc_col='SOC', time_col='Date')
    
    # Normalize data
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    features = ['Voltage', 'SOC', 'SOH', 'Current', 'PumpDutyCycle', 'LiquidLevel', 'Tx']
    train_data_combined = pd.concat(train_sequences, ignore_index=True)
    
    scaler_X.fit(train_data_combined[features])
    scaler_y.fit(train_data_combined[['MxTmp', 'MinTmp']])
    
    normalized_train_sequences = normalize_sequences(train_sequences, scaler_X, scaler_y, features)
    normalized_validation_sequences = normalize_sequences(validation_sequences, scaler_X, scaler_y, features)
    normalized_test_sequences = normalize_sequences(test_sequences, scaler_X, scaler_y, features)
    
    # Pad sequences for model input
    max_sequence_length = max(len(seq) for seq in normalized_train_sequences)
    X_train, y_train = pad_and_split_sequences(normalized_train_sequences, max_sequence_length)
    X_val, y_val = pad_and_split_sequences(normalized_validation_sequences, max_sequence_length)
    X_test, y_test = pad_and_split_sequences(normalized_test_sequences, max_sequence_length)
    
    # Build and train the LSTM model
    model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    
    model, history = train_model(model, X_train, y_train, X_val, y_val)
    
    # Evaluate the model on test data (May data)
    y_test_original, y_pred_original = evaluate_model(model, X_test, y_test, scaler_y)
    
    # Plot the comparison
    plot_comparison(y_test_original, y_pred_original)

    # Save the model
    model.save('temp_model_seasonal_with_env_tmp.h5')
