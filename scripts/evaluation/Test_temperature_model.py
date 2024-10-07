#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dropout, TimeDistributed, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences
import plotly.graph_objs as go
import joblib

# Load and preprocess data
df = pd.read_csv('filtered_file_2327200051.csv')
df = df.drop(['N', 'SN'], axis=1)
df = df.rename(columns={'time': 'Time', 'BtryPckV': 'Voltage', 'BtryPckI': 'Current', 'PmpDty': 'PumpDutyCycle', 'LqdLvl': 'LiquidLevel'})
df['Date'] = pd.to_datetime(df['Time'], format='%d-%m-%Y %H:%M:%S')

# Handle missing values in 'SOH'
df['SOH'].fillna(method='ffill', inplace=True)
df['SOH'].fillna(method='bfill', inplace=True)

# Load and process environment temperature data
env_temp_df = pd.read_csv('12J990_2024.csv')
env_temp_df['Date'] = pd.to_datetime(env_temp_df['Date'], format="%d-%m-%Y %H:%M", errors='coerce')

# Merge environment temperature with the main dataset based on nearest time
df = df.sort_values('Date')
env_temp_df = env_temp_df.sort_values('Date')
df = pd.merge_asof(df, env_temp_df[['Date', 'Tx']], on='Date', direction='nearest')

# Define function to create sequences based on SOC changes and time gaps
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

# Filter for testing (January and February data)
test_data = df[(df['Date'].dt.month == 1) | (df['Date'].dt.month == 2)]
test_sequences = create_combined_sequences(test_data, soc_col='SOC', time_col='Date')

# Load saved scalers
scaler_X = joblib.load('scaler_X_with_MxTmp_winter.pkl')
scaler_y = joblib.load('scaler_y_with_MxTmp_winter.pkl')

# Normalize sequences for testing
features = ['Voltage', 'SOC', 'SOH', 'Current', 'PumpDutyCycle', 'LiquidLevel', 'Tx']
def normalize_sequences(sequences, features):
    normalized_sequences = []
    for seq in sequences:
        normalized_features = scaler_X.transform(seq[features])
        normalized_targets = scaler_y.transform(seq[['MxTmp']])
        normalized_seq = np.hstack((normalized_features, normalized_targets))
        normalized_sequences.append(normalized_seq)
    return normalized_sequences

normalized_test_sequences = normalize_sequences(test_sequences, features)

# Pad sequences for LSTM input
max_sequence_length = 5897
def pad_and_split_sequences(sequences):
    X = pad_sequences([seq[:, :-1] for seq in sequences], maxlen=max_sequence_length, dtype='float32', padding='post', truncating='post')
    y = pad_sequences([seq[:, -1:] for seq in sequences], maxlen=max_sequence_length, dtype='float32', padding='post', truncating='post')
    y = y.reshape((y.shape[0], y.shape[1], 1))  # Reshape to (num_sequences, sequence_length, 1)
    return X, y

X_test, y_test = pad_and_split_sequences(normalized_test_sequences)

# Load the trained model
model = load_model('MxTmp_model_winter_with_env_tmp.h5')

# Evaluate model on the test set
test_loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss}')

# Make predictions on the test set
y_pred = model.predict(X_test)

# Inverse transform predictions and actual values
y_pred_original = scaler_y.inverse_transform(y_pred.reshape(-1, 1))
y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1))

# Create a comparison dataframe
comparison_df = pd.DataFrame({
    'Actual': y_test_original.flatten(),
    'Predicted': y_pred_original.flatten()
})

# Calculate evaluation metrics
mse = mean_squared_error(y_test_original, y_pred_original)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_original, y_pred_original)

print(f'Mean Squared Error (MSE): {mse}')
print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'R-squared (R²): {r2}')

# Plot the comparison of actual vs predicted values
fig = go.Figure()
fig.add_trace(go.Scatter(y=comparison_df['Actual'], mode='lines', name='Actual AvgTmp', line=dict(color='red')))
fig.add_trace(go.Scatter(y=comparison_df['Predicted'], mode='lines', name='Predicted AvgTmp', line=dict(color='blue', dash='dash')))
fig.update_layout(title='Comparison of Actual vs Predicted AvgTmp', xaxis_title='Timestep Index', yaxis_title='AvgTmp', width=1000, height=600)
fig.write_html("comparison_actual_vs_predicted_AvgTmp_yearly_2.html")

print("Plot saved as 'comparison_actual_vs_predicted_AvgTmp_yearly_2.html'")
