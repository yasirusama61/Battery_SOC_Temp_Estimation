#!/usr/bin/env python
# coding: utf-8
# Author: Usama Yasir Khan

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objs as go

# Step 1: Load and clean the data
df = pd.read_csv('example_filtered_file.csv')

# Drop unnecessary columns
df = df.drop(['N', 'SN', 'Lon', 'Lat'], axis=1)

# Rename columns for clarity
df = df.rename(columns={
    'time': 'Time',
    'BtryPckV': 'Voltage',
    'BtryPckI': 'Current',
    'PmpDty': 'PumpDutyCycle',
    'LqdLvl': 'LiquidLevel'
})

# Convert time column to datetime
df['Date'] = pd.to_datetime(df['Time'], format='%d-%m-%Y %H:%M:%S')

# Handle missing values in SOH
df['SOH'].fillna(method='ffill', inplace=True)
df['SOH'].fillna(method='bfill', inplace=True)

# Step 2: Load environment temperature data
env_temp_df = pd.read_csv('example_env_temperature.csv')
env_temp_df['Date'] = pd.to_datetime(env_temp_df['Date'], format="%d-%m-%Y %H:%M", errors='coerce')

# Filter data for relevant months
df = df[df['Date'].dt.month.isin([5, 6, 7, 8])]
env_temp_df = env_temp_df[env_temp_df['Date'].dt.month.isin([5, 6, 7, 8])]

# Sort data by Date
df = df.sort_values('Date')
env_temp_df = env_temp_df.sort_values('Date')

# Merge with environment temperature
merged_df = pd.merge_asof(df, env_temp_df[['Date', 'Tx']], on='Date', direction='nearest')

# Step 3: Create sequences based on SOC and time gaps
def create_combined_sequences(df, soc_col='SOC', time_col='Date', soc_threshold=5, time_threshold='1H'):
    sequences = []
    start_idx = 0
    for i in range(1, len(df)):
        time_gap = df[time_col].iloc[i] - df[time_col].iloc[i-1]
        soc_change = abs(df[soc_col].iloc[i] - df[soc_col].iloc[i-1])
        if time_gap >= pd.Timedelta(time_threshold) or soc_change >= soc_threshold:
            sequences.append(df.iloc[start_idx:i])
            start_idx = i
    if start_idx < len(df):
        sequences.append(df.iloc[start_idx:])
    return sequences

# Step 4: Normalize the data
features = ['Voltage', 'SOC', 'SOH', 'Current', 'PumpDutyCycle', 'LiquidLevel', 'Tx']
scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))

train_data_combined = pd.concat(create_combined_sequences(merged_df), ignore_index=True)
scaler_X.fit(train_data_combined[features])
scaler_y.fit(train_data_combined[['MxTmp', 'MinTmp']])

# Step 5: Normalize training, validation, and testing data
def normalize_sequences(sequences, features, scaler_X, scaler_y):
    normalized_sequences = []
    for seq in sequences:
        normalized_features = scaler_X.transform(seq[features])
        normalized_targets = scaler_y.transform(seq[['MxTmp', 'MinTmp']])
        normalized_seq = np.hstack((normalized_features, normalized_targets))
        normalized_sequences.append(normalized_seq)
    return normalized_sequences

train_sequences = create_combined_sequences(merged_df)
normalized_train_sequences = normalize_sequences(train_sequences, features, scaler_X, scaler_y)

# Step 6: Pad sequences
max_sequence_length = max(len(seq) for seq in normalized_train_sequences)

X_train = pad_sequences([seq[:, :-2] for seq in normalized_train_sequences], maxlen=max_sequence_length, dtype='float32', padding='post', truncating='post')
y_train = pad_sequences([seq[:, -2:] for seq in normalized_train_sequences], maxlen=max_sequence_length, dtype='float32', padding='post', truncating='post')

y_train = y_train.reshape((y_train.shape[0], y_train.shape[1], 2))

# Step 7: Prepare the LSTM model
model = Sequential()
model.add(LSTM(25, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.3))
model.add(TimeDistributed(Dense(2)))

# Compile model
optimizer = Adam(learning_rate=0.001, clipvalue=1.0)
model.compile(optimizer=optimizer, loss='mse')

# Step 8: Train the model
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train, 
    epochs=50, batch_size=16, 
    validation_data=(X_train, y_train),  # Use train data for simplicity
    callbacks=[early_stopping], verbose=1
)

# Step 9: Evaluate and predict
test_loss = model.evaluate(X_train, y_train)
print(f'Test Loss: {test_loss}')

y_pred = model.predict(X_train)

# Step 10: Post-processing and visualization
y_train_reshaped = y_train.reshape(-1, 2)
y_pred_reshaped = y_pred.reshape(-1, 2)

y_pred_original = scaler_y.inverse_transform(y_pred_reshaped)
y_train_original = scaler_y.inverse_transform(y_train_reshaped)

# Visualization
comparison_df = pd.DataFrame({
    'Actual MxTmp': y_train_original[:, 0],
    'Predicted MxTmp': y_pred_original[:, 0],
    'Actual MinTmp': y_train_original[:, 1],
    'Predicted MinTmp': y_pred_original[:, 1]
})

fig = go.Figure()
fig.add_trace(go.Scatter(y=comparison_df['Actual MxTmp'], mode='lines', name='Actual MxTmp', line=dict(color='red')))
fig.add_trace(go.Scatter(y=comparison_df['Predicted MxTmp'], mode='lines', name='Predicted MxTmp', line=dict(color='blue', dash='dash')))
fig.add_trace(go.Scatter(y=comparison_df['Actual MinTmp'], mode='lines', name='Actual MinTmp', line=dict(color='green')))
fig.add_trace(go.Scatter(y=comparison_df['Predicted MinTmp'], mode='lines', name='Predicted MinTmp', line=dict(color='orange', dash='dash')))

fig.update_layout(
    title='Comparison of Actual vs Predicted for MxTmp and MinTmp',
    xaxis_title='Timestep Index',
    yaxis_title='Temperature (MxTmp / MinTmp)',
    autosize=True,
    height=800
)
fig.write_html("example_comparison_plot.html")

# Save model
model.save('example_temp_model.h5')
