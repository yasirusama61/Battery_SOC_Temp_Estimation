# -*- coding: utf-8 -*-
"""
ConvLSTM for SOC Prediction

This script implements a Convolutional LSTM (ConvLSTM) model to predict the 
State of Charge (SOC) based on battery data including voltage, current, and temperature.

Steps:
1. Load and preprocess data
2. Split into training, validation, and testing sets
3. Normalize the data
4. Build ConvLSTM model
5. Train and evaluate the model
6. Visualize the results
"""

# Mount Google Drive to access data
from google.colab import drive
drive.mount('/content/drive')

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load dataset
base_dir = 'data/batter_data.parquet'
output_file = os.path.join(base_dir, 'combined.parquet')
combined_data = pd.read_parquet(output_file)

# Define features and target
features = ['Voltage(V)', 'Current(A)', 'Temperature(Degree Celsius)']
X = combined_data[features]
y = combined_data['SOC']

# Sequence length for time-series input
sequence_length = 30
X_sequences, y_soc = [], []

# Create sequences of features and corresponding target values
for i in range(len(X) - sequence_length):
    input_sequence = X.iloc[i:i+sequence_length].values
    target_soc = y.iloc[i+sequence_length]
    X_sequences.append(input_sequence)
    y_soc.append(target_soc)

# Convert lists to numpy arrays
X_sequences = np.array(X_sequences)
y_soc = np.array(y_soc)

print(f"Shape of input sequences: {X_sequences.shape}")
print(f"Shape of SOC target: {y_soc.shape}")

# Split the data into training, validation, and testing sets (70/15/15)
train_idx = int(len(X_sequences) * 0.7)
val_idx = int(len(X_sequences) * 0.85)

X_train, X_val, X_test = X_sequences[:train_idx], X_sequences[train_idx:val_idx], X_sequences[val_idx:]
y_train, y_val, y_test = y_soc[:train_idx], y_soc[train_idx:val_idx], y_soc[val_idx:]

print(f"Training data shape: {X_train.shape}")
print(f"Validation data shape: {X_val.shape}")
print(f"Testing data shape: {X_test.shape}")

# Normalize the input data using MinMaxScaler
scaler_X = MinMaxScaler(feature_range=(0, 1))
X_train_normalized = scaler_X.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
X_val_normalized = scaler_X.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
X_test_normalized = scaler_X.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

print(f"Normalized X_train shape: {X_train_normalized.shape}")

# Plot normalized data for training, validation, and testing sets
def plot_normalized_data(X_train_norm, X_val_norm, X_test_norm):
    train_len = X_train_norm.shape[0]
    val_len = X_val_norm.shape[0]
    test_len = X_test_norm.shape[0]

    train_indices = np.repeat(np.arange(train_len), X_train_norm.shape[1])
    val_indices = np.repeat(np.arange(val_len), X_val_norm.shape[1]) + train_len
    test_indices = np.repeat(np.arange(test_len), X_test_norm.shape[1]) + train_len + val_len

    groups = [0, 1, 2]  # Voltage, Current, Temperature indices
    group_labels = ['Voltage(V)', 'Current(A)', 'Temperature(Degree Celsius)']
    
    plt.figure(figsize=(12, 14))
    for i, group in enumerate(groups):
        plt.subplot(len(groups), 1, i+1)
        plt.plot(train_indices, X_train_norm[:, :, group].flatten(), label='Training Data', color='blue')
        plt.plot(val_indices, X_val_norm[:, :, group].flatten(), label='Validation Data', color='orange')
        plt.plot(test_indices, X_test_norm[:, :, group].flatten(), label='Testing Data', color='red')
        plt.title(group_labels[i])
        if i == len(groups) - 1:
            plt.xlabel('Time Steps')
        if i == 0:
            plt.legend()
    plt.tight_layout()
    plt.show()

plot_normalized_data(X_train_normalized, X_val_normalized, X_test_normalized)

# Reshape for ConvLSTM2D input (samples, time steps, rows, cols, channels)
X_train_conv = X_train_normalized.reshape((X_train_normalized.shape[0], X_train_normalized.shape[1], 1, 1, X_train_normalized.shape[2]))
X_val_conv = X_val_normalized.reshape((X_val_normalized.shape[0], X_val_normalized.shape[1], 1, 1, X_val_normalized.shape[2]))
X_test_conv = X_test_normalized.reshape((X_test_normalized.shape[0], X_test_normalized.shape[1], 1, 1, X_test_normalized.shape[2]))

print(f"Reshaped X_train_conv: {X_train_conv.shape}")

# Build ConvLSTM model
model = Sequential([
    ConvLSTM2D(filters=64, kernel_size=(1, 2), activation='relu', input_shape=(X_train_conv.shape[1], 1, 1, X_train_conv.shape[4]), return_sequences=True, padding='same'),
    Dropout(0.3),
    ConvLSTM2D(filters=32, kernel_size=(1, 2), activation='relu', return_sequences=False, padding='same'),
    Flatten(),
    Dense(50, activation='relu', kernel_regularizer='l2'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
model.summary()

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
model_checkpoint = ModelCheckpoint('/content/drive/My Drive/convolstm_model.h5', save_best_only=True, monitor='val_loss', mode='min')

# Train the model
history = model.fit(X_train_conv, y_train.reshape(-1, 1), epochs=100, batch_size=250,
                    validation_data=(X_val_conv, y_val.reshape(-1, 1)),
                    callbacks=[early_stopping, reduce_lr, model_checkpoint])

# Plot training history
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Evaluate the model
model = load_model('/content/drive/My Drive/convolstm_model.h5')
y_pred = model.predict(X_test_conv).reshape(-1)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Test MSE: {mse}")
print(f"Test MAE: {mae}")
print(f"Test R^2: {r2}")

# Plot actual vs predicted SOC values
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Actual SOC')
plt.plot(y_pred, label='Predicted SOC')
plt.title('Actual vs. Predicted SOC Values')
plt.xlabel('Sample Index')
plt.ylabel('SOC')
plt.legend()
plt.show()
