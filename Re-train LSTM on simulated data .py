#!/usr/bin/env python
# coding: utf-8

"""
Created by Usama Yasir Khan
Visualization and preprocessing script for LSTM training on SOC data.
"""

import os
import warnings
import pandas as pd
import plotly.graph_objects as go
from plotly_resampler import FigureResampler
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

# Load and preprocess data
def load_and_preprocess_data(data_file):
    df = pd.read_csv(data_file, parse_dates=['timestamp'])
    df['temperature'] = 25
    optimal_window = 10
    df['Voltage_Rolling_Mean'] = df['voltage'].rolling(window=optimal_window).mean()
    df['Current_Rolling_Mean'] = df['current'].rolling(window=optimal_window).mean()
    df['Voltage_Rate_of_Change'] = df['voltage'].diff()
    df['Current_Rate_of_Change'] = df['current'].diff()
    df.dropna(inplace=True)
    return df

# Update timestamps to current time
def update_timestamps(data_file, updated_file):
    df = pd.read_csv(data_file, parse_dates=['timestamp'])
    current_time = datetime.now()
    time_diff = current_time - df['timestamp'].iloc[0]
    df['timestamp'] = df['timestamp'] + time_diff
    df.to_csv(updated_file, index=False)
    print(f"Timestamps updated and saved to {updated_file}")

# Visualize data with resampling
def visualize_separately_with_resampling(df):
    fig_voltage_rm = FigureResampler(go.Figure())
    fig_voltage_rm.add_trace(go.Scattergl(x=df['timestamp'], y=df['Voltage_Rolling_Mean'], mode='lines', name='Voltage Rolling Mean'))
    fig_voltage_rm.update_layout(title='Voltage Rolling Mean Resampled Visualization', xaxis_title='Time', yaxis_title='Voltage Rolling Mean (V)')

    fig_current_rm = FigureResampler(go.Figure())
    fig_current_rm.add_trace(go.Scattergl(x=df['timestamp'], y=df['Current_Rolling_Mean'], mode='lines', name='Current Rolling Mean'))
    fig_current_rm.update_layout(title='Current Rolling Mean Resampled Visualization', xaxis_title='Time', yaxis_title='Current Rolling Mean (A)')

    fig_voltage_roc = FigureResampler(go.Figure())
    fig_voltage_roc.add_trace(go.Scattergl(x=df['timestamp'], y=df['Voltage_Rate_of_Change'], mode='lines', name='Voltage Rate of Change'))
    fig_voltage_roc.update_layout(title='Voltage Rate of Change Resampled Visualization', xaxis_title='Time', yaxis_title='Voltage Rate of Change (V/s)')

    fig_current_roc = FigureResampler(go.Figure())
    fig_current_roc.add_trace(go.Scattergl(x=df['timestamp'], y=df['Current_Rate_of_Change'], mode='lines', name='Current Rate of Change'))
    fig_current_roc.update_layout(title='Current Rate of Change Resampled Visualization', xaxis_title='Time', yaxis_title='Current Rate of Change (A/s)')
    
    fig_voltage_rm.show()
    fig_current_rm.show()
    fig_voltage_roc.show()
    fig_current_roc.show()

# Save all figures to a single HTML file
def save_all_figures_as_single_html(df, output_file):
    fig_soc = FigureResampler(go.Figure())
    fig_soc.add_trace(go.Scattergl(x=df['timestamp'], y=df['state_of_charge'], mode='lines', name='State of Charge'))
    fig_soc.update_layout(title='State of Charge (SOC) Resampled Visualization', xaxis_title='Time', yaxis_title='State of Charge')
    
    fig_current = FigureResampler(go.Figure())
    fig_current.add_trace(go.Scattergl(x=df['timestamp'], y=df['current'], mode='lines', name='Current'))
    fig_current.update_layout(title='Current Resampled Visualization', xaxis_title='Time', yaxis_title='Current (A)')
    
    fig_voltage = FigureResampler(go.Figure())
    fig_voltage.add_trace(go.Scattergl(x=df['timestamp'], y=df['voltage'], mode='lines', name='Voltage'))
    fig_voltage.update_layout(title='Voltage Resampled Visualization', xaxis_title='Time', yaxis_title='Voltage (V)')
    
    fig_voltage_rm = FigureResampler(go.Figure())
    fig_voltage_rm.add_trace(go.Scattergl(x=df['timestamp'], y=df['Voltage_Rolling_Mean'], mode='lines', name='Voltage Rolling Mean'))
    fig_voltage_rm.update_layout(title='Voltage Rolling Mean Resampled Visualization', xaxis_title='Time', yaxis_title='Voltage Rolling Mean (V)')
    
    fig_current_rm = FigureResampler(go.Figure())
    fig_current_rm.add_trace(go.Scattergl(x=df['timestamp'], y=df['Current_Rolling_Mean'], mode='lines', name='Current Rolling Mean'))
    fig_current_rm.update_layout(title='Current Rolling Mean Resampled Visualization', xaxis_title='Time', yaxis_title='Current Rolling Mean (A)')
    
    fig_voltage_roc = FigureResampler(go.Figure())
    fig_voltage_roc.add_trace(go.Scattergl(x=df['timestamp'], y=df['Voltage_Rate_of_Change'], mode='lines', name='Voltage Rate of Change'))
    fig_voltage_roc.update_layout(title='Voltage Rate of Change Resampled Visualization', xaxis_title='Time', yaxis_title='Voltage Rate of Change (V/s)')
    
    fig_current_roc = FigureResampler(go.Figure())
    fig_current_roc.add_trace(go.Scattergl(x=df['timestamp'], y=df['Current_Rate_of_Change'], mode='lines', name='Current Rate of Change'))
    fig_current_roc.update_layout(title='Current Rate of Change Resampled Visualization', xaxis_title='Time', yaxis_title='Current Rate of Change (A/s)')
    
    with open(output_file, 'w') as f:
        f.write(pio.to_html(fig_soc, full_html=False, include_plotlyjs='cdn'))
        f.write(pio.to_html(fig_current, full_html=False, include_plotlyjs='cdn'))
        f.write(pio.to_html(fig_voltage, full_html=False, include_plotlyjs='cdn'))
        f.write(pio.to_html(fig_voltage_rm, full_html=False, include_plotlyjs='cdn'))
        f.write(pio.to_html(fig_current_rm, full_html=False, include_plotlyjs='cdn'))
        f.write(pio.to_html(fig_voltage_roc, full_html=False, include_plotlyjs='cdn'))
        f.write(pio.to_html(fig_current_roc, full_html=False, include_plotlyjs='cdn'))

    print(f"All figures saved to {output_file}")

# Create sequences for LSTM
def create_sequences(df, features, sequence_length=90):
    X = df[features]
    y = df['state_of_charge']
    X_sequences = []
    y_soc = []
    for i in range(len(X) - sequence_length):
        input_sequence = X.iloc[i:i+sequence_length].values
        target_soc = y.iloc[i+sequence_length]
        X_sequences.append(input_sequence)
        y_soc.append(target_soc)
    return np.array(X_sequences), np.array(y_soc)

# Split and normalize data based on custom indices
def split_and_normalize_data_custom(X_sequences, y_soc, test_start_idx1, test_end_idx1, test_start_idx2, test_end_idx2):
    test_indices = np.concatenate([np.arange(test_start_idx1, test_end_idx1), np.arange(test_start_idx2, test_end_idx2)])
    train_indices = np.setdiff1d(np.arange(len(X_sequences)), test_indices)

    X_train = X_sequences[train_indices]
    y_train = y_soc[train_indices]

    X_test = X_sequences[test_indices]
    y_test = y_soc[test_indices]

    scaler_X = MinMaxScaler(feature_range=(0, 1))
    X_train_normalized = scaler_X.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_test_normalized = scaler_X.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

    scaler_y = MinMaxScaler(feature_range=(0, 1))
    y_train_normalized = scaler_y.fit_transform(y_train.reshape(-1, 1)).reshape(-1)
    y_test_normalized = scaler_y.transform(y_test.reshape(-1, 1)).reshape(-1)

    return X_train_normalized, X_test_normalized, y_train_normalized, y_test_normalized, scaler_X, scaler_y

# Build and train LSTM model
def build_and_train_model(X_train, y_train, input_shape, epochs=50, batch_size=250):
    model = Sequential()
    model.add(LSTM(30, input_shape=input_shape))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu', kernel_regularizer=l2()))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='sigmoid'))
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.0001)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, callbacks=[early_stopping, reduce_lr])
    return model, history

# Evaluate model
def evaluate_model(model, X_test, y_test, scaler_y):
    test_loss, test_mse = model.evaluate(X_test, y_test)
    print("Test Loss:", test_loss)
    print("Test MSE:", test_mse)
    y_test_pred_normalized = model.predict(X_test)
    y_test_pred = scaler_y.inverse_transform(y_test_pred_normalized)
    return y_test_pred

# Plot training history
def plot_training_history(history):
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Plot actual vs. predicted SOC values
def plot_actual_vs_predicted(y_test, y_test_pred):
    plt.figure(figsize=(10, 5))
    plt.plot(y_test, label='Actual SOC')
    plt.plot(y_test_pred, label='Predicted SOC', linestyle='--')
    plt.title('Comparison of Actual and Predicted SOC Values')
    plt.xlabel('Sample Index')
    plt.ylabel('SOC')
    plt.legend()
    plt.show()

# Plot prediction errors
def plot_prediction_errors(y_test, y_test_pred):
    errors = y_test - y_test_pred.flatten()
    plt.figure(figsize=(12, 6))
    plt.plot(errors, color='red', label='Prediction Errors')
    plt.axhline(y=0, color='black', linestyle='--')
    plt.xlabel('Sample Index')
    plt.ylabel('Prediction Error (Actual - Predicted SOC)')
    plt.title('Prediction Error Plot')
    plt.grid(True)
    plt.legend()
    plt.show()

# Main execution
if __name__ == "__main__":
    data_file = 'simulation_results_100percent.csv'
    updated_results_file = 'updated_simulation_results_100percent.csv'
    
    # Update timestamps and load data
    update_timestamps(data_file, updated_results_file)
    df = load_and_preprocess_data(updated_results_file)

    # Specify features for LSTM model
    features = ['voltage', 'current', 'temperature', 'Voltage_Rolling_Mean', 'Current_Rolling_Mean', 'Voltage_Rate_of_Change', 'Current_Rate_of_Change']

    # Create sequences for LSTM
    X_sequences, y_soc = create_sequences(df, features)

    # Define custom split indices for test and train sets
    test_start_idx1 = 0
    test_end_idx1 = 92738
    test_start_idx2 = 510000
    test_end_idx2 = len(X_sequences)

    # Split and normalize data
    X_train, X_test, y_train, y_test, scaler_X, scaler_y = split_and_normalize_data_custom(X_sequences, y_soc, test_start_idx1, test_end_idx1, test_start_idx2, test_end_idx2)
    
    # Save the scaler for future use
    joblib.dump(scaler_X, 'scaler_X.pkl')

    # Build and train LSTM model
    input_shape = (X_train.shape[1], X_train.shape[2])
    model, history = build_and_train_model(X_train, y_train, input_shape)

    # Evaluate model
    y_test_pred = evaluate_model(model, X_test, y_test, scaler_y)

    # Plot results
    plot_training_history(history)
    plot_actual_vs_predicted(y_test, y_test_pred)
    plot_prediction_errors(y_test, y_test_pred)

    # Save the model
    model.save('lstm_model_new_data_2.h5')
    print("Model saved to lstm_model_new_data_2.h5")
