import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import joblib
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv('filtered_file1.csv')

# Calculate the average temperature and drop the original 'MxTmp' and 'MinTmp' columns
df['AvgTmp'] = df[['MxTmp', 'MinTmp']].mean(axis=1)
df = df.drop(columns=['MxTmp', 'MinTmp'])

# Drop unnecessary columns
df = df.drop(['N', 'SN'], axis=1)

# Rename columns for clarity
df = df.rename(columns={
    'time': 'Time',
    'BtryPckV': 'Voltage',
    'BtryPckI': 'Current',
    'PmpDty': 'PumpDutyCycle',
    'LqdLvl': 'LiquidLevel',
})

# Convert 'Time' to datetime format
df['Time'] = pd.to_datetime(df['Time'], format='%d-%m-%Y %H:%M:%S')

# Handle missing values in 'SOH' using forward and backward fill
df['SOH'].fillna(method='ffill', inplace=True)
df['SOH'].fillna(method='bfill', inplace=True)

# Filter the data for the months May to August
summer_data = df[df['Time'].dt.month.isin([5, 6, 7, 8])]

# Sort the data by 'Time' column
df = df.sort_values('Time')

# Define function to create sequences based on SOC changes and time gaps
def create_combined_sequences(df, soc_col='SOC', time_col='Time', soc_threshold=5, time_threshold='1H'):
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

# Example usage to split data into sequences based on SOC changes and time gaps
summer_sequences = create_combined_sequences(summer_data, soc_col='SOC', time_col='Time', soc_threshold=5, time_threshold='1H')

# Split data into training, validation, and test sets
validation_days = ['2024-08-27', '2024-08-28', '2024-08-29', '2024-08-30']
train_data = df[(df['Time'].dt.month.isin([6, 7])) | ((df['Time'].dt.month == 8) & (~df['Time'].dt.date.astype(str).isin(validation_days)))]
validation_data = df[(df['Time'].dt.month == 8) & (df['Time'].dt.date.astype(str).isin(validation_days))]
test_data = df[df['Time'].dt.month == 5]

# Apply sequence creation to the datasets
train_sequences = create_combined_sequences(train_data, soc_col='SOC', time_col='Time', soc_threshold=5, time_threshold='1H')
validation_sequences = create_combined_sequences(validation_data, soc_col='SOC', time_col='Time', soc_threshold=5, time_threshold='1H')
test_sequences = create_combined_sequences(test_data, soc_col='SOC', time_col='Time', soc_threshold=5, time_threshold='1H')

# Normalize the sequences for model training
features = ['Voltage', 'SOC', 'SOH', 'Current', 'PumpDutyCycle', 'LiquidLevel', 'Tx']  # Add 'Tx' if applicable

# Initialize scalers
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

# Fit scalers on the training data
train_combined = pd.concat(train_sequences, ignore_index=True)
scaler_X.fit(train_combined[features])
scaler_y.fit(train_combined[['AvgTmp']])

# Normalize sequences
def normalize_sequences(sequences, scaler_X, scaler_y, features):
    normalized_sequences = []
    for seq in sequences:
        normalized_features = scaler_X.transform(seq[features])
        normalized_target = scaler_y.transform(seq[['AvgTmp']])
        normalized_sequences.append(np.hstack((normalized_features, normalized_target)))
    return normalized_sequences

normalized_train_sequences = normalize_sequences(train_sequences, scaler_X, scaler_y, features)
normalized_validation_sequences = normalize_sequences(validation_sequences, scaler_X, scaler_y, features)
normalized_test_sequences = normalize_sequences(test_sequences, scaler_X, scaler_y, features)

# Save the scalers for future use
joblib.dump(scaler_X, 'scaler_X_with_Tx.pkl')
joblib.dump(scaler_y, 'scaler_y_AvgTmp.pkl')

# Data preparation for model input (padding)
max_sequence_length = max(len(seq) for seq in normalized_train_sequences)

def pad_sequence_data(sequences, max_len):
    X = pad_sequences([seq[:, :-1] for seq in sequences], maxlen=max_len, padding='post', truncating='post')
    y = pad_sequences([seq[:, -1] for seq in sequences], maxlen=max_len, padding='post', truncating='post')
    return X, y

X_train, y_train = pad_sequence_data(normalized_train_sequences, max_sequence_length)
X_validation, y_validation = pad_sequence_data(normalized_validation_sequences, max_sequence_length)
X_test, y_test = pad_sequence_data(normalized_test_sequences, max_sequence_length)

# Reshape target variables
y_train = y_train.reshape((y_train.shape[0], y_train.shape[1], 1))
y_validation = y_validation.reshape((y_validation.shape[0], y_validation.shape[1], 1))
y_test = y_test.reshape((y_test.shape[0], y_test.shape[1], 1))

# Model Training
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.3))
model.add(TimeDistributed(Dense(1)))

# Compile the model
optimizer = Adam(learning_rate=0.001, clipvalue=1.0)
model.compile(optimizer=optimizer, loss='mse')

# Define early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, 
                    epochs=50, 
                    batch_size=16, 
                    validation_data=(X_validation, y_validation), 
                    callbacks=[early_stopping], 
                    verbose=1)

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss During Training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Save the model
model.save('temperature_prediction_model.h5')

# Test the model
y_pred = model.predict(X_test)

# Inverse transform the predictions
y_pred_original = scaler_y.inverse_transform(y_pred.reshape(-1, 1))
y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1))

# Compare actual vs predicted values
comparison_df = pd.DataFrame({
    'Actual': y_test_original.flatten(),
    'Predicted': y_pred_original.flatten()
})

# Plot predictions vs actuals
plt.figure(figsize=(10, 6))
plt.plot(comparison_df['Actual'], label='Actual AvgTmp', color='red')
plt.plot(comparison_df['Predicted'], label='Predicted AvgTmp', color='blue', linestyle='--')
plt.title('Comparison of Actual vs Predicted AvgTmp')
plt.xlabel('Timestep Index')
plt.ylabel('AvgTmp')
plt.legend()
plt.show()

# Save the comparison plot as an HTML file using Plotly (optional)
import plotly.graph_objs as go
fig = go.Figure()
fig.add_trace(go.Scatter(y=comparison_df['Actual'], mode='lines', name='Actual AvgTmp', line=dict(color='red')))
fig.add_trace(go.Scatter(y=comparison_df['Predicted'], mode='lines', name='Predicted AvgTmp', line=dict(color='blue', dash='dash')))
fig.update_layout(title='Comparison of Actual vs Predicted AvgTmp', xaxis_title='Timestep Index', yaxis_title='AvgTmp', width=1000, height=600)
fig.write_html("comparison_actual_vs_predicted_AvgTmp.html")

# Evaluate the model
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test_original, y_pred_original)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_original, y_pred_original)

print(f'Mean Squared Error (MSE): {mse}')
print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'R-squared (R²): {r2}')
