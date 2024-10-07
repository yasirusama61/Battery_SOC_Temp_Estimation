import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
import plotly.graph_objs as go
import joblib
import plotly.io as pio

# Function Definitions
def create_combined_sequences(df, soc_col='SOC', time_col='Date', soc_threshold=5, time_threshold='1H'):
    """
    Function to create sequences based on SOC changes and time gaps.
    """
    sequences = []
    start_idx = 0
    for i in range(1, len(df)):
        time_gap = df[time_col].iloc[i] - df[time_col].iloc[i - 1]
        soc_change = abs(df[soc_col].iloc[i] - df[soc_col].iloc[i - 1])
        if time_gap >= pd.Timedelta(time_threshold) or soc_change >= soc_threshold:
            sequence = df.iloc[start_idx:i]
            sequences.append(sequence)
            start_idx = i
    if start_idx < len(df):
        sequences.append(df.iloc[start_idx:])
    return sequences

# Data Loading and Preprocessing
df = pd.read_csv('filtered_file1.csv')
df = df.drop(['N', 'SN'], axis=1)
df = df.rename(columns={'time': 'Time', 'BtryPckV': 'Voltage', 'BtryPckI': 'Current', 'PmpDty': 'PumpDutyCycle', 'LqdLvl': 'LiquidLevel'})
df['Date'] = pd.to_datetime(df['Time'], format='%d-%m-%Y %H:%M:%S')

# Handle missing values in 'SOH'
df['SOH'].fillna(method='ffill', inplace=True)
df['SOH'].fillna(method='bfill', inplace=True)

# Load environment temperature data and resample to match main data
env_temp_df = pd.read_csv('12J990_2024.csv')
env_temp_df['Date'] = pd.to_datetime(env_temp_df['Date'], format="%d-%m-%Y %H:%M", errors='coerce')
env_temp_df = env_temp_df.set_index('Date').resample('10S').interpolate()
df = pd.merge_asof(df.sort_values('Date'), env_temp_df[['Tx']].sort_values('Date'), on='Date', direction='nearest', tolerance=pd.Timedelta('1H'))

# Feature Engineering
df['SOC_change'] = df['SOC'].diff()
df['Tx_change'] = df['Tx'].diff()
df['Current_change'] = df['Current'].diff()
df['Current_change_accel'] = df['Current_change'].diff()
df['Current_SOC_interaction'] = df['Current'] * df['SOC']
df['Current_Tx_interaction'] = df['Current'] * df['Tx']
df['MxTmp_Tx_Interaction'] = df['MxTmp'] * df['Tx']
df['MxTmp_PumpDutyCycle_Interaction'] = df['MxTmp'] * df['PumpDutyCycle']
df['Warm_Cold_Season'] = df['Date'].dt.month.apply(lambda x: 1 if x in [4, 5, 6, 7, 8, 9] else 0)
df['SOC_Season_Interaction'] = df['SOC'] * df['Warm_Cold_Season']
df = df.dropna()

import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.io as pio

# Load the training data (test_df_1)
train_df = pd.read_csv('filtered_file_2327200051.csv')
train_df = train_df.drop(['N', 'SN'], axis=1)
train_df = train_df.rename(columns={'time': 'Time', 'BtryPckV': 'Voltage', 'BtryPckI': 'Current', 'PmpDty': 'PumpDutyCycle', 'LqdLvl': 'LiquidLevel'})
train_df['Date'] = pd.to_datetime(train_df['Time'], format='%d-%m-%Y %H:%M:%S')
train_df['SOH'].fillna(method='ffill', inplace=True)
train_df['SOH'].fillna(method='bfill', inplace=True)
train_df = train_df.sort_values('Date')

# Load the environment temperature data (env_temp_df_2)
env_temp_df_2 = pd.read_csv('env_temp_df_2.csv')  # Ensure this file exists
env_temp_df_2['Date'] = pd.to_datetime(env_temp_df_2['Date'], format='%d-%m-%Y %H:%M:%S')
env_temp_df_2 = env_temp_df_2.set_index('Date').resample('10S').interpolate()

# Process the test data (test_df_2)
test_df_2 = pd.read_csv('filtered_file_2327200056.csv')
test_df_2 = test_df_2.drop(['N', 'SN'], axis=1)
test_df_2 = test_df_2.rename(columns={'time': 'Time', 'BtryPckV': 'Voltage', 'BtryPckI': 'Current', 'PmpDty': 'PumpDutyCycle', 'LqdLvl': 'LiquidLevel'})
test_df_2['Date'] = pd.to_datetime(test_df_2['Time'], format='%d-%m-%Y %H:%M:%S')
test_df_2['SOH'].fillna(method='ffill', inplace=True)
test_df_2['SOH'].fillna(method='bfill', inplace=True)
test_df_2 = test_df_2.sort_values('Date')

# Split test_df_2 into two parts based on the year
test_df_2023_2 = test_df_2[test_df_2['Date'].dt.year == 2023]
test_df_2024_2 = test_df_2[test_df_2['Date'].dt.year == 2024]

# Merge the 2023 test data with env_temp_df_2
test_df_2023_2 = pd.merge_asof(test_df_2023_2, env_temp_df_2[['Tx']], on='Date', direction='nearest', tolerance=pd.Timedelta('1H'))

# Merge the 2024 test data with env_temp_df_2
test_df_2024_2 = pd.merge_asof(test_df_2024_2, env_temp_df_2[['Tx']], on='Date', direction='nearest', tolerance=pd.Timedelta('1H'))

# Concatenate the two parts back together for the test set
test_df_2 = pd.concat([test_df_2023_2, test_df_2024_2]).sort_values(by='Date').reset_index(drop=True)

# Feature engineering for the test set
test_df_2['Extreme_Discharge_Flag'] = (test_df_2['Current'] < -100).astype(int)
test_df_2['MxTmp_PumpDutyCycle_Interaction'] = test_df_2['MxTmp'] * test_df_2['PumpDutyCycle']
test_df_2['MxTmp_Tx_Interaction'] = test_df_2['MxTmp'] * test_df_2['Tx']
test_df_2['Current_SOC_interaction'] = test_df_2['Current'] * test_df_2['SOC']


# Plot the results (MxTmp and Tx)
fig = go.Figure()

# Plot MxTmp
fig.add_trace(go.Scatter(x=test_df_2['Date'], y=test_df_2['MxTmp'], mode='lines', name='MxTmp', line=dict(color='red')))

# Plot Tx (environment temperature)
fig.add_trace(go.Scatter(x=test_df_2['Date'], y=test_df_2['Tx'], mode='lines', name='Tx (Env Temp)', line=dict(color='green', dash='dash')))

# Update layout for the plot
fig.update_layout(
    title='MxTmp and Environment Temperature (Tx)',
    xaxis_title='Date',
    yaxis_title='Temperature (°C)',
    width=1000,
    height=600
)

# Save the plot as an HTML file
fig.write_html("temperature_comparison_plot_test.html")

# Verify the filtered test data after the date range exclusion
print(test_df_2.shape)
print(test_df_2.head())

# Train-Validation Split
train_data = df[(df['Date'] >= '2024-01-01') & (df['Date'] <= '2024-07-31')]
validation_data = df[(df['Date'] >= '2024-08-01') & (df['Date'] <= '2024-08-31')]


# Create Sequences
train_sequences = create_combined_sequences(train_data, soc_col='SOC', time_col='Date')
validation_sequences = create_combined_sequences(validation_data, soc_col='SOC', time_col='Date')
test_seuqences= create_combined_sequences(test_df_2, soc_col='SOC', time_col='Date')
# Normalize Data
features = ['Voltage', 'SOC', 'SOH', 'Current', 'PumpDutyCycle', 'LiquidLevel', 'Tx', 'MxTmp_Tx_Interaction', 'MxTmp_PumpDutyCycle_Interaction']
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
train_data_combined = pd.concat(train_sequences, ignore_index=True)
scaler_X.fit(train_data_combined[features])
scaler_y.fit(train_data_combined[['MxTmp']])

# Normalize Sequences
def normalize_sequences(sequences, features):
    normalized = []
    for seq in sequences:
        normalized_features = scaler_X.transform(seq[features])
        normalized_target = scaler_y.transform(seq[['MxTmp']])
        normalized_seq = np.hstack((normalized_features, normalized_target))
        normalized.append(normalized_seq)
    return normalized

normalized_train_sequences = normalize_sequences(train_sequences, features)
normalized_validation_sequences = normalize_sequences(validation_sequences, features)
normalized_test_sequences = normalize_sequences(test_seuqences, features)

# Flatten Data for Feedforward Neural Network (FNN)
def flatten_sequences(sequences):
    X = np.concatenate([seq[:, :-1] for seq in sequences])
    y = np.concatenate([seq[:, -1:] for seq in sequences])
    return X, y

X_train, y_train = flatten_sequences(normalized_train_sequences)
X_validation, y_validation = flatten_sequences(normalized_validation_sequences)
X_test, y_test = flatten_sequences(normalized_test_sequences)

# Model Definition and Training
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dropout(0.3))
model.add(Dense(1, kernel_regularizer=regularizers.l2(0.001)))
model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse')

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, verbose=1)

history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation), epochs=50, batch_size=16, callbacks=[early_stopping, checkpoint, reduce_lr])

# Model Evaluation
test_loss = model.evaluate(X_test, y_test)
print(f"Validation Loss: {test_loss}")

# Save the Trained Model
model.save('FNN_model.h5')

# Plot Loss During Training
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Prediction and Metrics
y_pred = model.predict(X_test)
y_pred_original = scaler_y.inverse_transform(y_pred)
y_test_original = scaler_y.inverse_transform(y_test)

mse = mean_squared_error(y_test_original, y_pred_original)
r2 = r2_score(y_test_original, y_pred_original)

print(f"Mean Squared Error: {mse}")
print(f"R-Squared: {r2}")

# Plot Predicted vs Actual
plt.figure(figsize=(10, 6))
plt.plot(test_df['Date'], y_test_original, label='Actual', color='blue')
plt.plot(test_df['Date'], y_pred_original, label='Predicted', color='orange', linestyle='--')
plt.xlabel('Date')
plt.ylabel('MxTmp (Temperature)')
plt.legend()
plt.title('Actual vs Predicted MxTmp on Validation Data')
plt.show()

# Save comparison plot as HTML
fig = go.Figure()
fig.add_trace(go.Scatter(x=test_df_2['Date'], y=y_test_original.flatten(), mode='lines', name='Actual MxTmp', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=test_df_2['Date'], y=y_pred_original.flatten(), mode='lines', name='Predicted MxTmp', line=dict(color='orange', dash='dash')))
fig.update_layout(title='Actual vs Predicted MxTmp', xaxis_title='Date', yaxis_title='MxTmp')
pio.write_html(fig, file='comparison_plot.html', auto_open=True)
