# Split winter data (e.g., January to March)
winter_data = df[df['Month'].isin([1, 2, 3])]

# Split summer data (e.g., April to September)
summer_data = df[df['Month'].isin([4, 5, 6, 7, 9])]

# Choose a validation month (e.g., August)
validation_data = df[df['Month'] == 8]

# Apply sequence splitting based on SOC and time changes for winter, summer, and validation data
winter_sequences = create_combined_sequences(winter_data, time_col='Date', soc_threshold=5, time_threshold='1H')
summer_sequences = create_combined_sequences(summer_data, time_col='Date', soc_threshold=5, time_threshold='1H')
validation_sequences = create_combined_sequences(validation_data, time_col='Date', soc_threshold=5, time_threshold='1H')

# Define the feature columns
features = ['Voltage', 'SOC', 'SOH', 'Current', 'Tx', 'PumpDutyCycle', 'LiquidLevel', 
            'Current_change', 'Current_change_accel', 'Current_SOC_interaction', 'Month', 
            'DayOfYear', 'Warm_Cold_Season', 'SOC_Season_Interaction', 
            'Voltage_Season_Interaction', 'Current_Season_Interaction']

# Initialize the scalers for features and target (MxTmp)
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

# Combine winter, summer, validation, and test data for fitting the feature scaler
combined_data = pd.concat([pd.concat([seq[features] for seq in winter_sequences], ignore_index=True),
                           pd.concat([seq[features] for seq in summer_sequences], ignore_index=True),
                           pd.concat([seq[features] for seq in validation_sequences], ignore_index=True),
                           pd.concat([seq[features] for seq in test_sequences], ignore_index=True)])

# Fit the feature scaler (scaler_X) on the combined data
scaler_X.fit(combined_data)

# Fit the target scaler (MxTmp) on winter and summer data only (exclude validation and test for target scaling)
train_combined_target = pd.concat([pd.concat([seq[['MxTmp']] for seq in winter_sequences], ignore_index=True),
                                   pd.concat([seq[['MxTmp']] for seq in summer_sequences], ignore_index=True)])
scaler_y.fit(train_combined_target)

# Normalize the sequences for training, validation, and test sets
normalized_winter_sequences = [scaler_X.transform(seq[features]) for seq in winter_sequences]
normalized_summer_sequences = [scaler_X.transform(seq[features]) for seq in summer_sequences]
normalized_validation_sequences = [scaler_X.transform(seq[features]) for seq in validation_sequences]
normalized_test_sequences = [scaler_X.transform(seq[features]) for seq in test_sequences]

# Normalize the target (MxTmp) for winter, summer, validation, and test sequences
y_winter = [scaler_y.transform(seq[['MxTmp']]) for seq in winter_sequences]
y_summer = [scaler_y.transform(seq[['MxTmp']]) for seq in summer_sequences]
y_validation = scaler_y.transform(validation_data[['MxTmp']])  # Assuming 'validation_data' has the 'MxTmp' column
y_test = scaler_y.transform(test_df[['MxTmp']])  # Assuming 'test_df' has the 'MxTmp' column

# Flatten winter sequences for FNN training
X_winter = np.concatenate([seq for seq in normalized_winter_sequences])
y_winter = np.concatenate([target for target in y_winter])

# Flatten summer sequences for FNN fine-tuning
X_summer = np.concatenate([seq for seq in normalized_summer_sequences])
y_summer = np.concatenate([target for target in y_summer])

# Flatten validation sequences for model validation
X_validation = np.concatenate([seq for seq in normalized_validation_sequences])
y_validation = scaler_y.transform(validation_data[['MxTmp']])

# Ensure shapes are correct
print(f"X_winter shape: {X_winter.shape}, y_winter shape: {y_winter.shape}")
print(f"X_summer shape: {X_summer.shape}, y_summer shape: {y_summer.shape}")
print(f"X_validation shape: {X_validation.shape}, y_validation shape: {y_validation.shape}")

# Build and Compile the FNN Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers

# Define the FNN model with dropout, regularization, and a simplified structure
model = Sequential()
model.add(Dense(32, activation='relu', input_shape=(X_winter.shape[1],), kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.5))  # Increase dropout for regularization
model.add(Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.5))  # Additional dropout
model.add(Dense(1, kernel_regularizer=regularizers.l2(0.01)))  # Output layer

# Compile the model with Adam optimizer and mean squared error loss
optimizer = Adam(learning_rate=0.00001)
model.compile(optimizer=optimizer, loss='mse')

# Define Early Stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model on winter data and validate on the explicit validation set
history_winter = model.fit(X_winter, y_winter, validation_data=(X_validation, y_validation), 
                           epochs=100, callbacks=[early_stopping], verbose=1)

# Fine-tune the model on summer data and validate on the validation set
history_summer = model.fit(X_summer, y_summer, validation_data=(X_validation, y_validation), 
                           epochs=50, callbacks=[early_stopping], verbose=1)

# Predict on Test Data
X_test = np.concatenate([seq for seq in normalized_test_sequences])
test_predictions_scaled = model.predict(X_test)

# Inverse-transform the predictions back to the original scale (MxTmp)
test_predictions = scaler_y.inverse_transform(test_predictions_scaled)

# Model Summary
model.summary()

# Evaluate Model Performance (R-squared)
from sklearn.metrics import r2_score

r_squared = r2_score(y_test, test_predictions.flatten())
print(f'R-squared (R²) on the test set: {r_squared}')

# Plot Predicted vs Actual Values on Test Data
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(y_test, test_predictions.flatten(), color='blue', label='Predicted vs Actual')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Ideal Fit')
plt.title('Predicted vs Actual MxTmp')
plt.xlabel('Actual MxTmp')
plt.ylabel('Predicted MxTmp')
plt.legend()
plt.grid(True)
plt.show()

# Feature Importance using Permutation Importance
from sklearn.inspection import permutation_importance

# Assuming the model is already trained and X_test, y_test are available
result = permutation_importance(model, X_test, y_test, n_repeats=3, random_state=42, scoring='neg_mean_squared_error')

# Sort features by importance
sorted_idx = result.importances_mean.argsort()

# Plot feature importances
plt.barh(np.array(features)[sorted_idx], result.importances_mean[sorted_idx])
plt.xlabel("Permutation Importance")
plt.title("Feature Importance for FNN Model")
plt.show()
