#!/usr/bin/env python
# coding: utf-8

"""
Script for quantizing an LSTM model to uint8 format using TensorFlow Lite.

This script loads a trained LSTM model in HDF5 format, preprocesses test data, 
and quantizes the model to uint8 format. The quantized model is then saved 
as a TensorFlow Lite model file.

Dependencies:
- TensorFlow
- pandas
- numpy
- joblib

Usage:
1. Ensure the LSTM model, test data, and scaler are available in the specified paths.
2. Run the script to quantize the model and save it as a .tflite file.

Created by Usama Yasir Khan
"""

import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import joblib

def load_and_allocate_interpreter(model_path):
    # Load the quantized model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    # Verify input and output tensor details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("Input details:")
    for detail in input_details:
        print(f"  Name: {detail['name']}")
        print(f"  Shape: {detail['shape']}")
        print(f"  Data type: {detail['dtype']}")
        print(f"  Quantization: {detail['quantization']}")

    print("\nOutput details:")
    for detail in output_details:
        print(f"  Name: {detail['name']}")
        print(f"  Shape: {detail['shape']}")
        print(f"  Data type: {detail['dtype']}")
        print(f"  Quantization: {detail['quantization']}")

def load_and_preprocess_data(test_data_path, scaler_path):
    # Load the test dataset
    test_data = pd.read_csv(test_data_path)

    # Extract input features (excluding timestamp and state_of_charge)
    input_features = test_data[['voltage', 'current', 'temperature', 'Voltage_Rolling_Mean',
                                'Current_Rolling_Mean', 'Voltage_Rate_of_Change', 'Current_Rate_of_Change']].values

    # Load the scaler used during training
    scaler = joblib.load(scaler_path)

    # Preprocess the test data using the loaded scaler
    normalized_data = scaler.transform(input_features)
    return normalized_data

def reshape_data(data, time_steps=90):
    reshaped_data = []
    for i in range(len(data) - time_steps + 1):
        reshaped_data.append(data[i:i + time_steps])
    return np.array(reshaped_data)

def representative_dataset_gen(reshaped_data):
    for i in range(len(reshaped_data)):
        sample_data = reshaped_data[i:i+1].astype(np.float32)
        yield [sample_data]

def quantize_model(lstm_model, reshaped_data, output_path):
    # Convert and quantize the model using the representative dataset
    converter = tf.lite.TFLiteConverter.from_keras_model(lstm_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset_gen(reshaped_data)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    quantized_tflite_model = converter.convert()

    # Save the quantized model
    with open(output_path, 'wb') as f:
        f.write(quantized_tflite_model)

    print(f"Quantized model (UINT8) saved as '{output_path}'")

if __name__ == "__main__":
    # File paths
    lstm_model_path = 'lstm_model_new_data_2.h5'
    test_data_path = 'test_data.csv'
    scaler_path = 'Scaler_X.pkl'
    output_quantized_model_path = 'quantized_model_uint8.tflite'

    # Load the trained LSTM model
    lstm_model = load_model(lstm_model_path)

    # Load and preprocess test data
    normalized_data = load_and_preprocess_data(test_data_path, scaler_path)

    # Reshape data
    reshaped_data = reshape_data(normalized_data)

    # Quantize the model
    quantize_model(lstm_model, reshaped_data, output_quantized_model_path)

    # Verify the quantized model
    load_and_allocate_interpreter(output_quantized_model_path)
