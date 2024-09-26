# LSTM Train

This repository contains code for training Long Short-Term Memory (LSTM) models on state of charge (SOC) data from battery simulations. 
The code includes data preprocessing, visualization, and model training and evaluation.

## Table of Contents

- [Introduction](#introduction)
- [Data Preprocessing](#data-preprocessing)
- [Visualization](#visualization)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [How to Run](#how-to-run)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The goal of this project is to train an LSTM model to predict the state of charge (SOC) of a battery based on various input features such as voltage, current, temperature, and their rolling means and rates of change.

## Data Preprocessing

The data preprocessing steps include:
- Loading and updating timestamps.
- Calculating rolling means and rates of change for voltage and current.
- Creating sequences for LSTM input.
- Splitting the data into training and testing sets based on specified indices.
- Normalizing the input features and target variable.

## Visualization

The code includes functions to visualize the data using Plotly and Plotly Resampler. The visualizations include:
- Voltage Rolling Mean
- Current Rolling Mean
- Voltage Rate of Change
- Current Rate of Change

## Model Training

The LSTM model is defined and trained using TensorFlow/Keras. The model architecture includes:
- LSTM layers
- Dense layers
- Dropout layers for regularization

The training process uses early stopping and learning rate reduction callbacks to optimize the training.

## Evaluation

The model is evaluated on the test set using mean squared error (MSE) and other metrics. The evaluation includes:
- Plotting the training and validation loss
- Comparing actual vs. predicted SOC values
- Plotting prediction errors

## How to Run
1. **Clone the repository:**
   Open your terminal and run the following command to clone the repository and navigate into it:
   ```sh
   git clone https://github.com/XING-Mobility/lstm-train.git
   cd lstm-train
2. Install the required dependencies:
    Ensure you have pip installed.
    If pip is not installed, you can install it by following the instructions here.
    Then, run the following command to install all the required dependencies listed in the requirements.txt file:
   pip install -r requirements.txt
3. The requirements.txt file should contain the following libraries:
    pandas
    plotly
    plotly_resampler
    numpy
    scikit-learn
    tensorflow
    keras
    matplotlib
    joblib
4. Run the main script:
   Execute the main script to preprocess the data, train the LSTM model, and visualize the results
5. View the visualizations and model evaluation results:
   The script will output various visualizations and evaluation metrics to the console and/or save them as files, depending on the implementation.
