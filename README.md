# AI Battery Management System (AI BMS)

## Overview

The AI Battery Management System (AI BMS) is an advanced solution designed for real-time monitoring, state prediction, and optimization of battery performance. This system uses machine learning models to estimate key parameters like SOC (State of Charge), SOH (State of Health), and temperature. It also integrates with edge platforms to enable fast data processing and supports remote learning through OTA (Over-The-Air) updates.

The AI BMS is designed for high-efficiency management of battery packs, including thermal management, safety controls, and performance optimization. Its features aim to extend the lifecycle of battery systems, improve operational efficiency, and enhance the safety of electric vehicles or other energy storage applications.

## Features

- **High-Precision SOC Estimation**: Accurate state of charge predictions to prevent unexpected power outages.
- **SOH, SOE, and SOI Estimation**: Machine learning-based predictions for health, energy, and imbalance.
- **Temperature Prediction**: Real-time estimation of battery temperature to prevent overheating.
- **Immersion-Cooled Pack Management**: Integrated with cooling systems to ensure stable and safe operation.
- **Remote Learning and OTA Updates**: Use ECM models for continuous improvement of battery performance via over-the-air parameter updates.
- **Optimization of Charging Algorithms**: Algorithms to optimize charging cycles for longevity and performance.
- **Edge and Cloud Integration**: Seamlessly integrates with edge platforms for low-latency operations and cloud platforms for advanced simulations and monitoring.

## Project Structure

- `AWS_timestream_to_python.py`: Script for fetching data from AWS Timestream using boto3.
- `AvgTmp_model_LSTM.h5`: Pre-trained LSTM model for average temperature prediction.
- `FNN_Temp_prediction.py`: Script for predicting temperature using a Feedforward Neural Network.
- `FNN_model_yearly.h5`: Pre-trained FNN model for yearly temperature predictions.
- `PyBamm_simulation.py`: Script for battery simulations using the PyBaMM library.
- `Re-train LSTM on simulated data.py`: Script for retraining LSTM on simulated data.
- `Test_model.py`: Script for testing the saved models.
- `canbus_reader.py`: Script for reading CAN bus data.
- `convo_lstm.py`: Script for SOC prediction using Convolutional LSTM.
- `data_collection_preprocessing.py`: Script for cloud data collection and preprocessing.
- `extract_combine_current_data.py`: Script to extract and combine current data.
- `life_cycle_data_analysis.py`: Script for plotting dQ/dV life cycle curves.
- `lstm_model_new_data_2.h5`: Pre-trained LSTM model for temperature prediction.
- `octave_simulation_full.py`: Script for full simulation using Octave.
- `quantize_model.py`: Script for model quantization.
- `resample_current_data.py`: Script to resample current data.
- `rescale_current_data.py`: Script to rescale current data.
- `scaler_X.pkl`: Scaler object for feature normalization.
- `test_saved_model.py`: Script for testing a saved model.
- `train_FNN_winter_finetune.py`: Script for fine-tuning FNN model with winter data.
- `train_LSTM_Avg_temp_seasonal.py`: Script for training LSTM for average temperature with seasonal data.
- `train_LSTM_temp.py`: Script for training LSTM for temperature prediction.
- `train_LSTM_temperature_prediction.py`: Another script for LSTM temperature prediction.
- `README.md`: Documentation for the project.
- `requirements.txt`: Required dependencies for running the project.


## Setup Instructions

### Requirements

To run this project, you'll need the following dependencies:

- Python 3.7+
- TensorFlow
- PyTorch (optional for model experiments)
- Pybamm (for battery simulations)
- Pandas, NumPy, Matplotlib (for data processing and visualizations)
- boto3 (for AWS integration)

You can install the required dependencies using:

- `pip install -r requirements.txt`

### Data Setup

- **Access to Data**: The dataset used in this project is proprietary and cannot be shared publicly.
- **Using Sample Data**: To demonstrate the functionality, users can create or simulate dummy data using the provided scripts or use other publicly available datasets in a similar format. The structure of the dataset is outlined in the documentation within the `/data` folder.
**Preprocessing**: Use the scripts provided in the repository to preprocess the data before model training or evaluation. The `data_collection_preprocessing.py` and `extract_combine_current_data.py` scripts are used to fetch, process, and prepare the data.

## Running the Models

### Training

To train the SOC, SOH, and temperature prediction models, run the following command:

-`python scripts/Re-train LSTM on simulated data .py --model soc --input_path data/input_data.csv`

### Visualizations

Notebooks for visualizing predictions and model performance are not explicitly provided in this repository, but you can create custom visualizations based on the provided scripts using libraries such as `matplotlib` and `seaborn`.

### AWS Integration

The system is integrated with AWS services like Timestream and SageMaker for model deployment and data storage. The boto3 library is used for querying Timestream data.

Make sure your AWS credentials are configured using:

- `aws configure`

## Usage Examples

### SOC Prediction

To predict the state of charge (SOC) of a battery pack, use the pre-trained SOC model as follows:

-`python scripts/inference.py --model soc --input_path data/test_data.csv`

### Temperature Estimation

To estimate the temperature of battery cells based on real-time voltage, current, and environmental temperature data:

-`python scripts/inference.py --model temp --input_path data/temperature_data.csv`

### Edge and Cloud Deployment

The AI BMS supports deployment on edge platforms like NXP MCUs and cloud platforms like AWS SageMaker. The OTA update feature allows for remote learning and parameter adjustments to continuously optimize performance.

## Results

Model performance metrics, graphs, and comparisons between different configurations are saved in the `/results folder`. You can find:

- SOC vs Actual Graphs
- Temperature Prediction Accuracy
- Model Loss Curves (Training vs Validation)

## Contributing

Contributions to this project are welcome. Feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

### Author
Usama Yasir Khan
AI Engineer
XING Mobility
