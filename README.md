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

### Folders:
- `data/`:
    - **raw/**: Raw data from AWS Timestream or other sources.
    - **cleaned/**: Cleaned and processed data for model training.
  
- `scripts/`:
    - **aws_integration/**: 
        - `AWS_timestream_to_python.py`: Script for fetching data from AWS Timestream using boto3.
    - **preprocessing/**:
        - `data_collection_preprocessing.py`: Script for cloud data collection and preprocessing.
        - `extract_combine_current_data.py`: Script to extract and combine current data.
        - `resample_current_data.py`: Script to resample current data.
        - `rescale_current_data.py`: Script to rescale current data.
    - **model_training/**:
        - `train_LSTM_Avg_temp_seasonal.py`: Script for training LSTM for average temperature with seasonal data.
        - `train_FNN_winter_finetune.py`: Script for fine-tuning FNN model with winter data.
        - `train_LSTM_temp.py`: Script for training LSTM for temperature prediction.
        - `train_LSTM_temperature_prediction.py`: Another script for LSTM temperature prediction.
        - `Re-train LSTM on simulated data.py`: Script for retraining LSTM on simulated data.
    - **simulations/**:
        - `PyBamm_simulation.py`: Script for battery simulations using the PyBaMM library.
        - `octave_simulation_full.py`: Script for full simulation using Octave.
    - **evaluation/**:
        - `Test_model.py`: Script for testing the saved models.
        - `test_saved_model.py`: Script for testing a saved model.
        - `life_cycle_data_analysis.py`: Script for plotting dQ/dV life cycle curves.

- `models/`:
    - **AvgTmp_model_LSTM.h5**: Pre-trained LSTM model for average temperature prediction.
    - **FNN_model_yearly.h5**: Pre-trained FNN model for yearly temperature predictions.
    - **lstm_model_new_data_2.h5**: Pre-trained LSTM model for temperature prediction.
    - **scaler_X.pkl**: Scaler object for feature normalization.
  
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
- **Preprocessing**: Use the scripts provided in the repository to preprocess the data before model training or evaluation. The `data_collection_preprocessing.py` and `extract_combine_current_data.py` scripts are used to fetch, process, and prepare the data.

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

### SOC Prediction Analysis with Charge-Discharge Cycles at Different Temperatures

The plots below demonstrate the State of Charge (SOC) predictions under various temperature conditions, highlighting how the SOC behavior changes during charge-discharge cycles at different temperatures. The actual SOC values are represented by the blue lines, while the predicted SOC values are shown in red dashed lines.

#### Key Insights:
1. **Test Data at 0°C:**
   - The SOC cycles exhibit a regular pattern, but the charge-discharge rate is slightly slower compared to higher temperatures.
   - Predicted SOC closely follows the actual SOC, with only minor deviations observed, indicating that the model performs well under these cold conditions.

   ![SOC Prediction at 0°C](plots/soc_prediction_0C.png)

2. **Test Data at 10°C:**
   - The SOC curve appears more dynamic, with slightly higher amplitude oscillations than at 0°C. This reflects improved battery performance and efficiency at this moderate temperature.
   - Predictions are highly accurate, maintaining consistency with the actual SOC values throughout the charge-discharge cycles.

   ![SOC Prediction at 10°C](plots/soc_prediction_10C.png)

3. **Test Data at 25°C:**
   - The SOC cycles show even smoother behavior, with well-defined peaks and troughs, as the battery operates closer to its optimal temperature range.
   - The predicted SOC matches the actual SOC quite closely, showcasing the model's capability to accurately predict SOC in favorable temperature conditions.

   ![SOC Prediction at 25°C](plots/soc_prediction_25C.png)

These results illustrate that the SOC prediction model can effectively adapt to different temperature conditions, with consistent accuracy across varying temperatures. The model demonstrates robust predictive capability, which is critical for real-time battery management and optimization.


## Contributing

Contributions to this project are welcome. Feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

### Author
Usama Yasir Khan
AI Engineer
XING Mobility
