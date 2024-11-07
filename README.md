# 🔋 **AI Battery Management System (AI BMS)**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)](#)
[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-LSTM%20%7C%20FNN-yellowgreen)](#)
[![OTA Updates](https://img.shields.io/badge/OTA-Enabled-lightblue)](#)
[![Edge & Cloud](https://img.shields.io/badge/Integration-Edge%20%7C%20Cloud-blueviolet)](#)

---

## 🚀 Overview

The **AI Battery Management System (AI BMS)** is an advanced solution for real-time monitoring, state prediction, and optimization of battery performance. Using machine learning models, it estimates key parameters like **SOC** (State of Charge), **SOH** (State of Health), and **temperature**. Integrating with edge platforms for fast data processing and supporting **Over-The-Air (OTA)** updates for remote learning, the AI BMS is designed to manage battery packs efficiently, enhance safety, and extend lifecycle for applications such as electric vehicles and energy storage systems.

![AI BMS Workflow](plots/ai_bms_workflow.png)

---

## 🔄 Pipeline Description

1. **📊 Raw Data Collection**:
   - Battery sensors gather real-time data, including voltage, current, temperature, and other parameters, forming the foundation for accurate model predictions.

2. **⚙️ Data Pre-processing**:
   - Collected data is cleaned, resampled, and normalized to ensure accuracy in machine learning inputs, making it suitable for model training.

3. **🧠 Training ML Models**:
   - Machine learning models such as **LSTM** and **FNN** are trained on historical data to estimate SOC, SOH, and temperature, enabling predictive insights and monitoring.

4. **✅ Validation Step**:
   - Model validation fine-tunes hyperparameters to improve performance and ensure the model’s reliability.

5. **🌐 Deployment on Edge and Cloud**:
   - Models are deployed on **edge devices** for real-time SOC predictions and **cloud platforms** for remote monitoring, balancing low latency with flexible remote processing.

6. **📡 OTA Updates**:
   - OTA updates enable the model to adapt based on new data, ensuring the AI BMS remains accurate and up-to-date with continuous learning.

7. **📈 Prediction and Results**:
   - The AI BMS delivers accurate SOC predictions under various temperature conditions, crucial for real-time battery health management.

8. **📊 Visualization and Analysis**:
   - Visualization tools display SOC predictions, battery temperatures, and other metrics, providing insights for efficient monitoring and decision-making.

---

## 🌟 Features

- **🔋 High-Precision SOC Estimation**: Predicts state of charge with high accuracy, reducing risks of unexpected power loss.
- **🩺 SOH, SOE, and SOI Estimation**: Provides machine learning-based predictions for battery health, energy, and imbalance states.
- **🌡️ Temperature Prediction**: Real-time temperature estimation prevents overheating.
- **💧 Immersion-Cooled Pack Management**: Integrated with cooling systems to ensure safe, stable battery operation.
- **📡 Remote Learning & OTA Updates**: ECM models enable continuous battery performance improvements with over-the-air parameter updates.
- **⚡ Charging Algorithm Optimization**: Optimizes charging cycles to enhance battery longevity and performance.
- **🌐 Edge & Cloud Integration**: Supports seamless integration with **edge** platforms for low-latency operations and **cloud** platforms for advanced simulations and monitoring.

## 📁 Project Structure

### 📂 Folders:

- **`data/`**:
    - **`raw/`**: Contains raw data fetched from AWS Timestream or other sources.
    - **`cleaned/`**: Stores cleaned and processed data ready for model training.

- **`scripts/`**:
    - **`aws_integration/`**:
        - `AWS_timestream_to_python.py`: Fetches data from AWS Timestream using boto3.
    - **`preprocessing/`**:
        - `data_collection_preprocessing.py`: Collects and preprocesses cloud data.
        - `extract_combine_current_data.py`: Extracts and combines current data for analysis.
        - `resample_current_data.py`: Resamples the current data.
        - `rescale_current_data.py`: Rescales the current data for consistency.
    - **`model_training/`**:
        - `train_LSTM_Avg_temp_seasonal.py`: Trains LSTM for average temperature with seasonal data.
        - `train_FNN_winter_finetune.py`: Fine-tunes FNN model with winter data.
        - `train_LSTM_temp.py`: Trains LSTM for temperature prediction.
        - `train_LSTM_temperature_prediction.py`: Another script for LSTM temperature prediction.
        - `Re-train LSTM on simulated data.py`: Retrains LSTM on simulated data.
    - **`simulations/`**:
        - `PyBamm_simulation.py`: Simulates battery behavior using the PyBaMM library.
        - `octave_simulation_full.py`: Full simulation using Octave.
    - **`evaluation/`**:
        - `Test_model.py`: Tests the saved models.
        - `test_saved_model.py`: Evaluates a specific saved model.
        - `life_cycle_data_analysis.py`: Plots dQ/dV life cycle curves for analysis.

- **`models/`**:
    - **AvgTmp_model_LSTM.h5**: Pre-trained LSTM model for average temperature prediction.
    - **FNN_model_yearly.h5**: Pre-trained FNN model for yearly temperature predictions.
    - **lstm_model_new_data_2.h5**: Pre-trained LSTM model for temperature prediction.
    - **scaler_X.pkl**: Scaler object for feature normalization.

- **`README.md`**: Project documentation.
- **`requirements.txt`**: Lists dependencies for running the project.

---

## 🛠️ Setup Instructions

### 🔧 Requirements

To run this project, you’ll need the following dependencies:

- **Python 3.7+**
- **TensorFlow**
- **PyTorch** (optional for model experiments)
- **PyBaMM** (for battery simulations)
- **Pandas**, **NumPy**, **Matplotlib** (for data processing and visualizations)
- **boto3** (for AWS integration)

Install dependencies with:

```bash
pip install -r requirements.txt
```
## 🏃 Running the Models

### 🔄 Training

To train the SOC, SOH, and temperature prediction models, use the following command:

```bash
python scripts/Re-train\ LSTM\ on\ simulated\ data.py --model soc --input_path data/input_data.csv
```
### 📈 Visualizations

Notebooks for visualizing predictions and model performance are not explicitly provided in this repository, but you can create custom visualizations based on the provided scripts using libraries such as `matplotlib` and `seaborn`.

### ☁️ AWS Integration

The system is integrated with AWS services like Timestream and SageMaker for model deployment and data storage. The boto3 library is used for querying Timestream data.

Make sure your AWS credentials are configured using:

- `aws configure`

## 🚀 Usage Examples

### 🔋 SOC Prediction

To predict the **State of Charge (SOC)** of a battery pack using the pre-trained SOC model:

```bash
python scripts/inference.py --model soc --input_path data/test_data.csv
```

### 🌡️ Temperature Estimation

To estimate the temperature of battery cells based on real-time voltage, current, and environmental temperature data:

-`python scripts/inference.py --model temp --input_path data/temperature_data.csv`

### 🌐 Edge and Cloud Deployment

The AI BMS supports deployment on edge platforms like NXP MCUs and cloud platforms like AWS SageMaker. The OTA update feature allows for remote learning and parameter adjustments to continuously optimize performance.

## 📊 Results

Model performance metrics, graphs, and comparisons between different configurations are saved in the `/results` folder. Here you can find:

- **SOC vs. Actual Graphs**
- **Temperature Prediction Accuracy**
- **Model Loss Curves (Training vs Validation)**

---

### 🔍 SOC Prediction Analysis with Charge-Discharge Cycles at Different Temperatures

The plots below demonstrate **State of Charge (SOC)** predictions under various temperature conditions, showcasing how SOC behavior changes during charge-discharge cycles. The **actual SOC** values are represented by blue lines, while **predicted SOC** values are shown in red dashed lines.

#### 🔑 Key Insights

1. **🌡️ Test Data at 0°C**:
   - SOC cycles exhibit a regular pattern, but the charge-discharge rate is slightly slower compared to higher temperatures.
   - Predicted SOC closely follows the actual SOC, with only minor deviations observed, indicating strong model performance under cold conditions.

   ![SOC Prediction at 0°C](plots/soc_prediction_0C.png)

   - **Analysis**: The prediction model shows stable performance, tracking the actual SOC curve accurately even at low temperatures. This suggests that the model effectively captures battery behavior under cold conditions, crucial for applications where battery performance may be impacted by external cold environments.

2. **🌡️ Test Data at 10°C**:
   - The SOC curve appears more dynamic, with slightly higher amplitude oscillations than at 0°C, reflecting improved battery performance and efficiency at this moderate temperature.
   - Predictions are highly accurate, aligning consistently with actual SOC values throughout the charge-discharge cycles.

   ![SOC Prediction at 10°C](plots/soc_prediction_10C.png)

   - **Analysis**: At 10°C, the model's predictive capability continues to excel, indicating enhanced battery response. The close alignment between predicted and actual SOC values shows the model's robustness and accuracy under moderate temperatures, illustrating its adaptability to temperature variations.

3. **🌡️ Test Data at 25°C**:
   - SOC cycles are smoother with well-defined peaks and troughs, as the battery operates near its optimal temperature.
   - The predicted SOC closely matches the actual SOC, highlighting the model’s accuracy in ideal temperature conditions.

   ![SOC Prediction at 25°C](plots/soc_prediction_25C.png)

   - **Analysis**: The model demonstrates peak accuracy at 25°C, operating optimally within the ideal temperature range for battery performance. The close match between actual and predicted SOC values reinforces the model’s reliability under favorable temperature conditions.

## 🤝 Contributing

Contributions to this project are welcome! Feel free to open an **issue** or submit a **pull request** if you'd like to improve the project or add new features.

---

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

### 👤 Author

**Usama Yasir Khan**  
AI Engineer  
XING Mobility

