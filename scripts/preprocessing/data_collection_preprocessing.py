import pandas as pd
import boto3
from sklearn.preprocessing import MinMaxScaler

# Function to fetch data from AWS Timestream based on a specific OTA SN
def fetch_data_from_timestream(ota_sn):
    # Connect to AWS Timestream using boto3
    client = boto3.client('timestream-query')

    # Construct a query string to fetch data based on OTA SN
    query = f"""
        SELECT time, SOC, Voltage, Current, PumpDutyCycle, SOH, Temperature, LiquidLevel
        FROM your_timestream_table
        WHERE OTA_SN = '{ota_sn}'
        ORDER BY time DESC
    """
    
    # Execute the query and fetch results
    response = client.query(QueryString=query)
    
    # Parse the response into a DataFrame
    rows = []
    for row in response['Rows']:
        data = [col['ScalarValue'] for col in row['Data']]
        rows.append(data)

    # Define column names corresponding to the selected fields
    columns = ['time', 'SOC', 'Voltage', 'Current', 'PumpDutyCycle', 'SOH', 'Temperature', 'LiquidLevel']
    
    # Return DataFrame with parsed data
    return pd.DataFrame(rows, columns=columns)

# Function to clean and preprocess the data
def clean_data(df):
    # Convert numeric columns to proper data types (if necessary)
    df['SOC'] = pd.to_numeric(df['SOC'])
    df['Voltage'] = pd.to_numeric(df['Voltage'])
    df['Current'] = pd.to_numeric(df['Current'])
    df['PumpDutyCycle'] = pd.to_numeric(df['PumpDutyCycle'])
    df['SOH'] = pd.to_numeric(df['SOH'])
    df['Temperature'] = pd.to_numeric(df['Temperature'])
    df['LiquidLevel'] = pd.to_numeric(df['LiquidLevel'])

    # Example cleaning steps
    df = df.dropna()  # Drop any rows with missing values
    df = df[df['Temperature'] > 0]  # Remove unrealistic temperature values
    df = df[df['Voltage'] > 0]  # Remove rows with zero voltage
    return df

# Function to scale the selected columns for ML model training
def scale_data(df):
    scaler = MinMaxScaler()

    # Select relevant columns for scaling
    features = ['SOC', 'Voltage', 'Current', 'PumpDutyCycle', 'SOH', 'Temperature', 'LiquidLevel']
    
    # Fit and transform the data
    scaled_data = scaler.fit_transform(df[features])
    
    # Return the scaled data as a DataFrame
    return pd.DataFrame(scaled_data, columns=features)

# Main function to execute the pipeline
if __name__ == "__main__":
    # Define the OTA SN you want to fetch the data for
    ota_sn = "SN123456"
    
    # Fetch the data from AWS Timestream
    raw_data = fetch_data_from_timestream(ota_sn)
    
    # Print the raw data (optional)
    print("Raw Data:\n", raw_data.head())

    # Clean and preprocess the data
    cleaned_data = clean_data(raw_data)
    
    # Print the cleaned data (optional)
    print("Cleaned Data:\n", cleaned_data.head())

    # Scale the data for model input
    scaled_data = scale_data(cleaned_data)
    
    # Print the scaled data (optional)
    print("Scaled Data:\n", scaled_data.head())

    # Save scaled data to CSV (optional)
    scaled_data.to_csv("scaled_data.csv", index=False)
    
    print("Data preprocessing completed successfully.")
