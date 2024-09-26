#!/usr/bin/env python
# coding: utf-8

"""
This script queries data from Amazon Timestream using boto3, processes the results, and saves them locally as CSV files.
It also uploads the processed CSV files to an S3 bucket. This example handles pagination when fetching large datasets
and splits data into multiple batches to handle the size limitation.
"""

import boto3
import pandas as pd
import os

# Initialize the Timestream query client with the correct AWS region
client = boto3.client('timestream-query', region_name='us-east-1')

# Create a directory to store the fetched CSV files
output_dir = "timestream_batches"
os.makedirs(output_dir, exist_ok=True)

# Define the query to retrieve relevant data from Timestream
base_query = """
SELECT 
    time,
    N AS SN,
    MxTmp,
    MinTmp,
    BtryPckV,
    SOC,
    SOH,
    BtryPckI,
    MXDischgCurr,
    MXChgCurr,
    AvailChgPWR,
    AvailDischgPWR,
    PmpDty,
    SysSts,
    DschrgSts,
    LqdLvl,
    Lon,
    Lat
FROM "XM_DEV"."03_NuvoOTA_W015634"
"""

# Set the limit for each query to avoid overloading the service (1 million rows per request)
limit = 1000000

# Initialize variables to manage batches and row count
batch_number = 1
total_rows_fetched = 0
has_more_data = True

while has_more_data:
    # Add LIMIT clause to the base query
    query = f"{base_query} LIMIT {limit}"
    
    # Execute the query
    response = client.query(QueryString=query)
    
    # Store the batch data
    batch_data = response['Rows']
    
    # Handle pagination if more data is available (using NextToken)
    while 'NextToken' in response:
        response = client.query(QueryString=query, NextToken=response['NextToken'])
        batch_data.extend(response['Rows'])
    
    # Check if the batch contains data, otherwise break the loop
    if len(batch_data) == 0:
        has_more_data = False
        break
    
    # Update total rows fetched
    total_rows_fetched += len(batch_data)
    
    # Convert the batch data to a list of records
    records = []
    for row in batch_data:
        record = [cell.get('ScalarValue', None) for cell in row['Data']]
        records.append(record)
    
    # Define column names based on the query
    columns = ['time', 'SN', 'MxTmp', 'MinTmp', 'BtryPckV', 'SOC', 'SOH', 'BtryPckI', 
               'MXDischgCurr', 'MXChgCurr', 'AvailChgPWR', 'AvailDischgPWR', 'PmpDty', 
               'SysSts', 'DschrgSts', 'LqdLvl', 'Lon', 'Lat']
    
    # Create a DataFrame from the records
    df = pd.DataFrame(records, columns=columns)
    
    # Save the DataFrame to a CSV file
    csv_file = os.path.join(output_dir, f'battery_data_batch_{batch_number}.csv')
    df.to_csv(csv_file, index=False)
    
    print(f"Batch {batch_number} saved to '{csv_file}' with {len(batch_data)} rows.")
    
    # Increment batch number
    batch_number += 1

# Summary of total rows fetched
print(f"Total rows fetched: {total_rows_fetched}")

# Upload the CSV files to an S3 bucket
s3 = boto3.client('s3')
bucket_name = 'newdata-temp'

# Upload each CSV file to S3
for file_name in os.listdir(output_dir):
    if file_name.endswith(".csv"):
        s3_file_name = f"path/in/s3/{file_name}"
        s3.upload_file(os.path.join(output_dir, file_name), bucket_name, s3_file_name)
        print(f"CSV file '{file_name}' has been uploaded to 's3://{bucket_name}/{s3_file_name}'")

# If required, the script can also check the minimum and maximum time range in the table
def get_time_range():
    min_time_query = """
    SELECT 
        MIN_BY(time, time) as min_time
    FROM "XM_DEV"."03_NuvoOTA_W015634"
    """

    max_time_query = """
    SELECT 
        MAX_BY(time, time) as max_time
    FROM "XM_DEV"."03_NuvoOTA_W015634"
    """

    min_time_response = client.query(QueryString=min_time_query)
    max_time_response = client.query(QueryString=max_time_query)

    if min_time_response['Rows'] and max_time_response['Rows']:
        min_time = min_time_response['Rows'][0]['Data'][0]['ScalarValue']
        max_time = max_time_response['Rows'][0]['Data'][0]['ScalarValue']
        print(f"Data time range: {min_time} to {max_time}")
    else:
        print("Could not determine the time range. Please check the table data.")

