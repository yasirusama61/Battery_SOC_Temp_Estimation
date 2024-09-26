#!/usr/bin/env python
# coding: utf-8

"""
Created by Usama Yasir Khan
Script for resampling and scaling current data.
"""

import pandas as pd

def resample_current_data(input_file, output_file):
    """
    Resamples and scales current data from the input file.

    Parameters:
    input_file (str): Path to the input file containing combined current data.
    output_file (str): Path to the output file where resampled and scaled data will be saved.

    Returns:
    None
    """
    try:
        data = pd.read_csv(input_file)
        data['Time Stamp'] = pd.to_datetime(data['Time Stamp'])
        data.set_index('Time Stamp', inplace=True)
        
        resampled_data = data['Current [A]'].resample('1S').mean().interpolate()
        model_capacity = 5  # Model cell capacity in Ah
        data_capacity = 3  # Data cell capacity in Ah
        scaled_data = resampled_data * (model_capacity / data_capacity)
        
        scaled_data_df = scaled_data.reset_index()
        scaled_data_df.columns = ['timestamp', 'current']
        
        scaled_data_df.to_csv(output_file, index=False)
        print(f"Resampled and scaled current data saved to {output_file}")

    except Exception as e:
        print(f"Error processing file {input_file}: {e}")

if __name__ == "__main__":
    input_file = 'C:/Users/usama/LG_HG2_Original_Dataset_McMasterUniversity_Jan_2020/25degC_current_data/extracted_combined_current_data.csv'
    output_file = 'C:/Users/usama/LG_HG2_Original_Dataset_McMasterUniversity_Jan_2020/25degC_current_data/resampled_combined_current_data.csv'
    resample_current_data(input_file, output_file)
