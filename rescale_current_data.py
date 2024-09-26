#!/usr/bin/env python
# coding: utf-8

"""
Created by Usama Yasir
Script for rescaling current data.
"""

import pandas as pd

def rescale_current_data(input_file, output_file):
    """
    Rescales current data to match the model's cell capacity.

    Parameters:
    input_file (str): Path to the input file containing combined current data.
    output_file (str): Path to the output file where rescaled data will be saved.

    Returns:
    None
    """
    try:
        data = pd.read_csv(input_file)
        data['Time Stamp'] = pd.to_datetime(data['Time Stamp'])
        
        model_capacity = 5  # Model cell capacity in Ah
        data_capacity = 3  # Data cell capacity in Ah
        data['Current [A]'] = data['Current [A]'] * (model_capacity / data_capacity)
        
        scaled_data_df = data[['Time Stamp', 'Current [A]', 'File Name']]
        scaled_data_df.columns = ['timestamp', 'current', 'file_name']
        
        scaled_data_df.to_csv(output_file, index=False)
        print(f"Rescaled current data saved to {output_file}")

    except Exception as e:
        print(f"Error processing file {input_file}: {e}")

if __name__ == "__main__":
    input_file = 'C:/Users/usama/LG_HG2_Original_Dataset_McMasterUniversity_Jan_2020/25degC_current_data/extracted_combined_current_data.csv'
    output_file = 'C:/Users/usama/LG_HG2_Original_Dataset_McMasterUniversity_Jan_2020/25degC_current_data/rescaled_combined_current_data.csv'
    rescale_current_data(input_file, output_file)
