#!/usr/bin/env python
# coding: utf-8

"""
Created by Usama
Script for extracting and combining current data from CSV files.
"""

import os
import pandas as pd

def extract_and_combine_current_data(folder_path, output_file):
    """
    Extracts and combines current data from CSV files in the specified folder.

    Parameters:
    folder_path (str): Path to the folder containing the CSV files.
    output_file (str): Path to the output file where combined data will be saved.

    Returns:
    None
    """
    all_data = []

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path) and file_name.endswith('.csv'):
            try:
                current_data = pd.read_csv(file_path, skiprows=28)
                current_data = current_data.drop(0)  # Drop the row containing units
                column_names = ['Time Stamp', 'Step', 'Status', 'Prog Time', 'Step Time', 'Cycle', 
                                'Cycle Level', 'Procedure', 'Voltage [V]', 'Current [A]', 
                                'Temperature [C]', 'Capacity [Ah]', 'WhAccu [Wh]', 'Cnt [Cnt]', 'Unnamed: 14']
                current_data.columns = column_names
                
                if 'Current [A]' in current_data.columns and 'Time Stamp' in current_data.columns:
                    current_data['Current [A]'] = pd.to_numeric(current_data['Current [A]'], errors='coerce')
                    current_data['Time Stamp'] = pd.to_datetime(current_data['Time Stamp'], errors='coerce', infer_datetime_format=True)
                    current_data.dropna(subset=['Time Stamp', 'Current [A]'], inplace=True)
                    extracted_data = current_data[['Time Stamp', 'Current [A]']].copy()
                    extracted_data['File Name'] = file_name
                    all_data.append(extracted_data)
                else:
                    print(f"Required columns not found in {file_path}")
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

    combined_data = pd.concat(all_data)
    combined_data.to_csv(output_file, index=False)
    print(f"All extracted data saved to {output_file}")

if __name__ == "__main__":
    folder_path = 'C:/Users/usama/LG_HG2_Original_Dataset_McMasterUniversity_Jan_2020/25degC'
    output_file = 'C:/Users/usama/LG_HG2_Original_Dataset_McMasterUniversity_Jan_2020/25degC_current_data/extracted_combined_current_data.csv'
    extract_and_combine_current_data(folder_path, output_file)
