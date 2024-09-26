#!/usr/bin/env python
# coding: utf-8

"""
Created by Usama 
Script for running simulations on current profile data using Octave.
"""

import os
from oct2py import Oct2Py
import scipy.io
import numpy as np
import time
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

def run_simulation(oc, ik, Temp, deltaT, z0, ir0, h0, doHyst, cellModel):
    """
    Runs the simulation in Octave.

    Parameters:
    oc (Oct2Py): Oct2Py instance.
    ik (np.array): Current profile.
    Temp (float): Temperature.
    deltaT (float): Time step.
    z0 (float): Initial SOC.
    ir0 (float): Initial ir.
    h0 (float): Initial h.
    doHyst (int): Hysteresis flag.
    cellModel (dict): Cell model.

    Returns:
    dict: Simulation results.
    """
    ik_simulation = -ik
    
    oc.push('ik', ik_simulation)
    oc.push('Temp', Temp)
    oc.push('deltaT', deltaT)
    oc.push('z0', z0)
    oc.push('ir0', ir0)
    oc.push('h0', h0)
    oc.push('doHyst', doHyst)
    oc.push('cellModel', cellModel)
    
    start_time = time.time()
    try:
        r1 = oc.eval('simulationPAN(ik, Temp, deltaT, cellModel, z0, ir0, h0, doHyst)')
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    end_time = time.time()
    print(f"Simulation completed in {end_time - start_time:.2f} seconds")
    
    return r1

def process_files(input_file, output_folder, matCell, Temp, deltaT, z0, ir0, h0, doHyst, percentage=10):
    """
    Processes current profile data files and runs simulations.

    Parameters:
    input_file (str): Path to the input CSV file containing current data.
    output_folder (str): Path to the output folder for simulation results.
    matCell (dict): Loaded .mat cell model.
    Temp (float): Temperature.
    deltaT (float): Time step.
    z0 (float): Initial SOC.
    ir0 (float): Initial ir.
    h0 (float): Initial h.
    doHyst (int): Hysteresis flag.
    percentage (int): Percentage of the data to use for simulation.

    Returns:
    None
    """
    os.makedirs(output_folder, exist_ok=True)
    oc = Oct2Py()
    oc.addpath("C:/Users/usama/OctaveCellSimulator")
    cellModel = matCell['cellModel']

    try:
        data = pd.read_csv(input_file)
        num_rows = int(len(data) * (percentage / 100))
        data_subset = data.iloc[:num_rows]
        data_subset['current'] *= 1.13
        current_profile = data_subset['current'].values
        ik = current_profile.reshape(-1, 1)

        print(f"Running simulation for {percentage}% of the profile")
        simulation_data = run_simulation(oc, ik, Temp, deltaT, z0, ir0, h0, doHyst, cellModel)

        if simulation_data is not None:
            vk = simulation_data['vk']
            soc = simulation_data['z']
            temp = simulation_data['temp']

            timestamps = data_subset['timestamp'].values
            results_df = pd.DataFrame({
                'timestamp': timestamps,
                'voltage': vk.flatten(),
                'current': ik.flatten(),
                'temperature': temp.flatten(),
                'state_of_charge': soc.flatten()
            })

            output_file = os.path.join(output_folder, f"partial_simulation_results_{percentage}percent.csv")
            results_df.to_csv(output_file, index=False)
            print(f"Results saved to {output_file}")

    except Exception as e:
        print(f"Error processing file {input_file}: {e}")

if __name__ == "__main__":
    # Load the .mat file
    matCell = scipy.io.loadmat('2170M.mat')

    # Set up the simulation parameters
    Temp = 25  # temperature 25°C
    deltaT = 1  # time step in seconds
    z0 = 0    # initial SOC
    ir0 = 0
    h0 = 0
    doHyst = 1

    # Specify the input file and output folder
    input_file = 'C:/Users/usama/LG_HG2_Original_Dataset_McMasterUniversity_Jan_2020/25degC_current_data/resampled_combined_current_data.csv'
    output_folder = 'C:/Users/usama/LG_HG2_Original_Dataset_McMasterUniversity_Jan_2020/25degC_simulation_results_2'

    # Process the file and run simulations (set percentage to 15%)
    process_files(input_file, output_folder, matCell, Temp, deltaT, z0, ir0, h0, doHyst, percentage=100)
