#!/usr/bin/env python
# coding: utf-8

"""
Author: Usama Yasir Khan
Date: Aug 24, 2024
Description: 
This script processes DCIR (Direct Current Internal Resistance) data from CSV files. It computes 
dSOC/dV (rate of state-of-charge change over voltage), generates plots for dSOC/dV vs. Voltage, 
and tracks battery capacity and resistance degradation over cycles. The processed data is saved 
as CSV files, and the generated plots are displayed using Plotly.

Key Features:
- Calculate and plot dSOC/dV vs. Voltage for discharge steps.
- Track and plot capacity and resistance degradation vs. cycle number.
- Save processed data to CSV files.
"""

import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Define the directory containing the files
directory = 'raw data'

# Initialize empty lists to store data
dsoc_dv_data = []
voltage_soc_data = []

# Function to apply a simple moving average
def moving_average(data, window_size):
    """Apply a simple moving average to the data."""
    return np.convolve(data, np.ones(window_size), 'valid') / window_size

# Process all files in the directory
for filename in os.listdir(directory):
    if 'DCIR' in filename and filename.endswith('.csv'):
        filepath = os.path.join(directory, filename)
        print(f"Processing file: {filepath}")
        
        # Read the CSV file
        data = pd.read_csv(filepath)
        
        # Filter for 'Discharge' steps
        discharge_data = data[data['Step'] == 'Discharge']
        
        # Extract Voltage and discharge capacity (disAh)
        voltage = discharge_data['Voltage'].values
        disAh = discharge_data['disAh'].values
        
        # Calculate SOC in percentage
        total_capacity = disAh[-1]  # Assuming last value represents full capacity
        SOC = 100 * (1 - (disAh / total_capacity))  # SOC as percentage
        
        # Calculate dSOC/dV
        dSOC = np.diff(SOC)
        dV = np.diff(voltage)
        
        # Avoid division by zero or very small dV values
        valid_indices = np.abs(dV) > 1e-5
        dSOC = dSOC[valid_indices]
        dV = dV[valid_indices]
        voltage = voltage[:-1][valid_indices]
        dSOC_dV = dSOC / dV
        
        # Apply moving average to smooth the data
        window_size = 10  # Adjustable window size
        if len(dSOC_dV) >= window_size:
            dSOC_dV_smooth = moving_average(dSOC_dV, window_size)
            voltage_smooth = moving_average(voltage, window_size)
            
            # Extract information from the filename for the legend
            parts = filename.split('_')
            cycle = parts[1]
            discap = parts[3]
            dcir_value = parts[5]  # Assuming it's the 6th part of the filename
            
            # Label for the plot
            legend_label = f"{cycle}_{discap}_{dcir_value}"
            
            # Store dSOC/dV and Voltage for plotting
            dsoc_dv_data.append(pd.DataFrame({
                'dSOC/dV': dSOC_dV_smooth,
                'Voltage': voltage_smooth,
                'Label': legend_label
            }))
            
            # Store Voltage vs. SOC data for another plot
            voltage_soc_data.append(pd.DataFrame({
                'Voltage': voltage,
                'SOC': SOC[:-1][valid_indices],
                'Label': legend_label
            }))

# Concatenate all processed dSOC/dV data
dsoc_dv_df = pd.concat(dsoc_dv_data)

# Concatenate all Voltage vs. SOC data
voltage_soc_df = pd.concat(voltage_soc_data)

# Save Voltage vs. SOC data to a CSV file
voltage_soc_df.to_csv('voltage_soc_data.csv', index=False)
print("Voltage vs. SOC data saved to 'voltage_soc_data.csv'")

# Create the Plotly figure for dSOC/dV vs Voltage
fig1 = px.line(dsoc_dv_df, x='Voltage', y='dSOC/dV', color='Label',
               title='dSOC/dV vs Voltage (Discharge Steps Only)',
               width=1200, height=800)

# Display the plot
fig1.show()

# Optionally, save the plot as an HTML file
# fig1.write_html('dsoc_dv_vs_voltage_discharge_steps_smoothed_10.html')


# Part 2: Capacity and Resistance Tracking

# Initialize a list to store capacity, resistance, and cycle data
capacity_resistance_data = []

# Process files again for capacity and resistance
for filename in os.listdir(directory):
    if 'DCIR' in filename and filename.endswith('.csv'):
        filepath = os.path.join(directory, filename)
        
        # Read the CSV file
        data = pd.read_csv(filepath)
        
        # Extract discharge capacity (disAh)
        disAh = data['disAh'].values
        
        # Extract cycle number and resistance from the filename
        parts = filename.split('_')
        cycle_number = int(parts[1].replace('Cycle', ''))
        resistance_str = parts[-1].replace('DCIR_', '').replace('mOhm.csv', '')
        resistance_value = float(resistance_str)
        
        # Store the last capacity value as representative
        capacity_value = disAh[-1]
        
        # Append the data
        capacity_resistance_data.append({
            'Cycle': cycle_number,
            'Capacity [Ah]': capacity_value,
            'Resistance [mOhm]': resistance_value
        })

# Create a DataFrame for capacity and resistance data
capacity_resistance_df = pd.DataFrame(capacity_resistance_data)
capacity_resistance_df.sort_values(by='Cycle', inplace=True)

# Save the DataFrame to a CSV file
capacity_resistance_df.to_csv('capacity_resistance_data.csv', index=False)
print("Capacity and resistance data saved to 'capacity_resistance_data.csv'")

# Plot Capacity and Resistance vs Cycle
fig2 = go.Figure()

# Plot Capacity
fig2.add_trace(go.Scatter(x=capacity_resistance_df['Cycle'], 
                          y=capacity_resistance_df['Capacity [Ah]'],
                          mode='lines+markers', name='Capacity [Ah]', 
                          line=dict(color='red'), yaxis='y2'))

# Plot Resistance
fig2.add_trace(go.Scatter(x=capacity_resistance_df['Cycle'], 
                          y=capacity_resistance_df['Resistance [mOhm]'],
                          mode='lines+markers', name='Resistance [mOhm]', 
                          line=dict(color='blue')))

# Create a secondary y-axis for resistance
fig2.update_layout(
    title='Capacity and Resistance vs Cycles',
    xaxis_title='Cycle Number',
    yaxis=dict(
        title='Resistance [mOhm]',
        titlefont=dict(color='blue'),
        tickfont=dict(color='blue')
    ),
    yaxis2=dict(
        title='Capacity [Ah]',
        titlefont=dict(color='red'),
        tickfont=dict(color='red'),
        overlaying='y',
        side='right'
    ),
    width=700, height=700,
    template='plotly_white'
)

# Show the plot
fig2.show()