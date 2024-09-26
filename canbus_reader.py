#!/usr/bin/env python
# coding: utf-8

"""
Created by Usama
Script for reading and processing CAN bus data from BLF files using a DBC file.
"""

import os
import can
import cantools
import pandas as pd
import numpy as np

def blf_to_df(blf_file_path, dbc_file_path):
    """
    Reads and decodes CAN messages from a BLF file using a DBC file.

    Parameters:
    blf_file_path (str): Path to the BLF file.
    dbc_file_path (str): Path to the DBC file.

    Returns:
    pd.DataFrame: DataFrame containing the decoded messages.
    list: List of skipped message IDs.
    """
    db = cantools.database.load_file(dbc_file_path)
    reader = can.BLFReader(blf_file_path)
    messages = []
    skipped_ids = []
    all_keys = set()

    print("Messages from BLF file:")
    for msg in reader:
        print(f"ID: {msg.arbitration_id}, Hex ID: 0x{msg.arbitration_id:X}, Timestamp: {msg.timestamp}, Data: {msg.data}")
        if msg.is_error_frame:
            continue
        try:
            decoded_msg = db.decode_message(msg.arbitration_id, msg.data)
            decoded_msg['timestamp'] = msg.timestamp
            decoded_msg['arbitration_id'] = f"0x{msg.arbitration_id:X}"
            print(f"Decoded message: {decoded_msg}")
            messages.append(decoded_msg)
            all_keys.update(decoded_msg.keys())
        except KeyError:
            skipped_ids.append(f"0x{msg.arbitration_id:X}")
            print(f"Skipping message with ID 0x{msg.arbitration_id:X} - not defined in DBC file.")
            continue
        except Exception as e:
            print(f"Error decoding message with ID 0x{msg.arbitration_id:X}: {e}")
            continue

    complete_messages = [{key: msg.get(key, np.nan) for key in all_keys} for msg in messages]
    df = pd.DataFrame(complete_messages)
    print(f"DataFrame shape: {df.shape}")
    print(f"DataFrame head: {df.head()}")

    return df, skipped_ids

def process_can_messages(blf_file_path, dbc_file_path, target_ids):
    """
    Processes CAN messages and saves them to CSV files.

    Parameters:
    blf_file_path (str): Path to the BLF file.
    dbc_file_path (str): Path to the DBC file.
    target_ids (set): Set of target message IDs to process.

    Returns:
    dict: Dictionary of DataFrames indexed by arbitration ID.
    list: List of skipped message IDs.
    """
    db = cantools.database.load_file(dbc_file_path)
    reader = can.BLFReader(blf_file_path)
    dataframes = {}
    skipped_ids = []
    desired_signal_suffixes = {'HghstCellV', 'LwstCellV', 'BtryPckI', 'MxTmp', 'MinTmp', 'SOC'}

    print("Messages from BLF file:")
    for msg in reader:
        if msg.arbitration_id not in target_ids:
            continue
        print(f"ID: {msg.arbitration_id}, Hex ID: 0x{msg.arbitration_id:X}, Timestamp: {msg.timestamp}, Data: {msg.data}")
        if msg.is_error_frame:
            continue
        try:
            decoded_msg = db.decode_message(msg.arbitration_id, msg.data)
            filtered_msg = {key: value for key, value in decoded_msg.items() if any(key.endswith(suffix) for suffix in desired_signal_suffixes)}
            if not filtered_msg:
                skipped_ids.append(f"0x{msg.arbitration_id:X}")
                continue
            filtered_msg['timestamp'] = msg.timestamp
            filtered_msg['arbitration_id'] = f"0x{msg.arbitration_id:X}"
            print(f"Filtered message: {filtered_msg}")

            arbitration_id = f"0x{msg.arbitration_id:X}"
            if arbitration_id not in dataframes:
                dataframes[arbitration_id] = []
            dataframes[arbitration_id].append(filtered_msg)
        except KeyError:
            skipped_ids.append(f"0x{msg.arbitration_id:X}")
            print(f"Skipping message with ID 0x{msg.arbitration_id:X} - not defined in DBC file.")
            continue
        except Exception as e:
            print(f"Error decoding message with ID 0x{msg.arbitration_id:X}: {e}")
            continue

    for arb_id, messages in dataframes.items():
        all_keys = set(key for msg in messages for key in msg.keys())
        complete_messages = [{key: msg.get(key, np.nan) for key in all_keys} for msg in messages]
        df = pd.DataFrame.from_records(complete_messages)
        df = df.sort_values(by=['timestamp'])
        df.to_csv(f'can_data_{arb_id}.csv', index=False)

    print(f"Total unique IDs: {len(dataframes)}")
    
    return dataframes, skipped_ids

def filter_columns(columns, suffixes):
    """Filter columns to include only those with the exact desired suffixes."""
    filtered_columns = []
    for col in columns:
        for suffix in suffixes:
            if col.endswith(suffix) and (col == suffix or col[-len(suffix)-1] in {'_', '-', '.'}):
                filtered_columns.append(col)
                break
    return filtered_columns

def process_all_can_messages(blf_file_path, dbc_file_path, output_folder):
    """
    Processes all CAN messages and saves them to CSV files.

    Parameters:
    blf_file_path (str): Path to the BLF file.
    dbc_file_path (str): Path to the DBC file.
    output_folder (str): Path to the output folder.

    Returns:
    dict: Dictionary of DataFrames indexed by arbitration ID.
    list: List of skipped message IDs.
    """
    db = cantools.database.load_file(dbc_file_path)
    reader = can.BLFReader(blf_file_path)
    dataframes = {}
    skipped_ids = []
    desired_signal_suffixes = {'HghstCellV', 'LwstCellV', 'BtryPckI', 'MxTmp', 'MinTmp', 'SOC'}

    print("Messages from BLF file:")
    for msg in reader:
        print(f"ID: {msg.arbitration_id}, Hex ID: 0x{msg.arbitration_id:X}, Timestamp: {msg.timestamp}, Data: {msg.data}")
        if msg.is_error_frame:
            continue
        try:
            decoded_msg = db.decode_message(msg.arbitration_id, msg.data)
            filtered_columns = filter_columns(decoded_msg.keys(), desired_signal_suffixes)
            filtered_msg = {key: decoded_msg[key] for key in filtered_columns}
            if not filtered_msg:
                skipped_ids.append(f"0x{msg.arbitration_id:X}")
                continue
            filtered_msg['timestamp'] = msg.timestamp
            filtered_msg['arbitration_id'] = f"0x{msg.arbitration_id:X}"
            print(f"Filtered message: {filtered_msg}")

            arbitration_id = f"0x{msg.arbitration_id:X}"
            if arbitration_id not in dataframes:
                dataframes[arbitration_id] = []
            dataframes[arbitration_id].append(filtered_msg)
        except KeyError:
            skipped_ids.append(f"0x{msg.arbitration_id:X}")
            print(f"Skipping message with ID 0x{msg.arbitration_id:X} - not defined in DBC file.")
            continue
        except Exception as e:
            print(f"Error decoding message with ID 0x{msg.arbitration_id:X}: {e}")
            continue

    os.makedirs(output_folder, exist_ok=True)

    for arb_id, messages in dataframes.items():
        all_keys = set(key for msg in messages for key in msg.keys())
        complete_messages = [{key: msg.get(key, np.nan) for key in all_keys} for msg in messages]
        df = pd.DataFrame.from_records(complete_messages)
        df = df.sort_values(by=['timestamp'])
        df.to_csv(os.path.join(output_folder, f'can_data_{arb_id}.csv'), index=False)

    print(f"Total unique IDs: {len(dataframes)}")
    
    return dataframes, skipped_ids

def load_all_ids(blf_file_path, dbc_file_path):
    """
    Loads all CAN message IDs from a BLF file using a DBC file.

    Parameters:
    blf_file_path (str): Path to the BLF file.
    dbc_file_path (str): Path to the DBC file.

    Returns:
    pd.DataFrame: DataFrame containing the decoded messages.
    list: List of skipped message IDs.
    """
    db = cantools.database.load_file(dbc_file_path)
    reader = can.BLFReader(blf_file_path)
    messages = []
    skipped_ids = []

    print("Messages from BLF file:")
    for msg in reader:
        print(f"ID: {msg.arbitration_id}, Hex ID: 0x{msg.arbitration_id:X}, Timestamp: {msg.timestamp}, Data: {msg.data}")
        if msg.is_error_frame:
            continue
        try:
            decoded_msg = db.decode_message(msg.arbitration_id, msg.data)
            decoded_msg['timestamp'] = msg.timestamp
            decoded_msg['arbitration_id'] = f"0x{msg.arbitration_id:X}"
            print(f"Decoded message: {decoded_msg}")
            messages.append(decoded_msg)
        except KeyError:
            skipped_ids.append(f"0x{msg.arbitration_id:X}")
            print(f"Skipping message with ID 0x{msg.arbitration_id:X} - not defined in DBC file.")
            continue
        except Exception as e:
            print(f"Error decoding message with ID 0x{msg.arbitration_id:X}: {e}")
            continue

    df = pd.DataFrame.from_records(messages)
    print(f"DataFrame shape: {df.shape}")
    print(f"DataFrame head: {df.head()}")

    return df, skipped_ids

def combine_signals_to_single_cell(file_paths, desired_signal_suffixes):
    """
    Combines signals from multiple files into a single cell format.

    Parameters:
    file_paths (list): List of file paths to process.
    desired_signal_suffixes (list): List of desired signal suffixes.

    Returns:
    pd.DataFrame: DataFrame containing the combined signals.
    """
    combined_data = []

    for file_path in file_paths:
        df = pd.read_csv(file_path)
        extracted_columns = {suffix: df[[col for col in df.columns if col.endswith(suffix)]] for suffix in desired_signal_suffixes}
        extracted_df = pd.concat(extracted_columns, axis=1)
        combined_data.append(extracted_df)

    combined_df = pd.concat(combined_data, axis=1)
    combined_df.columns = ['_'.join(col) if isinstance(col, tuple) else col for col in combined_df.columns]

    single_cell_data = pd.DataFrame()
    timestamp_cols = [col for col in combined_df.columns if 'timestamp' in col.lower()]
    if timestamp_cols:
        single_cell_data['timestamp'] = combined_df[timestamp_cols[0]]

    suffixes = ['HghstCellV', 'LwstCellV', 'MxTmp', 'MinTmp', 'BtryPckI', 'SOC']
    columns = extract_columns(combined_df, suffixes)

    if 'HghstCellV' in columns and 'LwstCellV' in columns:
        voltage = (columns['HghstCellV'].mean(axis=1) + columns['LwstCellV'].mean(axis=1)) / 2
        single_cell_data['Voltage'] = voltage / 1000

    if 'MxTmp' in columns and 'MinTmp' in columns:
        temperature = (columns['MxTmp'].mean(axis=1) + columns['MinTmp'].mean(axis=1)) / 2
        single_cell_data['Temperature'] = temperature

    if 'BtryPckI' in columns:
        current = columns['BtryPckI'].mean(axis=1)
        single_cell_data['Current'] = current

    if 'SOC' in columns:
        single_cell_data['SOC'] = columns['SOC'].mean(axis=1)

    single_cell_data.to_parquet('single_cell_data.parquet', index=False)
    print(single_cell_data)

# usage for example
if __name__ == "__main__":
    blf_file_path = "Dmitry2024_01_14_20_35_09.blf"
    dbc_file_path = "CanDbD_Internal LMD Type-M BCU General Software Rev 2.5.2.dbc"
    output_folder = "output_can_data_3"
    target_ids = {66052, 66050, 66048}

    # Process CAN messages
    dataframes, skipped_ids = process_can_messages(blf_file_path, dbc_file_path, target_ids)

    # Display the DataFrames
    for arb_id, df in dataframes.items():
        print(f"DataFrame for {arb_id}:\n{df}\n")

    print(f"Skipped IDs: {skipped_ids}")

    # Combine signals to single cell data
    file_paths = ['C:/Users/usama/output_can_data_2/can_data_0x10200.csv', 
                  'C:/Users/usama/output_can_data_2/can_data_0x10202.csv', 
                  'C:/Users/usama/output_can_data_2/can_data_0x10204.csv']
    desired_signal_suffixes = ['timestamp', 'HghstCellV', 'LwstCellV', 'BtryPckI', 'MxTmp', 'MinTmp', 'SOC']
    combine_signals_to_single_cell(file_paths, desired_signal_suffixes)
