#!/usr/bin/env python
# coding: utf-8

"""
Battery Aging Simulation Script using PyBaMM

Developed by : Usama Yasir Khan , ML Engineer 

This script simulates the aging of a lithium-ion battery using the Doyle-Fuller-Newman (DFN) model
provided by PyBaMM. The simulation tracks important parameters such as voltage, current, and capacity
as the battery discharges. The script also calculates the differential voltage (dV) over state-of-charge (SOC)
to generate dV/dSOC curves, useful in understanding battery aging characteristics.

Features:
- Default and modified parameter simulations for positive and negative electrodes.
- SOC and dV/dSOC curve generation for both default and modified simulations.
- Data is saved in CSV format for further analysis.
"""

import pybamm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

def run_simulation(parameter_values, file_prefix):
    """
    Run the battery simulation using PyBaMM's DFN model.

    Parameters:
    - parameter_values: PyBaMM ParameterValues object
    - file_prefix: String, used to save the output CSV files with simulation results

    Returns:
    - sim: PyBaMM simulation object
    - solution: PyBaMM solution object
    """
    # Initialize the DFN model
    model = pybamm.lithium_ion.DFN()

    # Define the experiment steps (discharge until 2.5V at 0.4C rate)
    steps = ["Discharge at 0.4C until 2.5V"]

    # Specify the output variables to track
    output_variables = [
        "Current [A]",
        "Voltage [V]"
    ]

    start_time = time.time()

    try:
        # Create an experiment object
        experiment = pybamm.Experiment(steps, period="1 second")

        # Initialize the solver with appropriate settings
        solver = pybamm.CasadiSolver(
            mode="safe", dt_max=1, rtol=1e-7, atol=1e-9, return_solution_if_failed_early=True
        )

        # Create the simulation object with the specified parameters
        sim = pybamm.Simulation(
            model, experiment=experiment, parameter_values=parameter_values, solver=solver
        )

        # Set logging level to WARNING to reduce verbosity
        pybamm.set_logging_level("WARNING")

        # Solve the simulation
        solution = sim.solve()

        # Print summary of the simulation
        end_time = time.time()
        total_time = end_time - start_time
        print(f"Simulation completed successfully in {total_time:.2f} seconds")

        # Extract and save data to CSV
        voltage = solution["Voltage [V]"].entries
        current = solution["Current [A]"].entries
        time_data = solution["Time [s]"].entries

        df = pd.DataFrame({
            'Time [s]': time_data,
            'Voltage [V]': voltage,
            'Current [A]': current
        })

        df.to_csv(f'{file_prefix}_simulation_results.csv', index=False)

        return sim, solution

    except ValueError as e:
        print(f"ValueError during simulation: {e}")
    except pybamm.SolverError as e:
        print(f"SolverError during simulation: {e}")
        return None, None


def calculate_dv_dsoc(solution):
    """
    Calculate dV/dSOC curve from the simulation results.

    Parameters:
    - solution: PyBaMM solution object

    Returns:
    - soc: State-of-charge (SOC) values
    - dV_dSOC: Differential voltage over SOC values
    """
    voltage = solution["Voltage [V]"].entries
    current = solution["Current [A]"].entries
    time_data = solution["Time [s]"].entries

    # Convert time data to hours
    time_data_hours = time_data / 3600

    # Calculate cumulative capacity (Ah)
    cumulative_capacity = np.cumsum(current * np.diff(time_data_hours, prepend=0))

    # Calculate SOC from cumulative capacity
    nominal_capacity = 5  # Assuming a nominal battery capacity of 5 Ah
    soc = 1 - (cumulative_capacity / nominal_capacity)

    # Ensure SOC is non-negative
    soc = np.clip(soc, 0, 1)

    # Calculate dV/dSOC
    dSOC = np.diff(soc)
    dV = np.diff(voltage)
    dV_dSOC = dV / dSOC

    return soc[1:], dV_dSOC


# Simulation with default parameters
parameter_values_default = pybamm.ParameterValues("Chen2020")

print("Default Parameters:")
sim_default, solution_default = run_simulation(parameter_values_default, "default")

# Simulation with modified parameters (Positive electrode volume fraction)
parameter_values_pos_modified = pybamm.ParameterValues("Chen2020")
parameter_values_pos_modified.update({
    "Positive electrode active material volume fraction": 0.7 * parameter_values_pos_modified["Positive electrode active material volume fraction"]
})

print("Modified Positive Electrode Parameters:")
sim_pos_modified, solution_pos_modified = run_simulation(parameter_values_pos_modified, "positive_modified")

# Simulation with modified parameters (Negative electrode volume fraction)
parameter_values_neg_modified = pybamm.ParameterValues("Chen2020")
parameter_values_neg_modified.update({
    "Negative electrode active material volume fraction": 0.7 * parameter_values_neg_modified["Negative electrode active material volume fraction"]
})

print("Modified Negative Electrode Parameters:")
sim_neg_modified, solution_neg_modified = run_simulation(parameter_values_neg_modified, "negative_modified")

# Check if all simulations were successful and plot the results
if solution_default and solution_pos_modified and solution_neg_modified:
    # Plot the default simulation
    sim_default.plot()

    # Plot the positive electrode modified simulation
    sim_pos_modified.plot()

    # Plot the negative electrode modified simulation
    sim_neg_modified.plot()

    # Calculate dV/dSOC for each simulation
    soc_default, dV_dSOC_default = calculate_dv_dsoc(solution_default)
    soc_pos_modified, dV_dSOC_pos_modified = calculate_dv_dsoc(solution_pos_modified)
    soc_neg_modified, dV_dSOC_neg_modified = calculate_dv_dsoc(solution_neg_modified)

    # Plot dV/dSOC for all simulations
    plt.figure(figsize=(10, 6))
    plt.plot(soc_default, dV_dSOC_default, label="Default Parameters")
    plt.plot(soc_pos_modified, dV_dSOC_pos_modified, label="Positive Electrode Modified", linestyle="--")
    plt.plot(soc_neg_modified, dV_dSOC_neg_modified, label="Negative Electrode Modified", linestyle="--")
    plt.xlabel('Normalized SOC')
    plt.ylabel('dV/dSOC [V]')
    plt.title('dV/dSOC Curve - Comparison of Simulations')
    plt.legend()
    plt.grid(True)
    plt.show()

else:
    print("One or more simulations did not complete successfully.")
