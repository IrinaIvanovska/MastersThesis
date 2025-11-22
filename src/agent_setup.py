#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 29 13:32:43 2025

@author: ivanovsi
"""

# ----------------------------------------
# Task Loader for Meta-RL in CityLearn
# ----------------------------------------

import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Configuration
DATA_ROOT = '../outputs/phase3'
BUILDINGS = [f'Building_{i+1}' for i in range(6)]
OBS_COLUMNS = [
    "month", "hour", "day_type", "daylight_savings_status",
    "outdoor_dry_bulb_temperature",
    "diffuse_solar_irradiance", "direct_solar_irradiance",
    "carbon_intensity",
    "indoor_dry_bulb_temperature",
    "non_shiftable_load", "solar_generation",
    "electricity_pricing",
    "cooling_demand", "heating_demand", "dhw_demand",
    "occupant_count", "average_unmet_cooling_setpoint_difference"
]

def load_all_building_data(data_root):
    task_datasets = []
    for building in BUILDINGS:
        csv_path = os.path.join(data_root, building, 'data.csv')
        df = pd.read_csv(csv_path)

        # Select only relevant columns
        df = df[OBS_COLUMNS]

        # Normalize values per building (independently)
        scaler = MinMaxScaler()
        df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

        task_datasets.append(df_scaled)

    return task_datasets

# Load all task datasets
tasks = load_all_building_data(DATA_ROOT)

# Print dataset information
for i, task_df in enumerate(tasks):
    print(f'Task {i+1} - Shape: {task_df.shape}')

print("All tasks loaded and preprocessed. Ready for Reptile LSTM-Q agent.")
