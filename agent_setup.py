#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 23 07:32:43 2025

@author: ivanovsi
"""

# ----------------------------------------
# Task Loader for Meta-RL in CityLearn
# ----------------------------------------

import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Configuration
DATA_ROOT = 'outputs/phase3'
BUILDINGS = [f'Building_{i+1}' for i in range(6)]
OBS_COLUMNS = ["month", "hour", "day_type",
    "daylight_savings_status",
    "indoor_dry_bulb_temperature",
    "average_unmet_cooling_setpoint_difference",
    "indoor_relative_humidity",
    "non_shiftable_load",
    "dhw_demand",
    "cooling_demand",
    "heating_demand",
    "solar_generation",
    "occupant_count",
    "indoor_dry_bulb_temperature_cooling_set_point",
    "indoor_dry_bulb_temperature_heating_set_point",
    "hvac_mode",
    "outdoor_dry_bulb_temperature",
    "outdoor_relative_humidity",
    "diffuse_solar_irradiance",
    "direct_solar_irradiance",
    "carbon_intensity",
    "electricity_pricing",
    "outdoor_dry_bulb_temperature_predicted_2",
    "outdoor_relative_humidity_predicted_2",
    "diffuse_solar_irradiance_predicted_2",
    "direct_solar_irradiance_predicted_2",
    "electricity_pricing_predicted_2",
    "electricity_demand"
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

# Debug: print shape of each task
for i, task_df in enumerate(tasks):
    print(f'Task {i+1} - Shape: {task_df.shape}')

print("✅ All tasks loaded and preprocessed. Ready for Reptile LSTM-Q agent.")

