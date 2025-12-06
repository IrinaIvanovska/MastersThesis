#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Task Loader for Meta-RL in CityLearn

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
    # 1. Load all raw data first
    raw_dfs = []
    for building in BUILDINGS:
        csv_path = os.path.join(data_root, building, 'data.csv')
        df = pd.read_csv(csv_path)
        df = df[OBS_COLUMNS]
        raw_dfs.append(df)

    # 2. Define Training Data (Buildings 1-4 are indices 0-3)
    # fit the scaler only on training data to simulate real deployment
    train_dfs = raw_dfs[:4]
    combined_train_data = pd.concat(train_dfs, axis=0)

    # 3. Fit Global Scaler on Training Data
    scaler = MinMaxScaler()
    scaler.fit(combined_train_data)

    # 4. Transform ALL data (Train + Test) using the Training Scaler
    # preserves the relative magnitude differences of Test buildings
    task_datasets = []
    for df in raw_dfs:
        df_scaled = pd.DataFrame(scaler.transform(df), columns=df.columns)
        task_datasets.append(df_scaled)

    # Return scaler and columns for denormalization
    return task_datasets, scaler, OBS_COLUMNS

tasks, scaler, columns = load_all_building_data(DATA_ROOT)

for i, task_df in enumerate(tasks):
    print(f'Task {i+1} - Shape: {task_df.shape}')

print("All tasks loaded and preprocessed. Ready for Reptile LSTM-Q agent.")
