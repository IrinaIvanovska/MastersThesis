#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import warnings
import matplotlib.pyplot as plt
import pandas as pd
from citylearn.data import DataSet

plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.ymargin'] = 0
warnings.filterwarnings('ignore', category=DeprecationWarning)

#Config
DATASET_NAME = 'citylearn_challenge_2023_phase_3_1'   # Phase 3, 6 buildings
BUILDINGS = [f'Building_{i+1}' for i in range(6)]
columns_to_plot_building = ['solar_generation', 'cooling_demand', 'heating_demand', 'non_shiftable_load']
columns_to_plot_weather = ['outdoor_dry_bulb_temperature', 'outdoor_relative_humidity','diffuse_solar_irradiance', 'direct_solar_irradiance']


OUTPUT_DIR = '../outputs/phase3'
os.makedirs(OUTPUT_DIR, exist_ok=True)
MODELS_DIR = '../models'
os.makedirs(MODELS_DIR, exist_ok=True)


schema = DataSet().get_schema(DATASET_NAME)
root_dir = schema['root_directory']

# Process Each Building
for building in BUILDINGS:
    building_file = schema['buildings'][building]['energy_simulation']
    weather_file = schema['buildings'][building]['weather']
    pricing_file = schema['buildings'][building]['pricing']
    carbon_file = schema['buildings'][building]['carbon_intensity']


    # Load df
    df_building = pd.read_csv(os.path.join(root_dir, building_file))
    df_weather = pd.read_csv(os.path.join(root_dir, weather_file))
    df_pricing = pd.read_csv(os.path.join(root_dir, pricing_file))
    df_carbon = pd.read_csv(os.path.join(root_dir, carbon_file))
    df = pd.concat([df_building, df_weather, df_carbon, df_pricing], axis=1)


    # Save column names and head
    out_building_dir = os.path.join(OUTPUT_DIR, building)
    os.makedirs(out_building_dir, exist_ok=True)
    df.columns.to_series().to_csv(os.path.join(out_building_dir, 'columns.csv'))
    df.to_csv(os.path.join(out_building_dir, 'data.csv'), index=False)

    # Plot and save visualizations
    # Visualize raw input data before training
    print("\n Generating feature plots for each building...")
    # Visualize selected building-related columns
    for col in columns_to_plot_building:
        if col in df.columns:
            plt.figure(figsize=(10, 3))
            plt.plot(df[col], label=col, color='steelblue')
            plt.title(f'{building} - {col}')
            plt.xlabel('Time Step')
            plt.ylabel(col)
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(out_building_dir, f'{col}_building_plot.png'))
            plt.close()

    # Visualize selected weather-related columns
    for col in columns_to_plot_weather:
        if col in df.columns:
            plt.figure(figsize=(10, 3))
            plt.plot(df[col], label=col, color='darkorange')
            plt.title(f'{building} - {col}')
            plt.xlabel('Time Step')
            plt.ylabel(col)
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(out_building_dir, f'{col}_weather_plot.png'))
            plt.close()


print("Visualisations completed successfully for all 6 buildings (Phase 3.1). Outputs saved to 'outputs/phase3/'")


