#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 150

OUTPUT_DIR = '../outputs/graphics'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*80)
print("CREATING PLOTS")
print("="*80)

# Load data from CSV
df = pd.read_csv('../outputs/comprehensive_metrics/all_metrics.csv')

print("Using data:")
for _, row in df.iterrows():
    print(f"  {row['Method']:12s}: Carbon={row['Carbon_%']:+.2f}%, "
          f"Cost={row['Cost_%']:+.2f}%, Peak={row['Peak_%']:+.2f}%")

# Extract values
results = {}
for _, row in df.iterrows():
    results[row['Method']] = {
        'carbon': row['Carbon_%'],
        'cost': row['Cost_%'],
        'peak': row['Peak_%']
    }

COLORS = {
    'Meta-RL': '#E74C3C',
    'Standard RL': '#F39C12',
    'RBC': '#95A5A6'
}

fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(3)
width = 0.25

for i, model in enumerate(['Meta-RL', 'Standard RL', 'RBC']):
    values = [results[model]['carbon'], results[model]['cost'], results[model]['peak']]
    offset = (i - 1) * width
    bars = ax.bar(x + offset, values, width, label=model,
                   color=COLORS[model], alpha=0.85,
                   edgecolor='black', linewidth=0.8)
    for bar, val in zip(bars, values):
        height = bar.get_height()
        y_pos = height + (0.3 if height >= 0 else -0.3)
        va = 'bottom' if height >= 0 else 'top'
        ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                 f'{val:+.1f}%', ha='center', va=va,
                 fontweight='bold')

ax.axhline(y=0, color='black', linewidth=0.8)
ax.set_ylabel('Reduction/Savings (%)', fontweight='bold')
#ax.set_title('Test Performance on Unseen Buildings',
 #             fontweight='bold', fontsize=13, loc='left')
ax.set_xticks(x)
ax.set_xticklabels(['Carbon\nReduction', 'Cost\nSavings', 'Peak\nReduction'])
ax.legend(loc='upper right', fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim([-12, 12])

#plt.suptitle('Meta-RL for Building Energy Management: Complete Performance Analysis',
    #         fontsize=16, fontweight='bold', y=0.98)

save_path = f'{OUTPUT_DIR}/performance_comparison.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')


plt.close()

print(f" Saved: {save_path}")