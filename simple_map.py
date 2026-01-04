"""
VERY SIMPLE map generator - just the basics
Shows Hengsha Island location on a map
"""

import matplotlib.pyplot as plt
import os
from pathlib import Path

# Create figures directory if it doesn't exist
os.makedirs('D:/2026/Solar_PV_Forecasting_Benchmark/Solar_PV_Generation_LaTex_Work/figures', exist_ok=True)

# Your study site coordinates
latitude = 31.3403
longitude = 121.8389

# Create a simple plot
plt.figure(figsize=(10, 8))

# Plot the point
plt.plot(longitude, latitude, 'r*', markersize=30, label='Hengsha Island')

# Add labels
plt.text(longitude + 0.1, latitude + 0.1, 
         f'Hengsha Island\n({latitude}°N, {longitude}°E)',
         fontsize=12, bbox=dict(facecolor='white', edgecolor='red'))

# Set map limits (zoom level)
plt.xlim(120, 123)
plt.ylim(30, 33)

# Add grid
plt.grid(True, alpha=0.3)

# Labels and title
plt.xlabel('Longitude (°E)', fontsize=12)
plt.ylabel('Latitude (°N)', fontsize=12)
plt.title('Study Site: Hengsha Island, Shanghai, China', fontsize=14, fontweight='bold')
plt.legend()

# Make it look square
plt.gca().set_aspect('equal')

# Save
output_path = 'D:/2026/Solar_PV_Forecasting_Benchmark/Solar_PV_Generation_LaTex_Work/figures/simple_map.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✅ Simple map created: {output_path}")

# Don't show in QGIS (causes issues)
# plt.show()

exec(compile(Path('D:/2026/Solar_PV_Forecasting_Benchmark/Solar_PV_Generation_LaTex_Work/simple_map.py').read_text(), 'D:/2026/Solar_PV_Forecasting_Benchmark/Solar_PV_Generation_LaTex_Work/simple_map.py', 'exec'))
