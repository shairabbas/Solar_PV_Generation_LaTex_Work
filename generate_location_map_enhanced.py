"""
Enhanced Location Map Generator for Solar PV Forecasting Study
Generates publication-quality 2-panel map showing:
- Panel A: China regional context with Shanghai highlighted
- Panel B: Detailed Hengsha Island location with coordinates

Requirements:
pip install matplotlib cartopy numpy pillow

Author: Generated for Solar PV Benchmarking Study
Site: Hengsha Island, Shanghai, China (31.3403°N, 121.8389°E)
"""

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Rectangle, FancyBboxPatch
import numpy as np
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

# Study site coordinates
SITE_LAT = 31.3403
SITE_LON = 121.8389
SITE_NAME = "Hengsha Island"

# Create figure with 2 subplots
fig = plt.figure(figsize=(16, 8))

# ==================== PANEL A: Regional Context (China) ====================
ax1 = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())

# Set extent for China view (broader context)
china_extent = [73, 135, 15, 55]  # [lon_min, lon_max, lat_min, lat_max]
ax1.set_extent(china_extent, crs=ccrs.PlateCarree())

# Add map features
ax1.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='none', zorder=1)
ax1.add_feature(cfeature.OCEAN, facecolor='lightblue', zorder=0)
ax1.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='black', zorder=2)
ax1.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='gray', linestyle='--', zorder=2)
ax1.add_feature(cfeature.LAKES, facecolor='lightblue', edgecolor='black', linewidth=0.3, zorder=1)
ax1.add_feature(cfeature.RIVERS, edgecolor='blue', linewidth=0.3, zorder=1)

# Add gridlines
gl1 = ax1.gridlines(draw_labels=True, linewidth=0.5, color='gray', 
                    alpha=0.5, linestyle='--', zorder=3)
gl1.top_labels = False
gl1.right_labels = False
gl1.xlabel_style = {'size': 10}
gl1.ylabel_style = {'size': 10}

# Highlight study region with red rectangle
study_region_extent = [120.5, 122.5, 30.5, 32.0]  # Shanghai region
rect_width = study_region_extent[1] - study_region_extent[0]
rect_height = study_region_extent[3] - study_region_extent[2]

rect = FancyBboxPatch(
    (study_region_extent[0], study_region_extent[2]),
    rect_width, rect_height,
    linewidth=2.5, edgecolor='red', facecolor='none',
    transform=ccrs.PlateCarree(), zorder=5,
    linestyle='-', boxstyle="round,pad=0.05"
)
ax1.add_patch(rect)

# Add site marker
ax1.plot(SITE_LON, SITE_LAT, marker='*', markersize=20, color='red',
         markeredgecolor='darkred', markeredgewidth=1.5,
         transform=ccrs.PlateCarree(), zorder=6)

# Add labels
ax1.text(0.02, 0.98, '(a) Regional Context', transform=ax1.transAxes,
         fontsize=14, fontweight='bold', va='top', ha='left',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Add China label
ax1.text(105, 35, 'CHINA', transform=ccrs.PlateCarree(),
         fontsize=18, fontweight='bold', color='darkblue', 
         ha='center', va='center', alpha=0.7)

# Add Shanghai label
ax1.text(SITE_LON + 1.5, SITE_LAT + 0.5, 'Shanghai', 
         transform=ccrs.PlateCarree(),
         fontsize=11, fontweight='bold', color='red', ha='left',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                   edgecolor='red', alpha=0.9))

# ==================== PANEL B: Detailed Site View ====================
ax2 = fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree())

# Zoom to Hengsha Island area
detail_extent = [121.3, 122.3, 30.9, 31.8]
ax2.set_extent(detail_extent, crs=ccrs.PlateCarree())

# Add map features
ax2.add_feature(cfeature.LAND, facecolor='#e8f4e8', edgecolor='none', zorder=1)
ax2.add_feature(cfeature.OCEAN, facecolor='#d4e9f7', zorder=0)
ax2.add_feature(cfeature.COASTLINE, linewidth=1.2, edgecolor='black', zorder=2)
ax2.add_feature(cfeature.LAKES, facecolor='#d4e9f7', edgecolor='black', 
                linewidth=0.5, zorder=1)
ax2.add_feature(cfeature.RIVERS, edgecolor='#4682b4', linewidth=0.8, zorder=1)

# Add gridlines with labels
gl2 = ax2.gridlines(draw_labels=True, linewidth=0.8, color='gray',
                    alpha=0.6, linestyle='--', zorder=3)
gl2.top_labels = False
gl2.right_labels = False
gl2.xlabel_style = {'size': 11}
gl2.ylabel_style = {'size': 11}
gl2.xformatter = LongitudeFormatter()
gl2.yformatter = LatitudeFormatter()

# Add study site marker with enhanced visibility
ax2.plot(SITE_LON, SITE_LAT, marker='*', markersize=30, color='red',
         markeredgecolor='darkred', markeredgewidth=2,
         transform=ccrs.PlateCarree(), zorder=6)

# Add circle around site
circle = plt.Circle((SITE_LON, SITE_LAT), 0.08, color='red', fill=False,
                   linewidth=2, linestyle='--', alpha=0.7,
                   transform=ccrs.PlateCarree(), zorder=5)
ax2.add_patch(circle)

# Add site information box
info_text = f'{SITE_NAME}\n' \
            f'Lat: {SITE_LAT:.4f}°N\n' \
            f'Lon: {SITE_LON:.4f}°E\n' \
            f'Elevation: ~1.06 m'

ax2.text(0.98, 0.98, info_text, transform=ax2.transAxes,
         fontsize=11, va='top', ha='right', family='monospace',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                   edgecolor='red', linewidth=2, alpha=0.95))

# Add panel label
ax2.text(0.02, 0.98, '(b) Study Site Detail', transform=ax2.transAxes,
         fontsize=14, fontweight='bold', va='top', ha='left',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Add north arrow
arrow_x, arrow_y = 0.92, 0.15
ax2.annotate('N', xy=(arrow_x, arrow_y), xytext=(arrow_x, arrow_y - 0.08),
            xycoords='axes fraction', ha='center', va='center',
            fontsize=14, fontweight='bold',
            arrowprops=dict(arrowstyle='->', lw=2.5, color='black'))
ax2.text(arrow_x, arrow_y - 0.12, 'North', transform=ax2.transAxes,
        fontsize=9, ha='center', fontweight='bold')

# Add scale bar (approximate)
scale_lon_start = 121.4
scale_lon_end = 121.6
scale_lat = 31.0
ax2.plot([scale_lon_start, scale_lon_end], [scale_lat, scale_lat],
        color='black', linewidth=3, transform=ccrs.PlateCarree(), zorder=10)
ax2.plot([scale_lon_start, scale_lon_start], [scale_lat - 0.02, scale_lat + 0.02],
        color='black', linewidth=3, transform=ccrs.PlateCarree(), zorder=10)
ax2.plot([scale_lon_end, scale_lon_end], [scale_lat - 0.02, scale_lat + 0.02],
        color='black', linewidth=3, transform=ccrs.PlateCarree(), zorder=10)
ax2.text((scale_lon_start + scale_lon_end) / 2, scale_lat - 0.08,
        '~20 km', transform=ccrs.PlateCarree(),
        ha='center', fontsize=10, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

# Add data source annotation
fig.text(0.5, 0.02, 
         'Data Source: NASA POWER Database (2020-2024) | 43,824 hourly records | Subtropical Monsoon Climate',
         ha='center', fontsize=11, style='italic', color='darkblue',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', 
                   edgecolor='orange', alpha=0.8))

# Overall title
fig.suptitle('Solar PV Forecasting Study Site: Hengsha Island, Shanghai, China',
            fontsize=16, fontweight='bold', y=0.96)

# Adjust layout
plt.tight_layout(rect=[0, 0.05, 1, 0.94])

# Save figure
output_path = 'figures/location_map_enhanced.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', 
           facecolor='white', edgecolor='none')
print(f"✅ Map saved to: {output_path}")

# Also save as high-quality PDF for publication
output_path_pdf = 'figures/location_map_enhanced.pdf'
plt.savefig(output_path_pdf, bbox_inches='tight', 
           facecolor='white', edgecolor='none')
print(f"✅ PDF saved to: {output_path_pdf}")

plt.show()

print("\n" + "="*60)
print("LOCATION MAP GENERATION COMPLETE")
print("="*60)
print(f"Site: {SITE_NAME}")
print(f"Coordinates: {SITE_LAT}°N, {SITE_LON}°E")
print(f"Output files:")
print(f"  - PNG (300 DPI): {output_path}")
print(f"  - PDF (Vector): {output_path_pdf}")
print("="*60)
