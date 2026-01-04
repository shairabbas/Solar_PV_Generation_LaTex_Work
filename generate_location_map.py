"""
Generate a professional geographical location map for Hengsha Island, Shanghai
Inspired by publication-quality maps with satellite view and regional context
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np
from matplotlib.patches import ConnectionPatch
import matplotlib.lines as mlines

# Create figure with dual maps (satellite inset + regional context)
fig = plt.figure(figsize=(14, 10), dpi=300)

# ===== LEFT INSET: Detailed View with Satellite Context =====
ax_inset = plt.subplot(1, 2, 1)

# Hengsha Island detailed region
ax_inset.set_xlim(121.80, 121.95)
ax_inset.set_ylim(31.30, 31.40)
ax_inset.set_aspect('equal')

# Draw East China Sea (water) - light cyan
rect_sea = Rectangle((121.80, 31.30), 0.15, 0.10, 
                      facecolor='#87CEEB', edgecolor='none', alpha=0.6, zorder=1)
ax_inset.add_patch(rect_sea)

# Draw land areas - light tan/beige
rect_land = Rectangle((121.80, 31.35), 0.10, 0.05, 
                       facecolor='#F5DEB3', edgecolor='none', alpha=0.7, zorder=1)
ax_inset.add_patch(rect_land)

# Add grid pattern for satellite feel
for i in np.arange(121.80, 121.96, 0.02):
    ax_inset.axvline(x=i, color='gray', linestyle=':', alpha=0.15, linewidth=0.5, zorder=0)
for j in np.arange(31.30, 31.41, 0.02):
    ax_inset.axhline(y=j, color='gray', linestyle=':', alpha=0.15, linewidth=0.5, zorder=0)

# Mark Hengsha Island with large red star and circle
hengsha_lat, hengsha_lon = 31.3403, 121.8389
ax_inset.plot(hengsha_lon, hengsha_lat, 'r*', markersize=60, 
             label='Hengsha Island\n(Study Site)', zorder=5, markeredgecolor='darkred', 
             markeredgewidth=2)

# Add concentric circles around study site
circle1 = plt.Circle((hengsha_lon, hengsha_lat), 0.015, color='red', fill=False, 
                     linewidth=2.5, linestyle='-', zorder=4)
circle2 = plt.Circle((hengsha_lon, hengsha_lat), 0.025, color='red', fill=False, 
                     linewidth=1.5, linestyle='--', zorder=3)
ax_inset.add_patch(circle1)
ax_inset.add_patch(circle2)

# Add coordinate label box with shadow effect
coord_text = f'Hengsha Island\n{hengsha_lat:.4f}°N\n{hengsha_lon:.4f}°E\nElev. 1.06 m'
ax_inset.text(hengsha_lon - 0.038, hengsha_lat + 0.032, coord_text,
             fontsize=9, fontweight='bold', color='white', zorder=7,
             bbox=dict(boxstyle='round,pad=0.6', facecolor='darkred', 
                      edgecolor='white', linewidth=2, alpha=0.9))

# Add scale bar
scale_length = 0.05  # degrees
scale_x = [121.82, 121.82 + scale_length]
scale_y = [31.305, 31.305]
ax_inset.plot(scale_x, scale_y, 'k-', linewidth=3, zorder=6)
ax_inset.plot([121.82, 121.82], [31.303, 31.307], 'k-', linewidth=2, zorder=6)
ax_inset.plot([121.82 + scale_length, 121.82 + scale_length], [31.303, 31.307], 'k-', linewidth=2, zorder=6)
ax_inset.text(121.82 + scale_length/2, 31.301, f'{scale_length*111:.1f} km', 
             fontsize=8, ha='center', fontweight='bold', zorder=6)

# Add compass rose
compass_x, compass_y = 121.92, 31.385
arrow_size = 0.008
ax_inset.arrow(compass_x, compass_y, 0, arrow_size, head_width=0.003, head_length=0.003, 
              fc='black', ec='black', zorder=10, linewidth=2)
ax_inset.text(compass_x, compass_y + arrow_size + 0.005, 'N', fontsize=11, fontweight='bold', 
             ha='center', zorder=10)

# Format
ax_inset.set_xlabel('Longitude (°E)', fontsize=10, fontweight='bold')
ax_inset.set_ylabel('Latitude (°N)', fontsize=10, fontweight='bold')
ax_inset.set_title('(a) Hengsha Island - Detailed View', fontsize=11, fontweight='bold', 
                  pad=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
ax_inset.grid(False)
ax_inset.set_facecolor('#E8F4F8')
ax_inset.legend(loc='lower left', fontsize=9, framealpha=0.95)

# ===== RIGHT MAP: Regional Context (China and Shanghai) =====
ax_main = plt.subplot(1, 2, 2)

# China and region setup
ax_main.set_xlim(100, 135)
ax_main.set_ylim(15, 55)
ax_main.set_aspect('equal')

# Add country background colors
china_rect = Rectangle((100, 15), 35, 40, facecolor='#F0F0F0', edgecolor='black', 
                       linewidth=0.5, zorder=0)
ax_main.add_patch(china_rect)

# Add neighboring regions
sea_rect = Rectangle((100, 15), 35, 40, facecolor='#ADD8E6', edgecolor='none', 
                     alpha=0.1, zorder=0)
ax_main.add_patch(sea_rect)

# Mark major cities
major_cities = {
    'Beijing': (39.9042, 116.4074, 'k'),
    'Shanghai': (31.2304, 121.4737, 'b'),
    'Guangzhou': (23.1291, 113.2644, 'k'),
    'Chongqing': (29.4316, 106.9123, 'k'),
    'Nanjing': (32.0603, 118.7969, 'k'),
    'Hangzhou': (30.2741, 120.1551, 'k'),
}

for city, (lat, lon, color) in major_cities.items():
    if city == 'Shanghai':
        ax_main.plot(lon, lat, 'bs', markersize=12, zorder=5, markeredgecolor='darkblue', 
                    markeredgewidth=1)
        ax_main.text(lon - 1.5, lat - 1.5, city, fontsize=9, fontweight='bold', color='darkblue', zorder=5)
    else:
        ax_main.plot(lon, lat, 'k.', markersize=5, zorder=4, alpha=0.6)
        ax_main.text(lon - 0.5, lat - 1.2, city, fontsize=7, ha='center', alpha=0.7)

# Highlight Shanghai region with red box
shanghai_box = FancyBboxPatch((120.5, 29.8), 2.5, 3, 
                              boxstyle="round,pad=0.15", 
                              edgecolor='red', facecolor='red', 
                              alpha=0.15, linewidth=3, zorder=3, linestyle='-')
ax_main.add_patch(shanghai_box)

# Add "STUDY AREA" label in box
ax_main.text(121.75, 32.5, 'STUDY\nAREA', fontsize=9, fontweight='bold', 
            color='darkred', ha='center', zorder=4,
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.6))

# Add country label
ax_main.text(115, 45, 'CHINA', fontsize=20, fontweight='bold', alpha=0.15, zorder=1)

# Add East China Sea label
ax_main.text(125, 25, 'East China\nSea', fontsize=10, style='italic', 
            color='steelblue', alpha=0.6, ha='center')

# Format
ax_main.set_xlabel('Longitude (°E)', fontsize=10, fontweight='bold')
ax_main.set_ylabel('Latitude (°N)', fontsize=10, fontweight='bold')
ax_main.set_title('(b) Location Context - China and Shanghai', fontsize=11, fontweight='bold', 
                 pad=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
ax_main.grid(True, alpha=0.2, linestyle=':', linewidth=0.5)
ax_main.set_facecolor('#F5F5F5')

# ===== ADD CONNECTING ARROW BETWEEN MAPS =====
# Create arrow connecting the two maps
arrow = FancyArrowPatch((0.48, 0.5), (0.52, 0.5),
                       transform=fig.transFigure,
                       arrowstyle='<->',
                       mutation_scale=40,
                       linewidth=3,
                       color='red',
                       zorder=100)
fig.patches.append(arrow)

# ===== OVERALL FIGURE TITLE AND FOOTER =====
fig.suptitle('Geographic Location of Solar PV Forecasting Study Site', 
            fontsize=15, fontweight='bold', y=0.98)

# Add legend/information box
legend_text = ('Dataset: NASA POWER Database\n'
              'Period: 2020-2024 (Hourly Data)\n'
              'Location: Subtropical Monsoon Zone\n'
              'Records: 43,824 hourly samples')
fig.text(0.5, 0.02, legend_text,
        ha='center', fontsize=8, style='italic', color='black',
        bbox=dict(boxstyle='round', facecolor='lightyellow', 
                 edgecolor='gray', alpha=0.85, pad=0.8))

plt.tight_layout(rect=[0, 0.06, 1, 0.96])

# Save as high-quality PNG
output_path = r'd:\2026\Solar_PV_Forecasting_Benchmark\Solar_PV_Generation_LaTex_Work\figures\location_map.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"✓ Professional location map generated!")
print(f"✓ Saved to: {output_path}")
print(f"✓ Style: Dual-view (detailed satellite + regional context)")
print(f"✓ Quality: 300 DPI, publication-ready")

plt.close()
