"""
Automated Location Map Generator for QGIS Python Console
Creates publication-ready map of Hengsha Island study site
"""

from qgis.core import *
from qgis.utils import iface
from PyQt5.QtCore import QPointF, QSizeF
from PyQt5.QtGui import QColor, QFont

# Configuration
SITE_LAT = 31.3403
SITE_LON = 121.8389
SITE_NAME = "Hengsha Island"
OUTPUT_DIR = "D:/2026/Solar_PV_Forecasting_Benchmark/Solar_PV_Generation_LaTex_Work/figures/"

print("="*60)
print("AUTOMATED MAP GENERATION STARTED")
print("="*60)

# 1. Add OSM Basemap
print("📍 Step 1: Adding OpenStreetMap basemap...")
osm_url = "type=xyz&url=https://tile.openstreetmap.org/{z}/{x}/{y}.png&zmax=19&zmin=0"
osm_layer = QgsRasterLayer(osm_url, 'OpenStreetMap', 'wms')
if osm_layer.isValid():
    QgsProject.instance().addMapLayer(osm_layer)
    print("   ✅ OSM basemap loaded")
else:
    print("   ❌ Failed to load OSM")

# 2. Create Study Site Point
print("📍 Step 2: Creating study site point...")
layer = QgsVectorLayer('Point?crs=EPSG:4326', SITE_NAME, 'memory')
provider = layer.dataProvider()
provider.addAttributes([
    QgsField('name', QVariant.String),
    QgsField('latitude', QVariant.Double),
    QgsField('longitude', QVariant.Double)
])
layer.updateFields()

feature = QgsFeature()
feature.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(SITE_LON, SITE_LAT)))
feature.setAttributes([SITE_NAME, SITE_LAT, SITE_LON])
provider.addFeatures([feature])

# 3. Style the Point
print("📍 Step 3: Styling point as red star...")
symbol = QgsMarkerSymbol.createSimple({
    'name': 'star',
    'color': '#FF0000',
    'size': '6',
    'outline_color': '#8B0000',
    'outline_width': '0.8'
})
layer.renderer().setSymbol(symbol)

# 4. Add Labels
print("📍 Step 4: Adding labels...")
label_settings = QgsPalLayerSettings()
label_settings.fieldName = 'name'
label_settings.enabled = True

text_format = QgsTextFormat()
text_format.setFont(QFont("Arial", 11, QFont.Bold))
text_format.setSize(11)
text_format.setColor(QColor("#000000"))

buffer_settings = QgsTextBufferSettings()
buffer_settings.setEnabled(True)
buffer_settings.setSize(1.5)
buffer_settings.setColor(QColor("#FFFFFF"))
text_format.setBuffer(buffer_settings)

label_settings.setFormat(text_format)
labeling = QgsVectorLayerSimpleLabeling(label_settings)
layer.setLabeling(labeling)
layer.setLabelsEnabled(True)

# 5. Add to Map
QgsProject.instance().addMapLayer(layer)
print("   ✅ Point layer added")

# 6. Zoom to Site with Context
print("📍 Step 5: Setting map extent...")
extent = layer.extent()
extent.scale(15)  # Moderate zoom to show Hengsha Island and surrounding area
iface.mapCanvas().setExtent(extent)
iface.mapCanvas().refresh()
iface.mapCanvas().waitWhileRendering()
print("   ✅ Map centered on study site")

# 7. Export Map
print("📍 Step 6: Exporting map...")
from qgis.core import (QgsLayoutExporter, QgsLayoutItemMap, QgsLayoutPoint, 
                       QgsLayoutSize, QgsLayoutItemLabel, QgsLayoutItemPage,
                       QgsUnitTypes)

project = QgsProject.instance()
manager = project.layoutManager()

# Remove existing layouts
for layout in manager.layouts():
    if layout.name() == 'Hengsha_Map':
        manager.removeLayout(layout)

# Create new layout
layout = QgsPrintLayout(project)
layout.initializeDefaults()
layout.setName('Hengsha_Map')

# Set page size (A4 landscape)
pc = layout.pageCollection()
pc.pages()[0].setPageSize('A4', QgsLayoutItemPage.Landscape)

# Add map
map_item = QgsLayoutItemMap(layout)
map_item.attemptResize(QgsLayoutSize(270, 180, QgsUnitTypes.LayoutMillimeters))
map_item.attemptMove(QgsLayoutPoint(5, 20, QgsUnitTypes.LayoutMillimeters))
map_item.setExtent(iface.mapCanvas().extent())
map_item.setLayers([osm_layer, layer])  # Explicitly set layers
layout.addLayoutItem(map_item)
map_item.refresh()

# Add title
title = QgsLayoutItemLabel(layout)
title.setText(f"Study Site: {SITE_NAME}, Shanghai, China ({SITE_LAT}°N, {SITE_LON}°E)")
title.setFont(QFont("Arial", 14, QFont.Bold))
title.attemptMove(QgsLayoutPoint(5, 5, QgsUnitTypes.LayoutMillimeters))
title.attemptResize(QgsLayoutSize(270, 10, QgsUnitTypes.LayoutMillimeters))
layout.addLayoutItem(title)

manager.addLayout(layout)

# Export as PNG (300 DPI)
exporter = QgsLayoutExporter(layout)
export_settings = QgsLayoutExporter.ImageExportSettings()
export_settings.dpi = 300

png_output = OUTPUT_DIR + "location_map_qgis.png"
result = exporter.exportToImage(png_output, export_settings)

if result == QgsLayoutExporter.Success:
    print(f"   ✅ PNG exported: {png_output}")
else:
    print(f"   ❌ Export failed")

# Export as PDF
pdf_output = OUTPUT_DIR + "location_map_qgis.pdf"
pdf_result = exporter.exportToPdf(pdf_output, QgsLayoutExporter.PdfExportSettings())

if pdf_result == QgsLayoutExporter.Success:
    print(f"   ✅ PDF exported: {pdf_output}")

print("="*60)
print("MAP GENERATION COMPLETE!")
print(f"Site: {SITE_NAME}")
print(f"Coordinates: {SITE_LAT}°N, {SITE_LON}°E")
print("="*60)