from qgis.core import (
    QgsProject,
    QgsVectorLayer,
    QgsFeature,
    QgsGeometry,
    QgsPointXY,
    QgsField,
    QgsSymbol
)
from PyQt5.QtCore import QVariant
from PyQt5.QtGui import QColor

# -----------------------------
# 1. Set Project CRS (WGS 84)
# -----------------------------
project = QgsProject.instance()
project.setCrs(project.crs().fromEpsgId(4326))

# -----------------------------
# 2. Create Point Layer (Study Site)
# -----------------------------
point_layer = QgsVectorLayer(
    "Point?crs=EPSG:4326",
    "Hengsha_Island_Center",
    "memory"
)
provider = point_layer.dataProvider()
provider.addAttributes([QgsField("Name", QVariant.String)])
point_layer.updateFields()

# Coordinates of Hengsha Island
lon = 121.84
lat = 31.34

feature = QgsFeature()
feature.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(lon, lat)))
feature.setAttributes(["Hengsha Island"])
provider.addFeature(feature)

point_layer.updateExtents()
project.addMapLayer(point_layer)

# -----------------------------
# 3. Style the Point (FIXED)
# -----------------------------
symbol = QgsSymbol.defaultSymbol(point_layer.geometryType())
symbol.setSize(6)
symbol.setColor(QColor(255, 0, 0))   # RED
point_layer.renderer().setSymbol(symbol)
point_layer.triggerRepaint()

# -----------------------------
# 4. Create Buffer (Study Area)
# -----------------------------
buffer_layer = QgsVectorLayer(
    "Polygon?crs=EPSG:4326",
    "Study_Area_Buffer",
    "memory"
)
buffer_provider = buffer_layer.dataProvider()
buffer_provider.addAttributes([QgsField("Buffer_km", QVariant.Int)])
buffer_layer.updateFields()

# Buffer radius (approx. 5 km)
buffer_distance = 0.05

buffer_geom = feature.geometry().buffer(buffer_distance, 50)
buffer_feature = QgsFeature()
buffer_feature.setGeometry(buffer_geom)
buffer_feature.setAttributes([5])

buffer_provider.addFeature(buffer_feature)
buffer_layer.updateExtents()
project.addMapLayer(buffer_layer)

# -----------------------------
# 5. Style the Buffer (FIXED)
# -----------------------------
buffer_symbol = QgsSymbol.defaultSymbol(buffer_layer.geometryType())
buffer_symbol.setOpacity(0.25)
buffer_symbol.setColor(QColor(0, 0, 255))   # BLUE
buffer_layer.renderer().setSymbol(buffer_symbol)
buffer_layer.triggerRepaint()

# -----------------------------
# 6. Zoom to Study Area
# -----------------------------
iface.mapCanvas().setExtent(buffer_layer.extent())
iface.mapCanvas().refresh()

print("✅ Hengsha Island study area created successfully.")
