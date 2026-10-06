import pandas as pd
import folium
from pyproj import Transformer
from common import *

print("Loading data...")
cells = pd.read_parquet("runs/final/cells_wellhead.parquet")
wells = pd.read_csv("runs/final/wells_used.csv")

# Reconstruct well coordinates
root = zarr.open_group(ZARR_PATH, mode="r")
xw, yw = rc_to_xy(root, wells.taluk.values, wells.row.values, wells.col.values)
wells["lon"], wells["lat"] = Transformer.from_crs(32643, 4326, always_xy=True).transform(xw, yw)

# Initialize map centered on Karnataka
print("Generating map...")
m = folium.Map(location=[14.5, 76.5], zoom_start=8,
               tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
               attr="Esri World Street Map")

def get_color(pct):
    if pct > 75: return "#d73027"  # Red (Severe Stress - Top 25%)
    if pct > 50: return "#fc8d59"  # Orange (High Stress)
    if pct > 25: return "#fee090"  # Yellow (Moderate)
    return "#91bfdb"               # Blue (Low Stress)

# Add Predicted Stress Zones (We will only plot the top 75% to keep the map clean)
fg_pred = folium.FeatureGroup(name="Predicted Sub-Taluk Stress Zones")
for _, r in cells[cells.pct > 25].iterrows():
    folium.CircleMarker(
        location=[r.lat, r.lon], radius=4,
        color=get_color(r.pct), fill=True, fill_opacity=0.6, weight=0,
        popup=f"Taluk: {r.taluk}<br>Stress Percentile: {r.pct:.1f}%"
    ).add_to(fg_pred)

# Add Ground Truth Wells
fg_wells = folium.FeatureGroup(name="Ground Truth Wells")
for _, r in wells[wells.elig].iterrows():
    folium.Marker(
        location=[r.lat, r.lon],
        icon=folium.Icon(color="black", icon="info-sign"),
        popup=f"Well ID: {r.well}<br>Taluk: {r.taluk}<br>Depth Anomaly: {r.y:.2f}"
    ).add_to(fg_wells)

fg_pred.add_to(m)
fg_wells.add_to(m)
folium.LayerControl().add_to(m)

out_file = "groundwater_stress_map.html"
m.save(out_file)
print(f"✅ Map successfully generated! Open '{out_file}' in your web browser.")
