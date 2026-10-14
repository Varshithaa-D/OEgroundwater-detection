import os, zarr
import numpy as np, pandas as pd
import matplotlib.image as mpimg
from common import PatchStore, load_index, rc_to_xy, P

out_dir = "qgis_sample_patches"
os.makedirs(out_dir, exist_ok=True)

store = PatchStore(True)
df = load_index()
root = zarr.open_group("hydrosentry.zarr", mode="r")

sample = pd.concat([
    df[df.label == 0].sample(2, random_state=42),
    df[df.label == 1].sample(2, random_state=42)
])

def normalize(band):
    # Normalize band to 0-1 for visual color display
    b_min, b_max = np.nanpercentile(band, 2), np.nanpercentile(band, 98)
    return np.clip((band - b_min) / (b_max - b_min + 1e-6), 0, 1)

for i, row in enumerate(sample.itertuples()):
    arr = store.raw_patch(row.taluk, row.year, row.row, row.col)
    
    # Calculate geographical bounds
    center_x, center_y = rc_to_xy(root, [row.taluk], [row.row], [row.col])
    top_left_x = center_x[0] - (P/2 * 30.0) + 15.0 # Center of top-left pixel
    top_left_y = center_y[0] + (P/2 * 30.0) - 15.0
    
    # RGB Composite: Red=LST, Green=NDVI, Blue=SAR VV
    rgb = np.dstack([normalize(arr[3]), normalize(arr[0]), normalize(arr[6])])
    
    label_str = "Safe" if row.label == 0 else "OverExploited"
    base_name = f"{out_dir}/patch_{i+1}_{label_str}_{row.taluk}_{row.year}"
    
    # Save PNG Image
    mpimg.imsave(f"{base_name}.png", rgb)
    
    # Save World File (.pgw) for QGIS positioning
    with open(f"{base_name}.pgw", "w") as f:
        f.write("30.0\n")         # x-scale (30m resolution)
        f.write("0.0\n")          # y-rotation
        f.write("0.0\n")          # x-rotation
        f.write("-30.0\n")        # y-scale
        f.write(f"{top_left_x}\n")
        f.write(f"{top_left_y}\n")
        
    # Save Projection File (.prj) for UTM Zone 43N (Karnataka)
    with open(f"{base_name}.prj", "w") as f:
        f.write('PROJCS["WGS 84 / UTM zone 43N",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.01745329251994328,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]],UNIT["metre",1,AUTHORITY["EPSG","9001"]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",75],PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],PARAMETER["false_northing",0],AUTHORITY["EPSG","32643"],AXIS["Easting",EAST],AXIS["Northing",NORTH]]')

print(f"\n✅ Created 4 QGIS-ready patches in '{out_dir}/' folder.")
print("Just drag the .png files into QGIS. QGIS will read the .pgw automatically!")
