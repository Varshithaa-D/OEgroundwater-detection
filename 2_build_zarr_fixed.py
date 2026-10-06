# 2_build_zarr_fixed.py
import glob, os, json, time, shutil
import numpy as np, geopandas as gpd, rasterio, zarr
from rasterio.merge import merge
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import from_origin
from numcodecs import Blosc

RASTER_DIR = "HydroSentry_Full_State"
OUT_ZARR = "hydrosentry.zarr"

# Delete the Zarr store if it exists so we start fresh
if os.path.exists(OUT_ZARR):
    print("Deleting old Zarr store...")
    shutil.rmtree(OUT_ZARR)
    
YEARS = list(range(2005, 2012)) + list(range(2013, 2026))
BANDS = ["NDVI", "NDWI", "EVI", "LST", "rainfall_mm", "ET_mm", "VV", "VH", "SAR_valid", "elevation", "slope", "aspect_sin", "aspect_cos"]

def bounds_intersect(b1, b2):
    """Check if two bounding boxes (minx, miny, maxx, maxy) intersect."""
    return not (b1[2] < b2[0] or b1[0] > b2[2] or b1[3] < b2[1] or b1[1] > b2[3])

def get_tiles(pattern, bounds):
    paths = []
    for p in glob.glob(os.path.join(RASTER_DIR, pattern)):
        with rasterio.open(p) as src:
            if bounds_intersect(src.bounds, bounds):
                paths.append(p)
                
    if not paths: return None
    
    srcs = [rasterio.open(p) for p in paths]
    arr, _ = merge(srcs, bounds=bounds, res=30, resampling=Resampling.average, nodata=np.nan, dtype="float32")
    [s.close() for s in srcs]
    return arr

print("=" * 60)
print("BUILD ZARR STORE (FIXED)")
print("=" * 60)

taluk = gpd.read_file("labeled_taluks.gpkg").to_crs(32643)
root = zarr.open_group(OUT_ZARR, mode="w")
comp = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
quality_stats = []

t_start = time.time()
for tidx, (_, t) in enumerate(taluk.iterrows()):
    key = t["key"]
    b = tuple(t.geometry.bounds)
    b = (np.floor(b[0]/30)*30, np.floor(b[1]/30)*30, np.ceil(b[2]/30)*30, np.ceil(b[3]/30)*30)
    H, W = int((b[3]-b[1])/30), int((b[2]-b[0])/30)
    
    mask = rasterize([(t.geometry, 1)], out_shape=(H, W), transform=from_origin(b[0], b[3], 30, 30), fill=0).astype(bool)
    static = get_tiles("HS_StaticTerrain_30m*.tif", b)
    if static is None: static = np.zeros((4, H, W), dtype="float32")
    
    g = root.create_group(key)
    g.attrs.update({"origin": [float(b[0]), float(b[3])], "label": int(t["label_binary"]), "shape": [H, W]})
    g.create_dataset("mask", data=mask, chunks=(128, 128), compressor=comp)
    
    for yr in YEARS:
        opt = get_tiles(f"HS_OptDynamic_{yr}*.tif", b)
        sar = get_tiles(f"HS_SAR_{yr}*.tif", b)
        if opt is None: opt = np.full((6, H, W), np.nan, dtype="float32")
        if sar is None: sar = np.full((3, H, W), np.nan, dtype="float32")
        
        stack = np.concatenate([opt, sar, static], axis=0)
        
        # --- PHASE 1 FIXES ---
        # 1. Mask water bodies (NDWI > 0.3). NDWI is band index 1.
        water_mask = stack[1] > 0.3
        stack[:, water_mask] = np.nan 
        
        # 2. Mask outside taluk
        stack[:, ~mask] = np.nan
        
        # 3. Prevent Float16 overflow (clip to float16 max)
        stack = np.clip(stack, a_min=-65500, a_max=65500)
        
        # Quality report stats (mean of valid pixels)
        # Suppress warnings for all-NaN slices
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            valid_means = np.nanmean(stack, axis=(1,2))
        quality_stats.append([key, yr] + valid_means.tolist())

        g.create_dataset(str(yr), data=stack.astype("float16"), chunks=(len(BANDS), 128, 128), compressor=comp)
        
    elapsed = time.time() - t_start
    print(f"[{tidx+1:2d}/{len(taluk)}] Built {key:40s} | {elapsed:.0f}s elapsed")

json.dump(BANDS, open("channel_names.json", "w"))
pd.DataFrame(quality_stats, columns=["taluk", "year"] + BANDS).to_csv("phase1_quality_report.csv", index=False)
print("\nZarr rebuilt safely. Quality report saved.")
