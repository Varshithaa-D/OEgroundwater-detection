import pandas as pd, geopandas as gpd
from common import GPKG, GPKG_KEY

telem = pd.read_csv("manual_dataset&ground_truth/gwl_tel_6_hourly_karnataka_ka_2021_2025.csv")
LV = "Groundwater Level Telemetry 6 Hourly (meter)"
telem[LV] = pd.to_numeric(telem[LV], errors="coerce")
telem["t"] = pd.to_datetime(telem["Data Acquisition Time"], dayfirst=True, errors="coerce")

# Unique stations with coords
stations = (telem.dropna(subset=["Latitude", "Longitude"])
            .groupby("Station").agg(lat=("Latitude","first"), lon=("Longitude","first"),
                                    n=("Station","count"), n_valid=(LV,"count"),
                                    median_depth=(LV,"median")).reset_index())
print(f"Unique stations: {len(stations)}")

# Spatial join to labeled taluks
tg = gpd.read_file(GPKG).to_crs(4326)[[GPKG_KEY, "geometry"]]
pts = gpd.GeoDataFrame(stations, geometry=gpd.points_from_xy(stations.lon, stations.lat), crs=4326)
inside = gpd.sjoin(pts, tg, predicate="within")
print(f"\nStations inside the 54 taluks: {len(inside)}")
print(inside[["Station", GPKG_KEY, "n_valid", "median_depth"]].to_string(index=False))
