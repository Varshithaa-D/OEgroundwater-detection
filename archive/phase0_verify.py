"""
Phase 0 — Final Verification (3 scripts in 1)
Run this ONCE. Paste the full output back.
"""
import re, os, glob, pandas as pd, geopandas as gpd, rasterio

def norm(s):
    return re.sub(r"[^a-z]", "", str(s).lower())

print("=" * 70)
print("PART 1: CHECK LABELS — Taluk matching & wells per taluk")
print("=" * 70)

# --- Load shapefile (district is numeric code, not name) ---
taluk_shp = gpd.read_file("Taluk/Taluk.shp")
# Shapefile columns: KGISTalukN (taluk name), KGISDistri (district code)

# --- Load GWRA labels CSV ---
labels = pd.read_csv("manual_dataset&ground_truth/gwra_2025_oe_taluks.csv")
# CSV columns: district, taluk, label_binary, category

# The shapefile has NO district name, only a code. So we match on taluk name only.
# But taluk names CAN repeat across districts. To be safe, we match the 54 GEE names directly.
GEE_TALUKS = [
    'Bagalkote', 'Bailhongal', 'Savadatti', 'Devanahalli', 'Doddaballapura', 'Hoskote', 'Nelamangala',
    'Anekal', 'Bangalore-East', 'Bangalore-North', 'Bangalore-South', 'Yelahanka',
    'Harohalli', 'Magadi', 'Channapatna', 'Kanakpura', 'Gundlupet',
    'Bagepalli', 'Chikkaballapura', 'Chinthamani', 'Gauribidanur', 'Gudibande', 'Shidlagatta',
    'Ajjampura', 'Challakere', 'Chitradurga', 'Hiriyur', 'Holalkere', 'Hosadurga', 'Molakalmuru',
    'Channagiri', 'Jagaluru', 'Harihar', 'Honnali', 'Gajendragad', 'Naragund',
    'Arsikere', 'Channarayapatna', 'Arkalgud', 'Bangarpet', 'Kolar Gold Field', 'Kolara', 'Malur',
    'Mulabagilu', 'Srinivaspura', 'Chiknayakanahalli', 'Madhugiri', 'Sira', 'Tiptur', 'Tumakuru',
    'Pavagada', 'Kunigal', 'Hagaribommanahalli', 'Kotturu'
]

# Filter shapefile to our 54 taluks
study_taluks = taluk_shp[taluk_shp['KGISTalukN'].isin(GEE_TALUKS)].copy()
print(f"\nShapefile taluks matched: {len(study_taluks)} (expect 54)")

# Check for duplicate taluk names in shapefile
dupes = study_taluks['KGISTalukN'].value_counts()
dupes_multi = dupes[dupes > 1]
if len(dupes_multi) > 0:
    print(f"WARNING: Duplicate taluk names in shapefile: {dupes_multi.to_dict()}")
else:
    print("No duplicate taluk names in filtered shapefile. Good.")

# Attach labels from CSV (match on normalized taluk name)
labels['taluk_norm'] = labels['taluk'].apply(norm)
study_taluks['taluk_norm'] = study_taluks['KGISTalukN'].apply(norm)
study_taluks = study_taluks.merge(
    labels[['taluk_norm', 'label_binary', 'category', 'district']],
    on='taluk_norm', how='left'
)
print(f"\nLabel distribution in our 54 taluks:")
print(study_taluks['category'].value_counts())
print(f"\nlabel_binary distribution:")
print(study_taluks['label_binary'].value_counts())

# --- Spatial join wells to the 54 taluks ---
gwl = pd.read_csv("manual_dataset&ground_truth/gwl_manual_quarterly_cgwb_ka_2021_2025.csv")

# Get unique wells only (deduplicate by Station + Lat + Lon)
wells_unique = gwl.drop_duplicates(subset='Station')[['Station', 'Latitude', 'Longitude', 'District', 'Tehsil']].copy()
wells_gdf = gpd.GeoDataFrame(
    wells_unique,
    geometry=gpd.points_from_xy(wells_unique['Longitude'], wells_unique['Latitude']),
    crs="EPSG:4326"
)

# Reproject both to UTM 43N for accurate spatial join
study_taluks_utm = study_taluks.to_crs("EPSG:32643")
wells_utm = wells_gdf.to_crs("EPSG:32643")

joined = gpd.sjoin(wells_utm, study_taluks_utm[['KGISTalukN', 'label_binary', 'category', 'geometry']],
                   how='inner', predicate='within')

print(f"\n--- Wells falling inside the 54 study taluks ---")
print(f"Total unique wells inside study area: {len(joined)}")
print(f"\nWells per category:")
print(joined['category'].value_counts())

print(f"\nWells per taluk (top 20):")
per_taluk = joined.groupby(['KGISTalukN', 'category']).size().reset_index(name='n_wells')
per_taluk = per_taluk.sort_values('n_wells', ascending=False)
print(per_taluk.head(20).to_string(index=False))

print(f"\nWells per taluk (bottom 10):")
print(per_taluk.tail(10).to_string(index=False))

# Save the 54 labeled taluks as a GeoPackage for later use
study_taluks['key'] = study_taluks['district'].apply(norm) + "_" + study_taluks['KGISTalukN'].apply(norm)
study_taluks.to_crs("EPSG:32643").to_file("labeled_taluks.gpkg", driver="GPKG")
print("\nSaved labeled_taluks.gpkg (54 taluks with labels and geometry)")

print("\n" + "=" * 70)
print("PART 2: RASTER INVENTORY")
print("=" * 70)

files = glob.glob("HydroSentry_Full_State/*.tif")
groups = {}
for f in files:
    name = os.path.basename(f)
    m = re.match(r"(HS_[A-Za-z]+(?:_\d+m)?_?)(\d{4})?", name)
    if m:
        groups.setdefault((m.group(1).rstrip('_'), m.group(2)), []).append(f)
    else:
        groups.setdefault(("UNKNOWN", None), []).append(f)

for k, v in sorted(groups.items(), key=lambda x: str(x[0])):
    with rasterio.open(v[0]) as s:
        print(f"{k[0]:30s} year={k[1]}  tiles={len(v):2d}  bands={s.count}  "
              f"desc={s.descriptions}  res={s.res}  dtype={s.dtypes[0]}  nodata={s.nodata}")

print("\n" + "=" * 70)
print("PART 3: GWL CSV HEADER & SAMPLE")
print("=" * 70)

print("\nColumns:", gwl.columns.tolist())
print(f"\nDate column sample: {gwl['Data Acquisition Time'].head(3).tolist()}")
print(f"GWL column sample:  {gwl['Groundwater Level Quarterly Manual (meter)'].head(3).tolist()}")
print(f"Lat range: {gwl['Latitude'].min()} to {gwl['Latitude'].max()}")
print(f"Lon range: {gwl['Longitude'].min()} to {gwl['Longitude'].max()}")
print(f"Year range: {pd.to_datetime(gwl['Data Acquisition Time']).dt.year.min()} to "
      f"{pd.to_datetime(gwl['Data Acquisition Time']).dt.year.max()}")

# Also check telemetry CSV header
tel = pd.read_csv("manual_dataset&ground_truth/gwl_tel_6_hourly_karnataka_ka_2021_2025.csv", nrows=3)
print(f"\nTelemetry CSV columns: {tel.columns.tolist()}")

print("\n" + "=" * 70)
print("PHASE 0 VERIFICATION COMPLETE")
print("=" * 70)
