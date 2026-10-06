import os
import pandas as pd
import geopandas as gpd
import rasterio
import glob

def audit_dataset():
    print("=== PHASE 0 AUDIT ===")
    
    # 1. Raster Audit
    print("\n--- Raster Audit ---")
    tif_files = glob.glob("HydroSentry_Full_State/*.tif")
    if tif_files:
        sample_tif = tif_files[0]
        with rasterio.open(sample_tif) as src:
            print(f"Sample Raster: {os.path.basename(sample_tif)}")
            print(f"CRS: {src.crs}")
            print(f"Resolution: {src.res}")
            print(f"Shape: {src.shape}")
            print(f"NoData Value: {src.nodata}")
            print(f"Band Count: {src.count}")
            print(f"Bounds: {src.bounds}")
            print(f"Total TIFs found: {len(tif_files)}")
    else:
        print("No TIF files found in HydroSentry_Full_State")

    # 2. CSV Audit & Well Joining
    print("\n--- Well vs Taluk Analysis ---")
    try:
        # Load Wells
        gwl_df = pd.read_csv(r"manual_dataset&ground_truth\gwl_manual_quarterly_cgwb_ka_2021_2025.csv")
        wells = gpd.GeoDataFrame(
            gwl_df, geometry=gpd.points_from_xy(gwl_df.Longitude, gwl_df.Latitude), crs="EPSG:4326"
        )
        print(f"Total well records (2021-2025): {len(wells)}")
        print(f"Unique wells: {wells['Station'].nunique()}")
        
        # Load Taluks
        taluks = gpd.read_file(r"Taluk\Taluk.shp")
        if taluks.crs is None or taluks.crs.to_string() != "EPSG:4326":
            taluks = taluks.to_crs("EPSG:4326")
            
        # Load OE Labels
        oe_df = pd.read_csv(r"manual_dataset&ground_truth\gwra_2025_oe_taluks.csv")
        oe_labels = oe_df[oe_df['label_binary'].isin([0, 1])] # Only OE and Safe
        
        # Merge Labels to Shapefile
        # Note: We assume the shapefile has a column like 'KGISTalukN' or 'KGISDist_1' or 'TALUK_NAME'
        taluk_col = None
        for col in ['KGISTalukN', 'TALUK', 'Taluk_Name', 'KGISDist_1']:
            if col in taluks.columns:
                taluk_col = col
                break
        
        if taluk_col:
            # Simple spatial join
            joined = gpd.sjoin(wells, taluks, how="inner", predicate="within")
            
            # Map labels
            taluk_names = oe_labels['taluk'].str.lower().str.strip().tolist()
            labels = dict(zip(oe_labels['taluk'].str.lower().str.strip(), oe_labels['category']))
            
            joined['category'] = joined[taluk_col].str.lower().str.strip().map(labels)
            
            counts = joined['category'].value_counts()
            print("\nWell Records inside Labeled Taluks:")
            print(counts)
            
            unique_wells_by_cat = joined.groupby('category')['Station'].nunique()
            print("\nUnique Wells inside Labeled Taluks:")
            print(unique_wells_by_cat)
            
        else:
            print(f"Could not find Taluk name column in shapefile. Columns: {taluks.columns}")
            
        print("\n--- OE Taluk Discrepancy Check ---")
        oe_count = len(oe_labels[oe_labels['label_binary'] == 1])
        safe_count = len(oe_labels[oe_labels['label_binary'] == 0])
        print(f"Taluks in CSV with OE label (1): {oe_count}")
        print(f"Taluks in CSV with Safe label (0): {safe_count}")
        print(f"Total labeled taluks in CSV: {oe_count + safe_count}")
        
    except Exception as e:
        print(f"Error during spatial join/audit: {e}")

if __name__ == '__main__':
    audit_dataset()
