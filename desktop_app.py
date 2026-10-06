import sys, os, time
from datetime import datetime
import numpy as np
import tkinter as tk
from tkinter import ttk, scrolledtext
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import pandas as pd
import zarr

TARGET_TALUKS = [
    "Bagalkote", "Bailhongal", "Savadatti", "Devanahalli", "Doddaballapura", "Hoskote", "Nelamangala",
    "Anekal", "Bangalore-East", "Bangalore-North", "Bangalore-South", "Yelahanka", "Harohalli", "Magadi",
    "Gundlupet", "Bagepalli", "Chikkaballapura", "Chinthamani", "Gauribidanur", "Gudibande", "Shidlagatta",
    "Ajjampura", "Challakere", "Chitradurga", "Hiriyur", "Holalkere", "Hosadurga", "Channagiri", "Jagaluru",
    "Gajendragad", "Arsikere", "Bangarpet", "Kolar Gold Field", "Kolara", "Malur", "Mulabagilu", "Srinivaspura",
    "Chiknayakanahalli", "Madhugiri", "Sira", "Tiptur", "Tumakuru", "Hagaribommanahalli", "Kotturu"
]

class DesktopGISApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Groundwater Over-Exploitation Detection System")
        self.root.geometry("1250x750")
        self.root.configure(bg="#121212")
        
        self.init_msg = ""
        self.display_to_id = {}
        self.taluks = []
        
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            index_path = os.path.join(base_dir, "index.parquet")
            self.df = pd.read_parquet(index_path)
            
            # Load the user's ACTUAL deep learning predictions
            cells_path = os.path.join(base_dir, "runs", "final", "cells_wellhead.parquet")
            self.cells_df = pd.read_parquet(cells_path)
            
            raw_taluks = self.df['taluk_id'].unique()
            for target in TARGET_TALUKS:
                target_clean = target.lower().replace(" ", "").replace("-", "")
                for raw in raw_taluks:
                    raw_clean = raw.lower().replace(" ", "").replace("-", "").replace("_", "")
                    if target_clean in raw_clean or raw_clean.endswith(target_clean) or raw_clean.startswith(target_clean):
                        if target not in self.taluks:
                            self.taluks.append(target)
                            self.display_to_id[target] = raw
                        break
            self.taluks = sorted(self.taluks)
            self.init_msg = f"[Database] Loaded {len(self.taluks)} Target Over-Exploited Taluks.\n[Model] Loaded actual PyTorch deep learning predictions.\n"
        except Exception as e:
            self.taluks = ["Chitradurga", "Kolar"]
            self.display_to_id = {"Chitradurga": "chitradurga_chitradurga", "Kolar": "kolar_kolara"}
            self.df = None
            self.cells_df = None
            self.init_msg = f"[ERROR] Failed to load data: {e}\n"
            
        style = ttk.Style()
        try: style.theme_use('clam')
        except: pass
        style.configure("TFrame", background="#121212")
        style.configure("TLabel", background="#121212", foreground="white", font=("Segoe UI", 11))
        style.configure("TButton", font=("Segoe UI", 11, "bold"), background="#ff2e63", foreground="white")
        style.configure("Folium.TButton", font=("Segoe UI", 11, "bold"), background="#2ecc71", foreground="black")
        style.configure("TCheckbutton", background="#121212", foreground="white", font=("Segoe UI", 10))
        style.configure("TLabelframe", background="#121212", foreground="#00adb5")
        style.configure("TLabelframe.Label", background="#121212", foreground="#00adb5", font=("Segoe UI", 12, "bold"))
        
        left_frame = ttk.Frame(root, padding=20)
        left_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        ttk.Label(left_frame, text="GROUNDWATER DETECTION", font=("Segoe UI", 19, "bold"), foreground="#00adb5").pack(pady=5)
        ttk.Label(left_frame, text="Spatial Analysis Dashboard", foreground="#aaaaaa").pack(pady=5)
        
        input_group = ttk.LabelFrame(left_frame, text="Region Selection")
        input_group.pack(fill=tk.X, pady=20)
        
        ttk.Label(input_group, text="Select Taluk:").grid(row=0, column=0, padx=10, pady=15, sticky="e")
        self.taluk_var = tk.StringVar()
        if self.taluks:
            self.taluk_var.set(self.taluks[0])
        self.cb_taluk = ttk.Combobox(input_group, textvariable=self.taluk_var, values=self.taluks, state="readonly", width=25, font=("Segoe UI", 11))
        self.cb_taluk.grid(row=0, column=1, padx=10, pady=15)
        
        self.show_all_var = tk.BooleanVar(value=False)
        self.chk_all = ttk.Checkbutton(input_group, text="Show ALL areas (Log may lag)", variable=self.show_all_var)
        self.chk_all.grid(row=1, column=0, columnspan=2, pady=5, sticky="w", padx=10)
        
        self.btn = ttk.Button(left_frame, text="🔍 Detect Over-Exploited Areas", command=self.run_detection)
        self.btn.pack(fill=tk.X, pady=10)
        
        self.btn_folium = ttk.Button(left_frame, text="🌍 Export Interactive Web Map", command=self.generate_folium_map, style="Folium.TButton")
        self.btn_folium.pack(fill=tk.X, pady=10)
        
        self.log = scrolledtext.ScrolledText(left_frame, width=70, height=19, bg="#1b1b1b", fg="#00ff66", font=("Consolas", 9), borderwidth=0)
        self.log.pack(pady=15)
        self.log.insert(tk.END, "[System initialized] Ready to scan.\n")
        self.log.insert(tk.END, self.init_msg)
        
        right_frame = ttk.Frame(root, padding=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.fig = Figure(figsize=(7, 8), dpi=100, facecolor="#1e1e1e")
        self.ax1 = self.fig.add_subplot(211)
        self.ax2 = self.fig.add_subplot(212)
        self.ax1.axis("off"); self.ax2.axis("off")
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def run_detection(self):
        display_taluk = self.taluk_var.get()
        if not display_taluk: return
        raw_taluk_id = self.display_to_id.get(display_taluk)
        if not raw_taluk_id or self.cells_df is None: return
        
        self.log.delete(1.0, tk.END)
        self.log.insert(tk.END, f"[SYSTEM SCAN] Loading deep learning model outputs for {display_taluk}...\n")
        self.root.update()
        time.sleep(0.5)
        
        taluk_patches = self.cells_df[self.cells_df.taluk == raw_taluk_id].copy()
        if len(taluk_patches) == 0:
            self.log.insert(tk.END, "❌ No data found for this Taluk.\n")
            return
            
        self.log.insert(tk.END, f"📡 Extracting {len(taluk_patches)} spatial grid blocks...\n")
        self.root.update()
        
        taluk_patches['prob'] = taluk_patches['pct'] / 100.0
        overexploited = taluk_patches[taluk_patches['prob'] > 0.50].sort_values(by='prob', ascending=False)
        
        self.log.insert(tk.END, f"\n⚠️ CRITICAL AREAS DETECTED IN {display_taluk.upper()} (FROM DEEP LEARNING MODEL):\n")
        self.log.insert(tk.END, "-"*70 + "\n")
        
        if len(overexploited) == 0:
            self.log.insert(tk.END, "  [SAFE] No over-exploited areas detected in this Taluk.\n")
        else:
            top_print = overexploited if self.show_all_var.get() else overexploited.head(20) 
            
            for i, (_, row) in enumerate(top_print.iterrows()):
                patch_lat, patch_lon = row.lat, row.lon
                self.log.insert(tk.END, f" Rank {i+1:04d} | GPS: {patch_lat:.4f}, {patch_lon:.4f} | Stress Level: {row.prob*100:.1f}%\n")
                
            if not self.show_all_var.get() and len(overexploited) > 20:
                self.log.insert(tk.END, f" ... and {len(overexploited)-20} more critical areas mapped.\n")
                self.log.insert(tk.END, f" (Check 'Show ALL areas' box to display complete list)\n")
            
        self.log.insert(tk.END, "-"*70 + "\n")
        self.log.insert(tk.END, f"\n[ACTION] Generating Regional Maps...\n")
        self.root.update()
        
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            zarr_path = os.path.join(base_dir, "hydrosentry.zarr")
            root_zarr = zarr.open_group(zarr_path, mode="r")
            
            year = 2025
            arr = root_zarr[f"{raw_taluk_id}/{year}"]
            
            full_ndvi = np.asarray(arr[0]) 
            full_ndvi = np.nan_to_num(full_ndvi, nan=-1.0)
            
            heatmap = np.zeros_like(full_ndvi, dtype=np.float32)
            
            h = 16
            for _, row in taluk_patches.iterrows():
                r, c = int(row.row), int(row.col)
                r0, c0 = max(0, r - h), max(0, c - h)
                r1, c1 = min(full_ndvi.shape[0], r + h), min(full_ndvi.shape[1], c + h)
                heatmap[r0:r1, c0:c1] = float(row.prob)
                
            mask = full_ndvi == -1.0
            full_ndvi_display = full_ndvi.copy()
            full_ndvi_display[mask] = np.nan
            
            heatmap_display = heatmap.copy()
            heatmap_display[mask] = np.nan
            heatmap_display[heatmap_display < 0.50] = np.nan
            
            self.ax1.clear(); self.ax1.axis("off")
            self.ax1.imshow(full_ndvi_display, cmap="YlGn", vmin=0, vmax=0.8)
            self.ax1.set_title(f"Satellite Input Map ({display_taluk})", color="white", pad=10, fontsize=12, fontweight='bold')
            
            self.ax2.clear(); self.ax2.axis("off")
            self.ax2.imshow(full_ndvi_display, cmap="gray", alpha=0.5)
            self.ax2.imshow(heatmap_display, cmap="YlOrRd", alpha=0.9, vmin=0.5, vmax=1.0)
            self.ax2.set_title(f"Over-Exploited Stress Map", color="#ff2e63", pad=10, fontsize=12, fontweight='bold')
            
            self.fig.tight_layout()
            self.canvas.draw()
            self.log.insert(tk.END, "[SUCCESS] Regional maps rendered on dashboard.\n")
            
        except Exception as e:
            self.log.insert(tk.END, f"[ERROR] Failed to load maps: {e}\n")
            
        self.log.yview(tk.END)

    def generate_folium_map(self):
        display_taluk = self.taluk_var.get()
        if not display_taluk: return
        raw_taluk_id = self.display_to_id.get(display_taluk)
        if not raw_taluk_id or self.cells_df is None: return
        
        try:
            import folium
            import webbrowser
            
            self.log.insert(tk.END, f"\n[WEB MAP] Generating Interactive Web Map for {display_taluk}...\n")
            self.root.update()
            
            taluk_patches = self.cells_df[self.cells_df.taluk == raw_taluk_id].copy()
            taluk_patches['prob'] = taluk_patches['pct'] / 100.0
            
            overexploited = taluk_patches[taluk_patches['prob'] > 0.50].sort_values(by='prob', ascending=False)
            if len(overexploited) == 0:
                self.log.insert(tk.END, "❌ No over-exploited areas to map.\n")
                return
                
            top_print = overexploited.head(50) 
            
            center_lat = float(top_print.iloc[0].lat)
            center_lon = float(top_print.iloc[0].lon)
            
            m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}", attr="Google")
            
            for i, (_, row) in enumerate(top_print.iterrows()):
                r, c = int(row.row), int(row.col)
                patch_lat, patch_lon = row.lat, row.lon
                
                if i < 20:
                    color = "#ff2e63"
                    status = "SEVERE (Top 20)"
                else:
                    color = "#f39c12"
                    status = "HIGH WARNING"
                    
                folium.CircleMarker(
                    location=[patch_lat, patch_lon],
                    radius=8,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.8,
                    popup=f"<b>GPS: {patch_lat:.4f}, {patch_lon:.4f}</b><br>Stress Level: {row.prob*100:.1f}%<br>Status: {status}"
                ).add_to(m)
                
            map_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"Interactive_Map_{display_taluk}.html")
            m.save(map_path)
            
            self.log.insert(tk.END, f"[SUCCESS] Map saved successfully!\n")
            self.log.insert(tk.END, f"🌍 Opening Web Browser...\n")
            webbrowser.open(f"file://{map_path}")
            
        except ImportError:
            self.log.insert(tk.END, f"❌ 'folium' is not installed.\n[ACTION REQUIRED] Open your terminal and run: pip install folium\n")
        except Exception as e:
            self.log.insert(tk.END, f"❌ Failed to generate map: {e}\n")
            
        self.log.yview(tk.END)

if __name__ == "__main__":
    root = tk.Tk()
    app = DesktopGISApp(root)
    root.mainloop()
