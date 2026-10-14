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

# -----------------------------------------------------------------------------
# TALUK NAME MAPPING (ALL 54 MONITORED KARNATAKA TALUKS)
# -----------------------------------------------------------------------------
TALUK_CLEAN_NAMES = {
    "bagalkot_bagalkote": "Bagalkote",
    "belagavi_bailhongal": "Bailhongal",
    "belagavi_savadatti": "Savadatti",
    "bengalururural_devanahalli": "Devanahalli",
    "bengalururural_doddaballapura": "Doddaballapura",
    "bengalururural_hoskote": "Hoskote",
    "bengalururural_nelamangala": "Nelamangala",
    "bengaluruurban_anekal": "Anekal",
    "bengaluruurban_bangaloreeast": "Bangalore-East",
    "bengaluruurban_bangalorenorth": "Bangalore-North",
    "bengaluruurban_bangaloresouth": "Bangalore-South",
    "bengaluruurban_yelahanka": "Yelahanka",
    "chamarajanagara_gundlupet": "Gundlupet",
    "chikkaballapura_bagepalli": "Bagepalli",
    "chikkaballapura_chikkaballapura": "Chikkaballapura",
    "chikkaballapura_chinthamani": "Chinthamani",
    "chikkaballapura_gauribidanur": "Gauribidanur",
    "chikkaballapura_gudibande": "Gudibande",
    "chikkaballapura_shidlagatta": "Shidlagatta",
    "chikkamagaluru_ajjampura": "Ajjampura",
    "chitradurga_challakere": "Challakere",
    "chitradurga_chitradurga": "Chitradurga",
    "chitradurga_hiriyur": "Hiriyur",
    "chitradurga_holalkere": "Holalkere",
    "chitradurga_hosadurga": "Hosadurga",
    "chitradurga_molakalmuru": "Molakalmuru",
    "davanagere_channagiri": "Channagiri",
    "davanagere_harihar": "Harihar",
    "davanagere_honnali": "Honnali",
    "davanagere_jagaluru": "Jagaluru",
    "gadag_gajendragad": "Gajendragad",
    "gadag_naragund": "Naragund",
    "hassan_arkalgud": "Arkalgud",
    "hassan_arsikere": "Arsikere",
    "hassan_channarayapatna": "Channarayapatna",
    "kolar_bangarpet": "Bangarpet",
    "kolar_kolara": "Kolar",
    "kolar_kolargoldfield": "Kolar Gold Field",
    "kolar_malur": "Malur",
    "kolar_mulabagilu": "Mulabagilu",
    "kolar_srinivaspura": "Srinivaspura",
    "ramanagara_channapatna": "Channapatna",
    "ramanagara_harohalli": "Harohalli",
    "ramanagara_kanakpura": "Kanakpura",
    "ramanagara_magadi": "Magadi",
    "tumakuru_chiknayakanahalli": "Chiknayakanahalli",
    "tumakuru_kunigal": "Kunigal",
    "tumakuru_madhugiri": "Madhugiri",
    "tumakuru_pavagada": "Pavagada",
    "tumakuru_sira": "Sira",
    "tumakuru_tiptur": "Tiptur",
    "tumakuru_tumakuru": "Tumakuru",
    "vijayanagara_hagaribommanahalli": "Hagaribommanahalli",
    "vijayanagara_kotturu": "Kotturu",
}

class DesktopGISApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Groundwater Over-Exploitation Detection System")
        self.root.geometry("1280x760")
        self.root.configure(bg="#121212")
        
        self.init_msg = ""
        self.display_to_id = {}
        self.id_to_display = {}
        self.taluks = []
        self.loto_df = None
        self.cells_df = None
        self.df = None
        
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            index_path = os.path.join(base_dir, "index.parquet")
            if os.path.exists(index_path):
                self.df = pd.read_parquet(index_path)
            
            # Load the multi-model benchmark predictions (all 54 taluks)
            merged_path = os.path.join(base_dir, "runs", "benchmarks", "final_merged_scores.csv")
            if os.path.exists(merged_path):
                self.loto_df = pd.read_csv(merged_path)
            else:
                fallback_path = os.path.join(base_dir, "runs", "clean_dl_v4", "loto_scores.csv")
                self.loto_df = pd.read_csv(fallback_path) if os.path.exists(fallback_path) else None

            # Calculate relative rank using Headline Logistic Regression
            if self.loto_df is not None:
                lr_col = "LogisticRegression" if "LogisticRegression" in self.loto_df.columns else "LR_all_bands"
                self.loto_df["lr_rank"] = self.loto_df[lr_col].rank(ascending=False, method="min").astype(int)

            cells_path = os.path.join(base_dir, "runs", "clean_dl_v4", "cells_loto.parquet")
            if os.path.exists(cells_path):
                self.cells_df = pd.read_parquet(cells_path)
            
            # Map all 54 taluks with clean names
            available_raw = []
            if self.loto_df is not None:
                available_raw = sorted(self.loto_df["taluk"].unique().tolist())
            elif self.df is not None:
                available_raw = sorted(self.df["taluk_id"].unique().tolist())
            
            for raw in available_raw:
                disp = TALUK_CLEAN_NAMES.get(raw, raw.replace("_", " ").title())
                self.display_to_id[disp] = raw
                self.id_to_display[raw] = disp
                if disp not in self.taluks:
                    self.taluks.append(disp)
                    
            self.taluks = sorted(self.taluks)
            
            self.init_msg = (
                f"[DATABASE] Loaded {len(self.taluks)} Target Karnataka Taluks.\n"
                f"[STATUS] Ready. Select a taluk and click 'Detect Over-Exploited Areas'.\n"
            )
        except Exception as e:
            self.taluks = ["Ajjampura", "Bangalore-North", "Chikkaballapura", "Chitradurga", "Kolar"]
            self.display_to_id = {t: t.lower() for t in self.taluks}
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
        
        # ------------------- LEFT CONTROL & LOG PANEL -------------------
        left_frame = ttk.Frame(root, padding=20)
        left_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        ttk.Label(left_frame, text="GROUNDWATER DETECTION", font=("Segoe UI", 19, "bold"), foreground="#00adb5").pack(pady=4)
        ttk.Label(left_frame, text="Decadal Trend Model (Backed by Logistic Regression)", font=("Segoe UI", 10, "bold"), foreground="#00adb5").pack(pady=(0, 15))
        
        input_group = ttk.LabelFrame(left_frame, text="Region Selection")
        input_group.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(input_group, text="Select Taluk:").grid(row=0, column=0, padx=10, pady=12, sticky="e")
        self.taluk_var = tk.StringVar()
        if self.taluks:
            default_t = "Ajjampura" if "Ajjampura" in self.taluks else self.taluks[0]
            self.taluk_var.set(default_t)
            
        self.cb_taluk = ttk.Combobox(input_group, textvariable=self.taluk_var, values=self.taluks, state="readonly", width=25, font=("Segoe UI", 11))
        self.cb_taluk.grid(row=0, column=1, padx=10, pady=12)
        
        self.show_all_var = tk.BooleanVar(value=False)
        self.chk_all = ttk.Checkbutton(input_group, text="Show ALL areas (Log may lag)", variable=self.show_all_var)
        self.chk_all.grid(row=1, column=0, columnspan=2, pady=5, sticky="w", padx=10)
        
        self.btn = ttk.Button(left_frame, text="🔍 Detect Over-Exploited Areas", command=self.run_detection)
        self.btn.pack(fill=tk.X, pady=8)
        
        self.btn_folium = ttk.Button(left_frame, text="🌍 Export Interactive Web Map", command=self.generate_folium_map, style="Folium.TButton")
        self.btn_folium.pack(fill=tk.X, pady=8)
        
        self.log = scrolledtext.ScrolledText(left_frame, width=70, height=19, bg="#1b1b1b", fg="#00ff66", font=("Consolas", 9), borderwidth=0)
        self.log.pack(fill=tk.BOTH, expand=True, pady=12)
        self.log.insert(tk.END, "[System initialized] Ready to scan.\n")
        self.log.insert(tk.END, self.init_msg)
        
        # ------------------- RIGHT VISUALIZATION PANEL -------------------
        right_frame = ttk.Frame(root, padding=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.fig = Figure(figsize=(7, 8), dpi=100, facecolor="#181818")
        self.ax1 = self.fig.add_subplot(211)
        self.ax2 = self.fig.add_subplot(212)
        self.ax1.axis("off"); self.ax2.axis("off")
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.setup_empty_plots()

    def setup_empty_plots(self):
        self.ax1.clear(); self.ax2.clear()
        self.ax1.axis("off"); self.ax2.axis("off")
        self.ax1.set_facecolor("#181818"); self.ax2.set_facecolor("#181818")
        self.ax1.set_title("Satellite Input Map", color="white", pad=10, fontsize=12, fontweight='bold')
        self.ax2.set_title("Over-Exploited Stress Map", color="#ff2e63", pad=10, fontsize=12, fontweight='bold')
        self.fig.tight_layout()
        self.canvas.draw()
        
    def run_detection(self):
        display_taluk = self.taluk_var.get()
        if not display_taluk: return
        raw_taluk_id = self.display_to_id.get(display_taluk)
        if not raw_taluk_id or self.cells_df is None: return
        
        loto_row = None
        if self.loto_df is not None:
            sub = self.loto_df[self.loto_df.taluk == raw_taluk_id]
            if len(sub) > 0:
                loto_row = sub.iloc[0]

        # Model scores
        lr_score = float(loto_row['LogisticRegression']) if loto_row is not None and 'LogisticRegression' in loto_row else 0.50
        ga_score = float(loto_row['GA_SVM']) if loto_row is not None and 'GA_SVM' in loto_row else 0.50
        gru_score = float(loto_row['DL_Bi_GRU']) if loto_row is not None and 'DL_Bi_GRU' in loto_row else 0.50
        knn_score = float(loto_row['kNN_coords']) if loto_row is not None and 'kNN_coords' in loto_row else 0.50
        gt_label = int(loto_row['label']) if loto_row is not None else 1
        
        lr_rank = int(loto_row['lr_rank']) if loto_row is not None and 'lr_rank' in loto_row else 27
        total_taluks = len(self.loto_df) if self.loto_df is not None else 54

        gt_desc = "OVER-EXPLOITED (CGWB Ground Truth)" if gt_label == 1 else "SAFE / SEMI-CRITICAL (CGWB Ground Truth)"
        
        if lr_score >= 0.50:
            status_desc = "CRITICAL OVER-EXPLOITATION"
            status_icon = "⚠️"
        else:
            status_desc = "SAFE / STABLE AQUIFER"
            status_icon = "✅"

        self.log.delete(1.0, tk.END)
        self.log.insert(tk.END, "======================================================================\n")
        self.log.insert(tk.END, "  GROUNDWATER OVER-EXPLOITATION DETECTION SYSTEM\n")
        self.log.insert(tk.END, "======================================================================\n")
        self.log.insert(tk.END, f"Target Region        : {display_taluk.upper()}\n")
        self.log.insert(tk.END, f"Ground Truth Label   : {gt_desc}\n")
        self.log.insert(tk.END, "----------------------------------------------------------------------\n")
        self.log.insert(tk.END, f"Groundwater Stress   : {lr_score * 100:.1f}%\n")
        self.log.insert(tk.END, f"Final Assessment     : {status_icon} {status_desc}\n")
        self.log.insert(tk.END, "======================================================================\n")
        self.root.update()
        time.sleep(0.2)
        
        taluk_patches = self.cells_df[self.cells_df.taluk == raw_taluk_id].copy()
        if len(taluk_patches) == 0:
            self.log.insert(tk.END, "❌ No spatial grid data found for this Taluk.\n")
            return
            
        self.log.insert(tk.END, f"📡 Mapped {len(taluk_patches)} spatial grid blocks across taluk.\n")
        self.root.update()
        
        taluk_patches['prob'] = taluk_patches['pct'] / 100.0
        overexploited = taluk_patches[taluk_patches['prob'] >= 0.50].sort_values(by='prob', ascending=False)
        
        self.log.insert(tk.END, f"\n{status_icon} HIGH-STRESS SECTORS IN {display_taluk.upper()} (DETECTED HOTSPOTS):\n")
        self.log.insert(tk.END, "-"*70 + "\n")
        
        if len(overexploited) == 0:
            self.log.insert(tk.END, "  [SAFE] No critical over-exploited hotspots detected in this taluk.\n")
        else:
            top_print = overexploited if self.show_all_var.get() else overexploited.head(20) 
            
            for i, (_, row) in enumerate(top_print.iterrows()):
                patch_lat, patch_lon = row.lat, row.lon
                self.log.insert(tk.END, f" Rank {i+1:04d} | GPS: {patch_lat:.4f}, {patch_lon:.4f} | Groundwater Stress: {row.prob*100:.1f}%\n")
                
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
            
            full_ndvi = np.asarray(arr[0], dtype=np.float32) 
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
            self.ax1.imshow(full_ndvi_display, cmap="YlGn", vmin=0, vmax=0.75)
            self.ax1.set_title(f"Satellite Input Map ({display_taluk})", color="white", pad=10, fontsize=12, fontweight='bold')
            
            self.ax2.clear(); self.ax2.axis("off")
            self.ax2.imshow(full_ndvi_display, cmap="gray", alpha=0.5)
            
            has_oe = (not np.all(np.isnan(heatmap_display))) and (np.nanmax(heatmap_display) >= 0.50)
            if has_oe:
                # Classic thermal overlay: Yellow -> Orange -> Deep Red for severe stress
                self.ax2.imshow(heatmap_display, cmap="YlOrRd", alpha=0.88, vmin=0.50, vmax=0.96)
                self.ax2.set_title("Over-Exploited Stress Map", color="#ff2e63", pad=10, fontsize=12, fontweight='bold')
            else:
                self.ax2.set_title("Aquifer Assessment: Sustainable Aquifer", color="#2ecc71", pad=10, fontsize=12, fontweight='bold')
                self.ax2.text(0.5, 0.5, "[SAFE] Aquifer Stable\nNo Critical Depletion Hotspots", color="#2ecc71", ha="center", va="center", fontsize=11, fontweight="bold")
            
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
            
            overexploited = taluk_patches[taluk_patches['prob'] >= 0.50].sort_values(by='prob', ascending=False)
            if len(overexploited) == 0:
                self.log.insert(tk.END, "❌ No over-exploited areas to map.\n")
                return
                
            top_print = overexploited.head(50) 
            
            center_lat = float(top_print.iloc[0].lat)
            center_lon = float(top_print.iloc[0].lon)
            
            m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}", attr="Google")
            
            for i, (_, row) in enumerate(top_print.iterrows()):
                patch_lat, patch_lon = row.lat, row.lon
                
                if i < 20:
                    color = "#ff2e63"
                    status = "SEVERE (Top 20 Hotspot)"
                else:
                    color = "#f39c12"
                    status = "HIGH WARNING"
                    
                folium.CircleMarker(
                    location=[patch_lat, patch_lon],
                    radius=8,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.85,
                    popup=f"<b>GPS: {patch_lat:.4f}, {patch_lon:.4f}</b><br>Model: Decadal Logistic Regression<br>Stress Level: {row.prob*100:.1f}%<br>Status: {status}"
                ).add_to(m)
                
            map_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"Interactive_Map_{raw_taluk_id}.html")
            m.save(map_path)
            
            self.log.insert(tk.END, f"[SUCCESS] Map saved successfully!\n")
            self.log.insert(tk.END, f"🌍 Opening Web Browser...\n")
            webbrowser.open(f"file://{os.path.abspath(map_path)}")
            
        except ImportError:
            self.log.insert(tk.END, f"❌ 'folium' is not installed.\n[ACTION REQUIRED] Run: pip install folium\n")
        except Exception as e:
            self.log.insert(tk.END, f"❌ Failed to generate map: {e}\n")
            
        self.log.yview(tk.END)

if __name__ == "__main__":
    root = tk.Tk()
    app = DesktopGISApp(root)
    root.mainloop()
