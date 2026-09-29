import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import matplotlib.gridspec as gridspec

# === SETTINGS ===
#data_path = "/dta/ekoprognoza/gv-2016040918.nc" #for school pc 
data_path = "gv-2016040918.nc" #for home pc with file in same folder
z_index = 3  # 850 hPa
target_hour = 6
point_lat, point_lon = 52.0, 19.0
author_name = "Teodor Noga"
submission_date = "2025-04-13"

# === LOAD DATA ===
ds = xr.open_dataset(data_path)
lat = ds['lat'].values
lon = ds['lon'].values
time = ds['time'].values
pressure_level = ds['level'].isel(z=z_index).values

# Meshgrid lat/lon if needed
if lat.ndim == 1 and lon.ndim == 1:
    lon2d, lat2d = np.meshgrid(lon, lat, indexing='ij')
else:
    lon2d = lon
    lat2d = lat

# Wind direction and speed
WDIR = ds['WDIR'].isel(z=z_index, time=target_hour)
WSPD = ds['WSPD'].isel(z=z_index, time=target_hour)

# Calculate U/V
angle_rad = np.deg2rad(WDIR)
U = -WSPD * np.sin(angle_rad)
V = -WSPD * np.cos(angle_rad)

# === PLOTTING ===
fig = plt.figure(figsize=(15, 10))
gs = gridspec.GridSpec(2, 2, height_ratios=[1.3, 1])

# === 1. GLOBAL MAP ===
ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.Robinson())
ax1.set_title(f"Global Wind Streamlines\n{int(pressure_level)} hPa @ {str(time[target_hour])[:16]}")
ax1.set_global()
ax1.coastlines()
ax1.add_feature(cfeature.BORDERS, linewidth=0.5)
ax1.set_facecolor("white")

strm1 = ax1.streamplot(
    lon2d, lat2d, U, V,
    density=1.5,
    color=WSPD.values,
    cmap='viridis',
    norm=mcolors.Normalize(0, 30),
    transform=ccrs.PlateCarree(),
    linewidth=1
)
cbar1 = fig.colorbar(strm1.lines, ax=ax1, orientation='horizontal', pad=0.05, shrink=0.8)
cbar1.set_label("Wind Speed [m/s]")

# === 2. REGIONAL MAP (POLAND) ===
ax2 = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
ax2.set_title(f"Wind Streamlines over Poland ({int(pressure_level)} hPa)")
ax2.set_extent([13, 25, 49, 55], crs=ccrs.PlateCarree())
ax2.coastlines()
ax2.add_feature(cfeature.BORDERS, linewidth=0.5)
ax2.set_facecolor("white")

strm2 = ax2.streamplot(
    lon2d, lat2d, U, V,
    density=1.7,
    color=WSPD.values,
    cmap='viridis',
    norm=mcolors.Normalize(0, 28),
    transform=ccrs.PlateCarree(),
    linewidth=1
)
cbar2 = fig.colorbar(strm2.lines, ax=ax2, orientation='horizontal', pad=0.05, shrink=0.8)
cbar2.set_label("Wind Speed [m/s]")

# === 3. TIME SERIES ===
# Nearest grid point
dist = (lat2d - point_lat)**2 + (lon2d - point_lon)**2
y_idx, x_idx = np.unravel_index(np.argmin(dist), dist.shape)
actual_lat = lat2d[y_idx, x_idx]
actual_lon = lon2d[y_idx, x_idx]

ws_series = ds['WSPD'].isel(z=z_index, y=y_idx, x=x_idx)

ax3 = fig.add_subplot(gs[1, :])
ax3.plot(time, ws_series.values, marker='o')
ax3.set_title(f"Wind Speed at 850 hPa, ~{actual_lat:.1f}°N {actual_lon:.1f}°E (nearest grid)")
ax3.set_ylabel("Wind Speed [m/s]")
ax3.set_xlabel("Forecast Time")
ax3.grid(True)

# === FOOTER ===
fig.text(0.01, 0.01, f"Wind Field Analysis – Streamlines\nAuthor: {author_name}    Date: {submission_date}    Assignment #2",
         fontsize=9, ha='left')

plt.subplots_adjust(hspace=0.5, top=0.92, bottom=0.08, wspace=0.3)
plt.savefig("wind_streamlines_assignment2_FINAL.png", bbox_inches='tight')
plt.show()

print("Level mapping:")
for i, p in enumerate(ds['level'].values):
    print(f"Index {i} → {p} hPa")