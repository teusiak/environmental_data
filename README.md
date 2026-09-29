# Environmental Data Analysis: Clustering & Geospatial Visualization

Two coursework projects (Scientific Programming and Data Analysis, semester exchange at Hanyang University / Warsaw University of Technology) exploring environmental data with Python: unsupervised clustering of urban air-quality data, and geospatial visualization of a numerical weather forecast field.

## 1. Air quality clustering (`clustering/`)

K-means clustering of hourly air-quality measurements from an urban background monitoring station in Lublin, Poland, to identify recurring pollution regimes (e.g. traffic-related vs. background conditions).

- **Data**: hourly concentrations of 7 pollutants (NO2, CO, O3, PM10, PM2.5, SO2, C6H6) from the Polish national air-quality monitoring network (GIOŚ), station `LbLubObywate` (`clustering/data/LbLubObywate.xlsx`).
- **Method**: features standardized (`StandardScaler`), optimal cluster count selected with the elbow method, final clustering with `KMeans` (k=5), and PCA used to project clusters into 2D for visualization. A polar plot cross-references cluster membership against hour-of-day to reveal diurnal (e.g. rush-hour) pollution patterns.
- **Script**: `clustering/cluster_air_quality.py`
- **Results**: `clustering/results/` — elbow curve, PCA projection, polar hour-of-day plot, and pairwise pollutant scatter plots colored by cluster.

## 2. Wind field visualization (`wind_visualization/`)

Geospatial visualization of a numerical weather prediction (NWP) forecast field, plotting wind streamlines at multiple geographic scales from raw NetCDF model output.

- **Data**: a gridded NWP forecast in NetCDF format (`gv-2016040918.nc`), containing wind speed/direction on pressure levels over time. **Not included in this repo** (~190 MB, not authored by me) — the script expects a file with matching variable names (`WDIR`, `WSPD`, `lat`, `lon`, `time`, `level`) at the path set in the `data_path` variable.
- **Method**: loads the dataset with `xarray`, converts wind direction/speed to U/V vector components, and renders wind streamlines with `cartopy` + `matplotlib.streamplot` at two scales — a global Robinson projection and a regional (Poland) PlateCarree projection — plus a wind-speed time series at the nearest grid point to a chosen location.
- **Script**: `wind_visualization/wind_field_streamlines.py`
- **Results**: `wind_visualization/results/` — combined global/regional streamline map with time series panel.

## Tech stack

Python — pandas, NumPy, scikit-learn (KMeans, PCA, StandardScaler), matplotlib, xarray, cartopy, scipy (interpolation), openpyxl.

## Running it

```bash
pip install -r requirements.txt
```

- Clustering: from `clustering/`, run `python cluster_air_quality.py` (reads `data/LbLubObywate.xlsx`, writes plots to `results/` and a clustered copy of the spreadsheet).
- Wind visualization: from `wind_visualization/`, set `data_path` in `wind_field_streamlines.py` to a local NetCDF forecast file, then run the script.

## Author

Teodor Noga — [linkedin.com/in/teodor-noga-5a58231a9](https://www.linkedin.com/in/teodor-noga-5a58231a9)
