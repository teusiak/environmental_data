# Machine Learning Techniques in Environmental Analysis 
# Full Assignment Script - Cluster Analysis for Urban Background Station (Lublin)

import pandas as pd
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# --------------------- SETTINGS ---------------------
station_code = 'LbLubObywate' 
input_file = f'{station_code}.xlsx'
output_dir = 'plots'
os.makedirs(output_dir, exist_ok=True)

# ------------------ DATA LOADING ------------------
df = pd.read_excel(input_file)

# Main features – you can remove NO or NOx if not available
features = ['NO2', 'CO', 'O3', 'PM10', 'PM25', 'SO2', 'C6H6']

# Drop rows with missing pollutant data
df = df.dropna(subset=features)
print(f"Dataset shape after dropping NaNs: {df.shape}")

# ---------------- ELBOW METHOD ----------------
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[features])

inertias = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertias.append(kmeans.inertia_)

plt.figure()
plt.plot(K, inertias, 'bx-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.grid(True)
plt.savefig(f'{output_dir}/elbow_method.png', dpi=300, bbox_inches='tight')
plt.show()

# ---------------- FINAL CLUSTERING ----------------
optimal_k = 5  # Adjust after seeing elbow
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_data)

df.to_excel(f'{station_code}_clustered.xlsx', index=False)
print("Clustering done and saved.")

# ---------------- SCATTER PLOTS ----------------
def scatter_plot(x_col, y_col, filename):
    plt.figure(figsize=(8,6))
    scatter = plt.scatter(df[x_col], df[y_col], c=df['Cluster'], cmap='jet', s=10, alpha=0.7)
    plt.colorbar(scatter, label='Cluster')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f'{x_col} vs {y_col}')
    plt.grid(True)
    plt.savefig(f'{output_dir}/{filename}', dpi=300, bbox_inches='tight')
    plt.show()

# Standard comparisons
scatter_plot('PM10', 'SO2', 'pm10_vs_so2.png')
scatter_plot('NO2', 'O3', 'no2_vs_o3.png')
scatter_plot('PM25', 'C6H6', 'pm25_vs_c6h6.png')

# 🚦 Extra traffic-focused plots
scatter_plot('NO2', 'CO', 'no2_vs_co.png')  # NEW for traffic insight

# ---------------- PCA PLOT ----------------
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)

df['PCA1'] = pca_result[:, 0]
df['PCA2'] = pca_result[:, 1]

plt.figure(figsize=(8,6))
scatter = plt.scatter(df['PCA1'], df['PCA2'], c=df['Cluster'], cmap='jet', s=10, alpha=0.7)
plt.colorbar(scatter, label='Cluster')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('PCA Projection')
plt.grid(True)
plt.savefig(f'{output_dir}/pca_projection.png', dpi=300, bbox_inches='tight')
plt.show()

# ---------------- POLAR PLOT ----------------
# Uses precomputed hour_sin, hour_cos
df['angle'] = np.arctan2(df['hour_sin'], df['hour_cos'])
df['radius'] = df['PCA1'] + 0.2

fig, ax = plt.subplots(subplot_kw={'projection':'polar'}, figsize=(8,8))
scatter = ax.scatter(df['angle'], df['radius'], c=df['Cluster'], cmap='jet', s=10, alpha=0.7)
ax.set_xticks(np.linspace(0, 2*np.pi, 24, endpoint=False))
ax.set_xticklabels([f'{h}:00' for h in range(24)])
ax.set_title('Polar Plot (Hour vs. PCA1)')
plt.colorbar(scatter, label='Cluster')
plt.savefig(f'{output_dir}/polar_plot_pca1.png', dpi=300, bbox_inches='tight')
plt.show()

# ---------------- CLUSTER STATS ----------------
cluster_means = df.groupby('Cluster')[features].mean()
print("\nCluster Averages:\n", cluster_means)
cluster_means.to_excel(f'{output_dir}/cluster_means.xlsx')
