import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pymatgen.core import Structure
# Assuming 'matten' library is installed and 'predict' is accessible
from matten.predict import predict
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.isotonic import IsotonicRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error

plt.rcParams.update({
    "font.size": 18,
    "axes.labelsize": 18,
    "axes.titlesize": 20,
    "legend.fontsize": 16,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "axes.linewidth": 1.3,
    "figure.dpi": 140,
    "legend.framealpha": 0.97
})

def get_bulk_modulus_voigt(c):
    return (c[0,0] + c[1,1] + c[2,2] + 2*(c[0,1] + c[1,2] + c[0,2])) / 9

def tensor3x3x3x3_to_voigt(c):
    voigt = np.zeros((6,6))
    voigt_map = [(0,0),(1,1),(2,2),(1,2),(0,2),(0,1)]
    for i in range(6):
        for j in range(6):
            voigt[i,j] = c[voigt_map[i][0],voigt_map[i][1],voigt_map[j][0],voigt_map[j][1]]
    return voigt

def parse_system_pressure(folder_name):
    if '_' not in folder_name or not folder_name.endswith('GPa'):
        return None, None
    system, pstr = folder_name.split('_', 1)
    try:
        pressure = float(pstr.replace('GPa',''))
        return system, pressure
    except:
        return system, None

# --- Load Data ---
base_dir = 'gnn_dataset'
results_data_list = [] # Renamed to avoid conflict with 'results' folder name

for dirname in sorted(os.listdir(base_dir)):
    folder = os.path.join(base_dir, dirname)
    if not os.path.isdir(folder): continue
    system, pressure = parse_system_pressure(dirname)
    if system is None or pressure is None: continue
    poscar_path = os.path.join(folder, 'POSCAR')
    target_path = os.path.join(folder, 'target.json')
    if not os.path.exists(poscar_path) or not os.path.exists(target_path): continue

    with open(target_path) as f:
        dft_data = json.load(f)
    dft_bulk = dft_data.get('bulk_modulus', None)
    if dft_bulk is None: continue

    struct = Structure.from_file(poscar_path)
    tensor = predict(struct) # This uses the 'matten' library
    voigt = tensor3x3x3x3_to_voigt(tensor)
    matten_bulk = get_bulk_modulus_voigt(voigt)
    tensor_norm = np.linalg.norm(voigt)

    results_data_list.append({
        'system': system,
        'pressure': pressure,
        'dft_bulk': dft_bulk,
        'matten_raw': matten_bulk,
        'tensor_norm': tensor_norm
    })

df = pd.DataFrame(results_data_list).sort_values(['system', 'pressure'])

# --- Correction Models ---
X = df[['matten_raw']].values
X_full = df[['matten_raw', 'pressure', 'tensor_norm']]
y = df['dft_bulk'].values

models_matten = {
    'lin': LinearRegression().fit(X, y),
    'poly2': make_pipeline(PolynomialFeatures(2), LinearRegression()).fit(X, y),
    'poly3': make_pipeline(PolynomialFeatures(3), LinearRegression()).fit(X, y),
    'poly4': make_pipeline(PolynomialFeatures(4), LinearRegression()).fit(X, y),
    'iso': IsotonicRegression(out_of_bounds='clip').fit(df['matten_raw'], y),
    'exp': make_pipeline(PolynomialFeatures(degree=2), LinearRegression()).fit(np.log1p(X), y),
    'knn': GridSearchCV(KNeighborsRegressor(), {'n_neighbors': range(1, 11)}, cv=3).fit(X, y).best_estimator_,
    'gpr': GaussianProcessRegressor(kernel=RBF() + WhiteKernel(), alpha=1e-5).fit(X, y),
    'rf': GridSearchCV(RandomForestRegressor(random_state=0), {
        'n_estimators': [100, 200],
        'max_depth': [None, 20],
        'min_samples_leaf': [1, 2]
    }, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1).fit(X_full, y).best_estimator_
}

# --- Predictions ---
for name, model in models_matten.items():
    if name == 'iso':
        df[f'matten_{name}'] = model.predict(df['matten_raw'])
    elif name == 'exp':
        df[f'matten_{name}'] = model.predict(np.log1p(X))
    elif name == 'rf':
        df[f'matten_{name}'] = model.predict(X_full)
    else:
        df[f'matten_{name}'] = model.predict(X)

# --- Metrics (including raw MatTen) ---
metrics = {'Model': [], 'R2': [], 'MAE': []}
metrics['Model'].append("MatTen")
metrics['R2'].append(r2_score(df['dft_bulk'], df['matten_raw']))
metrics['MAE'].append(mean_absolute_error(df['dft_bulk'], df['matten_raw']))
for name in models_matten:
    metrics['Model'].append('MatTen+' + name)
    metrics['R2'].append(r2_score(df['dft_bulk'], df[f'matten_{name}']))
    metrics['MAE'].append(mean_absolute_error(df['dft_bulk'], df[f'matten_{name}']))

metrics_df = pd.DataFrame(metrics)
metrics_df.iloc[1:] = metrics_df.iloc[1:].sort_values('MAE').values

# --- Create Results Directory Structure ---
output_results_dir = 'results_ml_ecs'
output_figures_dir = os.path.join(output_results_dir, 'figures')
os.makedirs(output_figures_dir, exist_ok=True)

# Save metrics CSV to results directory
metrics_df.to_csv(os.path.join(output_results_dir, "bulkmod_correction_metrics.csv"), index=False)


best_model_idx = metrics_df['MAE'][1:].argmin() + 1
best_model_name = metrics_df.iloc[best_model_idx]['Model'].replace('MatTen+', '')

# --- Plot Setup ---
marker_list = ['o', 's', 'D', '^', 'v', 'P', '*', 'X', '<', '>']
system_list = sorted(df['system'].unique())
cmap = plt.get_cmap('tab10')
color_dict = {sys: cmap(i % 10) for i, sys in enumerate(system_list)}

max_p_obs = max([df[df['system']==s]['pressure'].max() for s in system_list])
extrap_p = np.linspace(0, max_p_obs*1.4, 180)


# --- MatTen Only Plot: DFT vs Raw MatTen (Interpolation & Extrapolation) ---
plt.figure(figsize=(13,8))
for i, system in enumerate(system_list):
    marker = marker_list[i % len(marker_list)]
    color = color_dict[system]
    group = df[df['system'] == system]
    p_obs = group['pressure'].values
    dft = group['dft_bulk'].values
    matten_raw_obs = group['matten_raw'].values

    # --- Interpolation region: plot lines through actual data points ---
    plt.scatter(p_obs, dft, color=color, edgecolor='k', marker=marker, s=110, label=f"{system}: DFT", zorder=10)
    plt.plot(p_obs, matten_raw_obs, '-', color=color, lw=2.7, alpha=0.93, zorder=5)

    # --- Extrapolation region: start from last observed point ---
    max_obs_p = p_obs.max()
    p_extrap = extrap_p[extrap_p > max_obs_p]
    if len(p_extrap) > 0:
        # Fit (linear or poly as justified) to observed points for trend
        deg = 1
        coeffs_raw = np.polyfit(p_obs, matten_raw_obs, deg=deg)
        matten_raw_extrap = np.polyval(coeffs_raw, p_extrap)

        # Append the last observed point to start the extrapolation line smoothly
        plt.plot(np.concatenate([[max_obs_p], p_extrap]), np.concatenate([[matten_raw_obs[-1]], matten_raw_extrap]),
                 '-', color=color, lw=2.7, alpha=0.47, zorder=4)

# Shade extrapolation region (only once)
plt.axvspan(max_p_obs, plt.gca().get_xlim()[1], color='gray', alpha=0.14, label='Extrapolation Region')
plt.axvline(max_p_obs, color='gray', ls='--', lw=2, alpha=0.4)

# Custom legend for MatTen Only plot
legend_elems_matten_only = []
for i, system in enumerate(system_list):
    marker = marker_list[i % len(marker_list)]
    color = color_dict[system]
    legend_elems_matten_only.append(Line2D([0], [0], marker=marker, color='w', label=f"{system}: DFT",
                                            markerfacecolor=color, markeredgecolor='k', markersize=13, linewidth=0))
legend_elems_matten_only.append(Line2D([0], [0],  color='k', lw=3, label='Solid: MatTen', linestyle='-'))
legend_elems_matten_only.append(Line2D([0], [0], color='gray', lw=7, alpha=0.13, label='Extrapolation Region'))

plt.xlabel('Pressure (GPa)', fontsize=20)
plt.ylabel('Bulk Modulus (GPa)', fontsize=20)
plt.title('Bulk Modulus: MatTen Predictions', fontsize=22)
plt.legend(handles=legend_elems_matten_only, ncol=2, loc='upper left', frameon=True, borderaxespad=0.8, handletextpad=1.2)
plt.grid(alpha=0.25, linestyle=':', linewidth=1.5)
plt.tight_layout()
plt.savefig(os.path.join(output_figures_dir, 'bulkmod_matten_only_projection_per_system.png'), dpi=600, bbox_inches='tight')
plt.close()


# --- MatTen and Best Corrected Model Plot: DFT vs MatTen vs Best Corrected (Interpolation & Extrapolation) ---
plt.figure(figsize=(13,8))
for i, system in enumerate(system_list):
    marker = marker_list[i % len(marker_list)]
    color = color_dict[system]
    group = df[df['system'] == system]
    p_obs = group['pressure'].values
    dft = group['dft_bulk'].values
    matten_raw_obs = group['matten_raw'].values
    tensor_norm_obs = group['tensor_norm'].values
    best_model_obs = group[f'matten_{best_model_name}'].values

    # --- Interpolation region: plot lines through actual data points ---
    plt.scatter(p_obs, dft, color=color, edgecolor='k', marker=marker, s=110, label=f"{system}: DFT", zorder=10)
    plt.plot(p_obs, matten_raw_obs, '-', color=color, lw=2.7, alpha=0.93, zorder=5)
    plt.plot(p_obs, best_model_obs, '--', color=color, lw=3.1, alpha=0.98, zorder=6)

    # --- Extrapolation region: start from last observed point ---
    max_obs_p = p_obs.max()
    p_extrap = extrap_p[extrap_p > max_obs_p]
    if len(p_extrap) > 0:
        # MatTen extrapolation: continue the observed trend (linear fit)
        deg = 1
        coeffs_raw = np.polyfit(p_obs, matten_raw_obs, deg=deg)
        matten_raw_extrap = np.polyval(coeffs_raw, p_extrap)
        plt.plot(
            np.concatenate([[max_obs_p], p_extrap]),
            np.concatenate([[matten_raw_obs[-1]], matten_raw_extrap]),
            '-', color=color, lw=2.7, alpha=0.47, zorder=4
        )

        # Best model extrapolation: continue the trend of best_model_obs (linear fit)
        coeffs_best = np.polyfit(p_obs, best_model_obs, deg=deg)
        best_model_extrap = np.polyval(coeffs_best, p_extrap)
        plt.plot(
            np.concatenate([[max_obs_p], p_extrap]),
            np.concatenate([[best_model_obs[-1]], best_model_extrap]),
            '--', color=color, lw=3.1, alpha=0.5, zorder=5
        )

# Shade extrapolation region (only once)
plt.axvspan(max_p_obs, plt.gca().get_xlim()[1], color='gray', alpha=0.14, label='Extrapolation Region')
plt.axvline(max_p_obs, color='gray', ls='--', lw=2, alpha=0.4)

# Custom legend for projection plot
legend_elems_projection = []
for i, system in enumerate(system_list):
    marker = marker_list[i % len(marker_list)]
    color = color_dict[system]
    legend_elems_projection.append(Line2D([0], [0], marker=marker, color='w', label=f"{system}: DFT",
                                             markerfacecolor=color, markeredgecolor='k', markersize=13, linewidth=0))
legend_elems_projection.append(Line2D([0], [0], color='k', lw=3, label='Solid: MatTen', linestyle='-'))
legend_elems_projection.append(Line2D([0], [0], color='k', lw=3, label=f'Dashed: MatTen+{best_model_name}', linestyle='--'))
legend_elems_projection.append(Line2D([0], [0], color='gray', lw=7, alpha=0.13, label='Extrapolation Region'))

plt.xlabel('Pressure (GPa)', fontsize=20)
plt.ylabel('Bulk Modulus (GPa)', fontsize=20)
plt.title(f'Bulk Modulus: Interpolation & Extrapolation\n(Solid: MatTen, Dashed: MatTen+{best_model_name})', fontsize=22)
plt.legend(handles=legend_elems_projection, ncol=2, loc='upper left', frameon=True, borderaxespad=0.8, handletextpad=1.2)
plt.grid(alpha=0.25, linestyle=':', linewidth=1.5)
plt.tight_layout()
plt.savefig(os.path.join(output_figures_dir, 'bulkmod_matten_ml_projection_per_system.png'), dpi=600, bbox_inches='tight')
plt.close()

# --- Residuals Plot ---
df['residual_raw'] = df['dft_bulk'] - df['matten_raw']
df['residual_best'] = df['dft_bulk'] - df[f'matten_{best_model_name}']
plt.figure(figsize=(13,7))
for i, (system, group) in enumerate(df.groupby('system')):
    color = color_dict[system]
    plt.plot(group['pressure'], group['residual_raw'], 'o-', color=color, alpha=0.45, ms=9, lw=1.5, label=f"{system}: MatTen")
    plt.plot(group['pressure'], group['residual_best'], 's--', color=color, alpha=0.95, ms=9, lw=2.2, label=f"{system}: MatTen+{best_model_name}")

plt.axhline(0, color='black', ls=':', lw=2)
plt.xlabel('Pressure (GPa)')
plt.ylabel(r'Residual: DFT $-$ Prediction (GPa)')
plt.title(f'Residuals vs Pressure (Best Model: {best_model_name})')
plt.legend(ncol=2, loc='upper left', frameon=True)
plt.grid(alpha=0.23, linestyle=':', linewidth=1.2)
plt.tight_layout()
plt.savefig(os.path.join(output_figures_dir, 'residual_comparison_matten_raw_vs_best_corrected.png'), dpi=410)
plt.close()

# --- R² and MAE Bar Plots ---
fig, ax = plt.subplots(1, 2, figsize=(13.5, 6))
bar_colors = ['tab:blue' if i==best_model_idx else 'tab:gray' for i in range(len(metrics_df))]
bars1 = ax[0].bar(metrics_df['Model'], metrics_df['R2'], color=bar_colors, width=0.8)
ax[0].set_title(r"$R^2$ Score (Bulk Modulus)", fontsize=17)
ax[0].set_ylabel("$R^2$")
ax[0].set_ylim(-2, 1.1)
ax[0].grid(axis='y', ls=':', alpha=0.6)
ax[0].tick_params(axis='x', rotation=45)

bars2 = ax[1].bar(metrics_df['Model'], metrics_df['MAE'], color=bar_colors, width=0.8)
ax[1].set_title("Mean Absolute Error (Bulk Modulus)", fontsize=17)
ax[1].set_ylabel("MAE (GPa)")
ax[1].grid(axis='y', ls=':', alpha=0.6)
ax[1].tick_params(axis='x', rotation=45)

for i, bar in enumerate(bars1):
    if i == best_model_idx:
        bar.set_edgecolor('k')
        bar.set_linewidth(2.2)
for i, bar in enumerate(bars2):
    if i == best_model_idx:
        bar.set_edgecolor('k')
        bar.set_linewidth(2.2)

plt.suptitle("Correction Model Performance: Bulk Modulus", fontsize=19)
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(os.path.join(output_figures_dir, "R2_bulkmod_correction_model_comparison.png"), dpi=400)
plt.close()

# --- Interpolation Only Plot (Original for reference, kept as is) ---
plt.figure(figsize=(13,8))
for i, (system, group) in enumerate(df.groupby('system')):
    color = color_dict[system]
    marker = marker_list[i % len(marker_list)]
    p_obs = np.array(group['pressure'])
    dft = np.array(group['dft_bulk'])
    matten_raw_obs = np.array(group['matten_raw'])
    best_model_obs = np.array(group[f'matten_{best_model_name}'])

    plt.scatter(p_obs, dft, color=color, edgecolor='k', marker=marker, s=90, label=f'{system}: DFT')
    plt.plot(p_obs, matten_raw_obs, '-', color=color, lw=2.2, alpha=0.9)
    plt.plot(p_obs, best_model_obs, '--', color=color, lw=2.6, alpha=0.98)

# -- Legend fix --
legend_elems = []
for i, system in enumerate(system_list):
    marker = marker_list[i % len(marker_list)]
    color = color_dict[system]
    legend_elems.append(Line2D([0], [0], marker=marker, color='w', label=f"{system}: DFT",
                                     markerfacecolor=color, markeredgecolor='k', markersize=11, linewidth=0))
legend_elems.append(Line2D([0], [0], color='k', lw=3, label='Solid: MatTen', linestyle='-'))
legend_elems.append(Line2D([0], [0], color='k', lw=3, label=f'Dashed: MatTen+{best_model_name}', linestyle='--'))

plt.xlabel('Pressure (GPa)')
plt.ylabel('Bulk Modulus (GPa)')
plt.title(f'Bulk Modulus Interpolation (Best Model: {best_model_name})')
plt.legend(handles=legend_elems, ncol=2, loc='upper left', frameon=True,
           borderaxespad=0.8, handletextpad=1.1)
plt.grid(alpha=0.21, linestyle=':', linewidth=1.3)
plt.tight_layout()
plt.savefig(os.path.join(output_figures_dir, 'bulkmod_interpolation_per_system.png'), dpi=420)
plt.close()

print(f"\n✅ All plots and metrics complete! See '{output_results_dir}' folder for:")
print(f"  - {os.path.join(output_figures_dir, 'bulkmod_matten_only_projection_per_system.png')} (MatTen only plot)")
print(f"  - {os.path.join(output_figures_dir, 'bulkmod_matten_ml_projection_per_system.png')} (Best corrected model plot)")
print(f"  - {os.path.join(output_figures_dir, 'bulkmod_interpolation_per_system.png')} (observed only)")
print(f"  - {os.path.join(output_figures_dir, 'residual_comparison_matten_raw_vs_best_corrected.png')} (residuals)")
print(f"  - {os.path.join(output_figures_dir, 'R2_bulkmod_correction_model_comparison.png')} (R2/MAE bars)\n")
print(f"  - {os.path.join(output_results_dir, 'bulkmod_correction_metrics.csv')} (metrics CSV)")
