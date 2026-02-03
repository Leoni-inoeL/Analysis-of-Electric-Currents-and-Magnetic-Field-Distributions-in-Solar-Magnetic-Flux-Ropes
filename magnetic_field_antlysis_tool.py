from paraview.simple import *
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from paraview import vtk
import os

matplotlib.use('Agg')

view = GetActiveView()
if not view:
    view = CreateRenderView()
    Render(view)

source = GetActiveSource()
Show(source, view)
Render()


def compute_cylindrical_coordinates(points_array, center, Z_prime, X_prime, Y_prime):
    points = np.array(points_array)
    center_np = np.array(center)

    r_vec = points - center_np
    z_cyl = np.dot(r_vec, Z_prime)

    z_cyl_reshaped = z_cyl.reshape(-1, 1)
    r_proj = r_vec - z_cyl_reshaped * Z_prime

    r = np.linalg.norm(r_proj, axis=1)
    phi = np.zeros_like(r)

    for i in range(len(r)):
        if r[i] > 1e-10:
            e_plane = r_proj[i] / r[i]
            cos_phi = np.dot(e_plane, X_prime)
            sin_phi = np.dot(e_plane, Y_prime)
            phi[i] = np.arctan2(sin_phi, cos_phi)
        else:
            phi[i] = 0.0

    return r, phi, z_cyl


def decompose_vector_cylindrical(points_array, vectors_array, center, Z_prime, X_prime, Y_prime):
    points = np.array(points_array)
    vectors = np.array(vectors_array)
    center_np = np.array(center)

    r_vec = points - center_np
    B_n = np.dot(vectors, Z_prime)

    B_r = np.zeros(len(points))
    B_phi = np.zeros(len(points))

    for i in range(len(points)):
        z_val = np.dot(r_vec[i], Z_prime)
        r_proj = r_vec[i] - z_val * Z_prime
        r_dist = np.linalg.norm(r_proj)

        if r_dist > 1e-10:
            e_r = r_proj / r_dist
            e_phi = np.cross(Z_prime, e_r)
            B_r[i] = np.dot(vectors[i], e_r)
            B_phi[i] = np.dot(vectors[i], e_phi)
        else:
            B_r[i] = np.dot(vectors[i], X_prime)
            B_phi[i] = np.dot(vectors[i], Y_prime)

    return B_r, B_phi, B_n


def decompose_curl_cylindrical(points_array, curl_array, center, Z_prime, X_prime, Y_prime):
    points = np.array(points_array)
    curl_vectors = np.array(curl_array)
    center_np = np.array(center)

    r_vec = points - center_np
    curl_B_n = np.dot(curl_vectors, Z_prime)

    curl_B_r = np.zeros(len(points))
    curl_B_phi = np.zeros(len(points))

    for i in range(len(points)):
        z_val = np.dot(r_vec[i], Z_prime)
        r_proj = r_vec[i] - z_val * Z_prime
        r_dist = np.linalg.norm(r_proj)

        if r_dist > 1e-10:
            e_r = r_proj / r_dist
            e_phi = np.cross(Z_prime, e_r)
            curl_B_r[i] = np.dot(curl_vectors[i], e_r)
            curl_B_phi[i] = np.dot(curl_vectors[i], e_phi)
        else:
            curl_B_r[i] = np.dot(curl_vectors[i], X_prime)
            curl_B_phi[i] = np.dot(curl_vectors[i], Y_prime)

    return curl_B_r, curl_B_phi, curl_B_n


def create_sphere_heatmap_profiles(df, center, radius, base_path):
    points_local = df[['x_prime', 'y_prime']].values
    distances = np.sqrt(points_local[:, 0] ** 2 + points_local[:, 1] ** 2)

    inside_mask = distances <= radius
    df_sphere = df[inside_mask].copy()
    df_sphere['r'] = distances[inside_mask]

    if len(df_sphere) < 3:
        return None

    components = ['B_n', 'B_r', 'B_phi']
    components = [c for c in components if c in df.columns]

    fig = plt.figure(figsize=(20, 12))

    profiles_data = {}

    for idx, component in enumerate(components):
        ax1 = plt.subplot(3, 4, idx * 4 + 1)

        sc1 = ax1.scatter(df_sphere['x_prime'], df_sphere['y_prime'],
                          c=df_sphere[component], cmap='coolwarm',
                          s=100, alpha=0.7, edgecolors='black', linewidth=0.5)

        circle = plt.Circle((0, 0), radius, color='black', fill=False,
                            linestyle='--', linewidth=2)
        ax1.add_patch(circle)

        ax1.set_xlabel("X' [m]", fontsize=10)
        ax1.set_ylabel("Y' [m]", fontsize=10)
        ax1.set_title(f"{component} Distribution", fontsize=11, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle=':')
        ax1.set_aspect('equal')
        ax1.set_xlim(-radius * 1.1, radius * 1.1)
        ax1.set_ylim(-radius * 1.1, radius * 1.1)
        cbar1 = plt.colorbar(sc1, ax=ax1)
        cbar1.set_label(f'{component} [T]', fontsize=9)

        n_profile_bins = min(15, max(5, len(df_sphere) // 10))
        x_bins = np.linspace(-radius, radius, n_profile_bins + 1)
        y_bins = np.linspace(-radius, radius, n_profile_bins + 1)

        x_centers = (x_bins[1:] + x_bins[:-1]) / 2
        y_centers = (y_bins[1:] + y_bins[:-1]) / 2

        x_profile = []
        x_std = []
        x_valid = []

        for i in range(len(x_bins) - 1):
            circle_radius_at_x = np.sqrt(max(0, radius ** 2 - x_centers[i] ** 2))

            if circle_radius_at_x > 0.01:
                mask = (df_sphere['x_prime'] >= x_bins[i]) & (df_sphere['x_prime'] < x_bins[i + 1])
                mask = mask & (df_sphere['y_prime'].abs() <= circle_radius_at_x)

                if mask.sum() > 0:
                    values = df_sphere.loc[mask, component]
                    x_profile.append(values.mean())
                    x_std.append(values.std(ddof=1) if len(values) > 1 else 0)
                    x_valid.append(True)
                else:
                    x_profile.append(np.nan)
                    x_std.append(0)
                    x_valid.append(False)
            else:
                x_profile.append(np.nan)
                x_std.append(0)
                x_valid.append(False)

        ax2 = plt.subplot(3, 4, idx * 4 + 2)
        valid_x = np.array(x_valid)

        if valid_x.any():
            x_centers_valid = x_centers[valid_x]
            x_profile_valid = np.array(x_profile)[valid_x]
            x_std_valid = np.array(x_std)[valid_x]

            ax2.errorbar(x_centers_valid, x_profile_valid, yerr=x_std_valid,
                         fmt='b-', linewidth=2.5, markersize=0, capsize=3,
                         elinewidth=1.5, capthick=1.5, label='Mean ± Std')
            ax2.fill_between(x_centers_valid,
                             x_profile_valid - x_std_valid,
                             x_profile_valid + x_std_valid,
                             alpha=0.2, color='blue', label='Std deviation')

        ax2.set_xlabel("X' coordinate [m]", fontsize=10)
        ax2.set_ylabel(f"{component} [T]", fontsize=10)
        ax2.set_title(f"{component} along X' axis", fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle=':')
        ax2.legend(fontsize=9, loc='best')
        ax2.set_xlim(-radius * 1.05, radius * 1.05)

        y_profile = []
        y_std = []
        y_valid = []

        for i in range(len(y_bins) - 1):
            circle_radius_at_y = np.sqrt(max(0, radius ** 2 - y_centers[i] ** 2))

            if circle_radius_at_y > 0.01:
                mask = (df_sphere['y_prime'] >= y_bins[i]) & (df_sphere['y_prime'] < y_bins[i + 1])
                mask = mask & (df_sphere['x_prime'].abs() <= circle_radius_at_y)

                if mask.sum() > 0:
                    values = df_sphere.loc[mask, component]
                    y_profile.append(values.mean())
                    y_std.append(values.std(ddof=1) if len(values) > 1 else 0)
                    y_valid.append(True)
                else:
                    y_profile.append(np.nan)
                    y_std.append(0)
                    y_valid.append(False)
            else:
                y_profile.append(np.nan)
                y_std.append(0)
                y_valid.append(False)

        ax3 = plt.subplot(3, 4, idx * 4 + 3)
        valid_y = np.array(y_valid)

        if valid_y.any():
            y_centers_valid = y_centers[valid_y]
            y_profile_valid = np.array(y_profile)[valid_y]
            y_std_valid = np.array(y_std)[valid_y]

            ax3.errorbar(y_centers_valid, y_profile_valid, yerr=y_std_valid,
                         fmt='r-', linewidth=2.5, markersize=0, capsize=3,
                         elinewidth=1.5, capthick=1.5, label='Mean ± Std')
            ax3.fill_between(y_centers_valid,
                             y_profile_valid - y_std_valid,
                             y_profile_valid + y_std_valid,
                             alpha=0.2, color='red', label='Std deviation')

        ax3.set_xlabel("Y' coordinate [m]", fontsize=10)
        ax3.set_ylabel(f"{component} [T]", fontsize=10)
        ax3.set_title(f"{component} along Y' axis", fontsize=11, fontweight='bold')
        ax3.grid(True, alpha=0.3, linestyle=':')
        ax3.legend(fontsize=9, loc='best')
        ax3.set_xlim(-radius * 1.05, radius * 1.05)

        ax4 = plt.subplot(3, 4, idx * 4 + 4)

        radial_bins = np.linspace(0, radius, min(15, max(5, len(df_sphere) // 5)))
        radial_centers = (radial_bins[1:] + radial_bins[:-1]) / 2
        radial_profile = []
        radial_std = []

        for i in range(len(radial_bins) - 1):
            mask = (df_sphere['r'] >= radial_bins[i]) & (df_sphere['r'] < radial_bins[i + 1])
            if mask.sum() > 0:
                values = df_sphere.loc[mask, component]
                radial_profile.append(values.mean())
                radial_std.append(values.std(ddof=1) if len(values) > 1 else 0)
            else:
                radial_profile.append(np.nan)
                radial_std.append(0)

        valid_radial = ~np.isnan(radial_profile)
        if valid_radial.any():
            radial_profile_valid = np.array(radial_profile)[valid_radial]
            radial_std_valid = np.array(radial_std)[valid_radial]
            radial_centers_valid = radial_centers[valid_radial]

            ax4.errorbar(radial_centers_valid, radial_profile_valid, yerr=radial_std_valid,
                         fmt='g-', linewidth=2.5, markersize=0, capsize=3,
                         elinewidth=1.5, capthick=1.5, label='Mean ± Std')
            ax4.fill_between(radial_centers_valid,
                             radial_profile_valid - radial_std_valid,
                             radial_profile_valid + radial_std_valid,
                             alpha=0.2, color='green', label='Std deviation')

        ax4.set_xlabel('Radial distance [m]', fontsize=10)
        ax4.set_ylabel(f"{component} [T]", fontsize=10)
        ax4.set_title(f"{component} radial profile", fontsize=11, fontweight='bold')
        ax4.grid(True, alpha=0.3, linestyle=':')
        ax4.legend(fontsize=9, loc='best')

        profiles_data[component] = {
            'x_centers': x_centers.tolist(),
            'x_profile': x_profile,
            'x_std': x_std,
            'x_valid': x_valid,
            'y_centers': y_centers.tolist(),
            'y_profile': y_profile,
            'y_std': y_std,
            'y_valid': y_valid,
            'radial_centers': radial_centers.tolist(),
            'radial_profile': radial_profile,
            'radial_std': radial_std
        }

    plt.suptitle(f'Magnetic Field Analysis Inside Sphere\nRadius: {radius} m | Points inside: {len(df_sphere)}',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    output_path = os.path.join(base_path, 'sphere_analysis.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()

    # Create separate DataFrames for each profile type
    profiles_x_df = pd.DataFrame({'x_coordinate': x_centers})
    profiles_y_df = pd.DataFrame({'y_coordinate': y_centers})
    profiles_radial_df = pd.DataFrame({'radial_distance': radial_centers})

    for component in components:
        if component in profiles_data:
            profiles_x_df[f'{component}_x_profile'] = profiles_data[component]['x_profile']
            profiles_x_df[f'{component}_x_std'] = profiles_data[component]['x_std']

            profiles_y_df[f'{component}_y_profile'] = profiles_data[component]['y_profile']
            profiles_y_df[f'{component}_y_std'] = profiles_data[component]['y_std']

            profiles_radial_df[f'{component}_radial_profile'] = profiles_data[component]['radial_profile']
            profiles_radial_df[f'{component}_radial_std'] = profiles_data[component]['radial_std']

    csv_output_x = os.path.join(base_path, 'sphere_profiles_x.csv')
    csv_output_y = os.path.join(base_path, 'sphere_profiles_y.csv')
    csv_output_radial = os.path.join(base_path, 'sphere_profiles_radial.csv')

    profiles_x_df.to_csv(csv_output_x, index=False)
    profiles_y_df.to_csv(csv_output_y, index=False)
    profiles_radial_df.to_csv(csv_output_radial, index=False)

    return df_sphere, profiles_data


base_path = 'C:/Users/user/Desktop/SummerPractice/'
os.makedirs(base_path, exist_ok=True)

center = [1.5, 0.0, 0.2]
radius = 0.2

clip = Clip(Input=source)
clip.ClipType = 'Sphere'
clip.ClipType.Center = center
clip.ClipType.Radius = radius
clip.Invert = 1
Show(clip, view)
Render()

integrate = IntegrateVariables(Input=clip)
UpdatePipeline()
Show(integrate, view)
Render()
data = servermanager.Fetch(integrate)
cell_data = data.GetCellData()

volume = 0
b1_total, b2_total, b3_total = 0, 0, 0

values = {'Volume': 0, 'b1': 0, 'b2': 0, 'b3': 0}

for i in range(cell_data.GetNumberOfArrays()):
    arr = cell_data.GetArray(i)
    name = arr.GetName()

    if arr.GetNumberOfComponents() == 1 and arr.GetNumberOfTuples() > 0:
        if name in values:
            values[name] = arr.GetValue(0)

volume = values['Volume']
b1_total = values['b1']
b2_total = values['b2']
b3_total = values['b3']

b1_mean = b1_total / volume
b2_mean = b2_total / volume
b3_mean = b3_total / volume
B_mean = np.array([b1_mean, b2_mean, b3_mean])
B_norm = B_mean / np.linalg.norm(B_mean)

arbitrary = np.array([0.0, 0.0, 1.0]) if abs(B_norm[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
X_prime = np.cross(arbitrary, B_norm)
X_prime = X_prime / np.linalg.norm(X_prime)
Y_prime = np.cross(B_norm, X_prime)

calc_mag = Calculator(Input=source)
calc_mag.AttributeType = 'Cell Data'
calc_mag.ResultArrayName = 'B_magnitude'
calc_mag.Function = 'sqrt(b1*b1 + b2*b2 + b3*b3)'
Show(calc_mag, view)
Render()

calc_vector = Calculator(Input=calc_mag)
calc_vector.AttributeType = 'Cell Data'
calc_vector.ResultArrayName = 'B_vector'
calc_vector.Function = 'b1*iHat + b2*jHat + b3*kHat'
Show(calc_vector, view)
Render()

cell_to_point_B = CellDatatoPointData(Input=calc_vector)
cell_to_point_B.ProcessAllArrays = 1
UpdatePipeline()

gradient_exact = Gradient(Input=cell_to_point_B)
gradient_exact.ScalarArray = ['POINTS', 'B_vector']
gradient_exact.ComputeGradient = 0
gradient_exact.ComputeDivergence = 0
gradient_exact.ComputeVorticity = 1
gradient_exact.ComputeQCriterion = 0
gradient_exact.VorticityArrayName = 'curl_B_exact'
UpdatePipeline()

slice_exact_curl = Slice(Input=gradient_exact)
slice_exact_curl.SliceType = 'Plane'
slice_exact_curl.SliceType.Origin = center
slice_exact_curl.SliceType.Normal = B_norm.tolist()
UpdatePipeline()

exact_curl_output = base_path + 'slice_data_curl_exact.csv'
SaveData(exact_curl_output, slice_exact_curl,
         PointDataArrays=['b1', 'b2', 'b3', 'B_vector', 'curl_B_exact', 'B_magnitude'])

cell_to_point_slice = CellDatatoPointData(Input=gradient_exact)
UpdatePipeline()
slice1 = Slice(Input=cell_to_point_slice)
slice1.SliceType = 'Plane'
slice1.SliceType.Origin = center
slice1.SliceType.Normal = B_norm.tolist()
Show(slice1, view)
Render()
UpdatePipeline()

arrow_length_Z = radius * 8.0
arrow_length_XY = radius * 4.0

line_X = Line()
line_X.Point1 = center
line_X.Point2 = [center[0] + X_prime[0] * arrow_length_XY,
                 center[1] + X_prime[1] * arrow_length_XY,
                 center[2] + X_prime[2] * arrow_length_XY]
Show(line_X, view)
Render()
axis_X_display = GetDisplayProperties(line_X, view)
axis_X_display.DiffuseColor = [1.0, 0.0, 0.0]  # red
axis_X_display.LineWidth = 6.0
axis_X_display.AmbientColor = [1.0, 0.0, 0.0]

line_Y = Line()
line_Y.Point1 = center
line_Y.Point2 = [center[0] + Y_prime[0] * arrow_length_XY,
                 center[1] + Y_prime[1] * arrow_length_XY,
                 center[2] + Y_prime[2] * arrow_length_XY]
Show(line_Y, view)
Render()
axis_Y_display = GetDisplayProperties(line_Y, view)
axis_Y_display.DiffuseColor = [0.0, 1.0, 0.0]  # green
axis_Y_display.LineWidth = 6.0
axis_Y_display.AmbientColor = [0.0, 1.0, 0.0]

line_Z = Line()
line_Z.Point1 = center
line_Z.Point2 = [center[0] + B_norm[0] * arrow_length_Z,
                 center[1] + B_norm[1] * arrow_length_Z,
                 center[2] + B_norm[2] * arrow_length_Z]
Show(line_Z, view)
Render()
axis_Z_display = GetDisplayProperties(line_Z, view)
axis_Z_display.DiffuseColor = [0.0, 0.0, 1.0]  # blue
axis_Z_display.LineWidth = 8.0
axis_Z_display.AmbientColor = [0.5, 0.5, 1.0]

center_sphere = Sphere()
center_sphere.Center = center
center_sphere.Radius = radius * 0.15
center_sphere.ThetaResolution = 16
center_sphere.PhiResolution = 16
Show(center_sphere, view)
center_display = GetDisplayProperties(center_sphere, view)
center_display.DiffuseColor = [1.0, 1.0, 0.0]  # yellow
center_display.AmbientColor = [1.0, 1.0, 0.0]

sphere_seed = Sphere()
sphere_seed.Center = center
sphere_seed.Radius = radius * 0.8
sphere_seed.ThetaResolution = 16
sphere_seed.PhiResolution = 16

stream = StreamTracerWithCustomSource(Input=calc_vector, SeedSource=sphere_seed)
stream.Vectors = ['CELLS', 'B_vector']
stream.MaximumStreamlineLength = radius * 3
Show(stream, view)
stream_display = GetDisplayProperties(stream, view)
stream_display.LineWidth = 2.5
ColorBy(stream_display, ('CELLS', 'B_magnitude'))

slice_display = GetDisplayProperties(slice1, view)
slice_display.Representation = 'Surface'
ColorBy(slice_display, ('POINTS', 'B_magnitude'))

b_lut = GetColorTransferFunction('B_magnitude')
b_lut.ApplyPreset('Cool to Warm (Extended)', True)

output_path = base_path + 'slice_data_with_curl_and_B.csv'
cell_to_point_for_slice = CellDatatoPointData(Input=slice1)
UpdatePipeline()
SaveData(output_path, cell_to_point_for_slice,
         PointDataArrays=['b1', 'b2', 'b3', 'B_vector', 'curl_B_exact', 'B_magnitude'])

HideAll()
Show(source, view)
source_display = GetDisplayProperties(source, view)
source_display.Opacity = 0.0
source_display.Representation = 'Surface'

Show(slice1, view)
Show(line_X, view)
Show(line_Y, view)
Show(line_Z, view)
Show(center_sphere, view)
Show(stream, view)

view.CameraPosition = [center[0] + radius * 8,
                       center[1] + radius * 8,
                       center[2] + radius * 8]
view.CameraFocalPoint = center
view.CameraViewUp = [0, 0, 1]
view.CameraParallelScale = radius * 5.0
view.OrientationAxesVisibility = 1
Render()

SaveScreenshot(base_path + 'magnetic_analysis_local.png', view, ImageResolution=[1600, 900])

df = pd.read_csv(output_path)

curl_components = [f'curl_B_exact:{i}' for i in range(3)]
points_cols = ['Points:0', 'Points:1', 'Points:2']

missing_curl = [col for col in curl_components if col not in df.columns]
if missing_curl:
    curl_components = []
    for i in range(3):
        for pattern in [f'curl_B_exact:{i}', f'curl_B_exact_{i}', f'curl_B_exact.{i}']:
            if pattern in df.columns:
                curl_components.append(pattern)
                break

if len(curl_components) == 3 and all(col in df.columns for col in points_cols):
    points = df[points_cols].values
    B_vectors = df[['b1', 'b2', 'b3']].values
    curl_vectors = df[curl_components].values

    r, phi, z_cyl = compute_cylindrical_coordinates(points, center, B_norm, X_prime, Y_prime)
    B_r, B_phi, B_n = decompose_vector_cylindrical(points, B_vectors, center, B_norm, X_prime, Y_prime)

    curl_B_r_exact, curl_B_phi_exact, curl_B_n_exact = decompose_curl_cylindrical(
        points, curl_vectors, center, B_norm, X_prime, Y_prime
    )

    df['r'] = r
    df['phi_rad'] = phi
    df['phi_deg'] = np.degrees(phi)
    df['z_cyl'] = z_cyl
    df['B_r'] = B_r
    df['B_phi'] = B_phi
    df['B_n'] = B_n
    df['curl_B_r_exact'] = curl_B_r_exact
    df['curl_B_phi_exact'] = curl_B_phi_exact
    df['curl_B_n_exact'] = curl_B_n_exact
    df['x_prime'] = r * np.cos(phi)
    df['y_prime'] = r * np.sin(phi)

    cylindrical_output = base_path + 'slice_data_cylindrical.csv'
    df.to_csv(cylindrical_output, index=False)
else:
    exit()

points_local = df[['x_prime', 'y_prime']].values
distances = np.sqrt(points_local[:, 0]**2 + points_local[:, 1]**2)
inside_mask = distances <= radius
df_sphere = df[inside_mask].copy()

sphere_cylindrical_output = base_path + 'slice_data_cylindrical_sphere.csv'
df_sphere.to_csv(sphere_cylindrical_output, index=False)

fig_task4, axes_task4 = plt.subplots(1, 3, figsize=(18, 5))

# B_n

sc1 = axes_task4[0].scatter(df['x_prime'], df['y_prime'], c=df['B_n'],
                            cmap='coolwarm', s=8, alpha=0.8)
axes_task4[0].set_xlabel('X\' (m)')
axes_task4[0].set_ylabel('Y\' (m)')
axes_task4[0].set_title('B_n distribution')
axes_task4[0].set_aspect('equal')
axes_task4[0].grid(True, alpha=0.3)
plt.colorbar(sc1, ax=axes_task4[0]).set_label('B_n [T]', rotation=270, labelpad=15)

# B_r
sc2 = axes_task4[1].scatter(df['x_prime'], df['y_prime'], c=df['B_r'],
                            cmap='coolwarm', s=8, alpha=0.8)
axes_task4[1].set_xlabel('X\' (m)')
axes_task4[1].set_ylabel('Y\' (m)')
axes_task4[1].set_title('B_r distribution')
axes_task4[1].set_aspect('equal')
axes_task4[1].grid(True, alpha=0.3)
plt.colorbar(sc2, ax=axes_task4[1]).set_label('B_r [T]', rotation=270, labelpad=15)

# B_phi
sc3 = axes_task4[2].scatter(df['x_prime'], df['y_prime'], c=df['B_phi'],
                            cmap='coolwarm', s=8, alpha=0.8)
axes_task4[2].set_xlabel('X\' (m)')
axes_task4[2].set_ylabel('Y\' (m)')
axes_task4[2].set_title('B_φ distribution')
axes_task4[2].set_aspect('equal')
axes_task4[2].grid(True, alpha=0.3)
plt.colorbar(sc3, ax=axes_task4[2]).set_label('B_φ [T]', rotation=270, labelpad=15)

plt.suptitle(f'Magnetic Field Components (O={center}, R={radius}m)', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(base_path + 'task4_B_components.png', dpi=150, bbox_inches='tight')
plt.close()

fig_task5, axes_task5 = plt.subplots(1, 3, figsize=(18, 5))

# curl_B_n
sc4 = axes_task5[0].scatter(df['x_prime'], df['y_prime'], c=df['curl_B_n_exact'],
                            cmap='coolwarm', s=8, alpha=0.8)
axes_task5[0].set_xlabel('X\' (m)')
axes_task5[0].set_ylabel('Y\' (m)')
axes_task5[0].set_title('∇×B_n')
axes_task5[0].set_aspect('equal')
axes_task5[0].grid(True, alpha=0.3)
plt.colorbar(sc4, ax=axes_task5[0]).set_label('∇×B_n [A/m²]', rotation=270, labelpad=15)

# curl_B_r
sc5 = axes_task5[1].scatter(df['x_prime'], df['y_prime'], c=df['curl_B_r_exact'],
                            cmap='coolwarm', s=8, alpha=0.8)
axes_task5[1].set_xlabel('X\' (m)')
axes_task5[1].set_ylabel('Y\' (m)')
axes_task5[1].set_title('∇×B_r')
axes_task5[1].set_aspect('equal')
axes_task5[1].grid(True, alpha=0.3)
plt.colorbar(sc5, ax=axes_task5[1]).set_label('∇×B_r [A/m²]', rotation=270, labelpad=15)

# curl_B_phi
sc6 = axes_task5[2].scatter(df['x_prime'], df['y_prime'], c=df['curl_B_phi_exact'],
                            cmap='coolwarm', s=8, alpha=0.8)
axes_task5[2].set_xlabel('X\' (m)')
axes_task5[2].set_ylabel('Y\' (m)')
axes_task5[2].set_title('∇×B_φ')
axes_task5[2].set_aspect('equal')
axes_task5[2].grid(True, alpha=0.3)
plt.colorbar(sc6, ax=axes_task5[2]).set_label('∇×B_φ [A/m²]', rotation=270, labelpad=15)

plt.suptitle(f'∇×B Components\nO={center}, R={radius}m',
             fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(base_path + 'task5_curl_B.png', dpi=150, bbox_inches='tight')
plt.close()


df_sphere, profiles_data = create_sphere_heatmap_profiles(df, center, radius, base_path)
