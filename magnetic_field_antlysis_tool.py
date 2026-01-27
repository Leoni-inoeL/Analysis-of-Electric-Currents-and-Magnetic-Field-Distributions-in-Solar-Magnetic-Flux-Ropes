from paraview.simple import *
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from paraview import vtk
import os

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


base_path = 'C:/Users/user/Desktop/SummerPractice/'

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

for i in range(cell_data.GetNumberOfArrays()):
    arr = cell_data.GetArray(i)
    name = arr.GetName()

    if arr.GetNumberOfComponents() == 1 and arr.GetNumberOfTuples() > 0:
        value = arr.GetValue(0)

        if name == 'Volume':
            volume = value
        elif name == 'b1':
            b1_total = value
        elif name == 'b2':
            b2_total = value
        elif name == 'b3':
            b3_total = value

if volume > 0:
    b1_mean = b1_total / volume
    b2_mean = b2_total / volume
    b3_mean = b3_total / volume

    B_mean = np.array([b1_mean, b2_mean, b3_mean])
    B_norm = B_mean / np.linalg.norm(B_mean)

    """print(f"\nMean field results:")
    print(f"  B_mean: [{b1_mean:.6e}, {b2_mean:.6e}, {b3_mean:.6e}]")
    print(f"  |B_mean|: {np.linalg.norm(B_mean):.6e}")
    print(f"  B_norm (Z' axis): {B_norm}")"""

else:
    print("Error: sphere volume = 0")
    B_mean = np.array([1.0, 0.0, 0.0])
    B_norm = B_mean

if abs(B_norm[2]) < 0.9:
    arbitrary = np.array([0.0, 0.0, 1.0])
else:
    arbitrary = np.array([1.0, 0.0, 0.0])

X_prime = np.cross(arbitrary, B_norm)
X_prime = X_prime / np.linalg.norm(X_prime)
Y_prime = np.cross(B_norm, X_prime)

"""print(f"Local coordinate axes:")
print(f"  X' axis: {X_prime}")
print(f"  Y' axis: {Y_prime}")
print(f"  Z' axis: {B_norm}")"""

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

# Using existing current density j = (j1, j2, j3) as ∇ × B
# ∇ × B = μ₀ * j => curl_B ~= j

sphere_seed = Sphere()
sphere_seed.Center = center
sphere_seed.Radius = radius * 0.8
sphere_seed.ThetaResolution = 16
sphere_seed.PhiResolution = 16

stream = StreamTracerWithCustomSource(
    Input=calc_vector,
    SeedSource=sphere_seed
)
stream.Vectors = ['CELLS', 'B_vector']
stream.MaximumStreamlineLength = radius * 3
Show(stream, view)
Render()

slice1 = Slice(Input=calc_mag)
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

line_Y = Line()
line_Y.Point1 = center
line_Y.Point2 = [center[0] + Y_prime[0] * arrow_length_XY,
                 center[1] + Y_prime[1] * arrow_length_XY,
                 center[2] + Y_prime[2] * arrow_length_XY]
Show(line_Y, view)
Render()

line_Z = Line()
line_Z.Point1 = center
line_Z.Point2 = [center[0] + B_norm[0] * arrow_length_Z,
                 center[1] + B_norm[1] * arrow_length_Z,
                 center[2] + B_norm[2] * arrow_length_Z]
Show(line_Z, view)
Render()

output_path = base_path + 'slice_data_local.csv'

SaveData(output_path, slice1, PointDataArrays=['b1', 'b2', 'b3', 'j1', 'j2', 'j3', 'B_magnitude'])

HideAll()
Show(source, view)
source_display = GetDisplayProperties(source, view)
source_display.Opacity = 0.05
source_display.Representation = 'Surface'

Show(line_X, view)
axis_X_display = GetDisplayProperties(line_X, view)
axis_X_display.DiffuseColor = [1.0, 0.0, 0.0]
axis_X_display.LineWidth = 6.0
axis_X_display.AmbientColor = [1.0, 0.0, 0.0]

Show(line_Y, view)
axis_Y_display = GetDisplayProperties(line_Y, view)
axis_Y_display.DiffuseColor = [0.0, 1.0, 0.0]
axis_Y_display.LineWidth = 6.0
axis_Y_display.AmbientColor = [0.0, 1.0, 0.0]

Show(line_Z, view)
axis_Z_display = GetDisplayProperties(line_Z, view)
axis_Z_display.DiffuseColor = [0.0, 0.0, 1.0]
axis_Z_display.LineWidth = 8.0
axis_Z_display.AmbientColor = [0.5, 0.5, 1.0]

center_sphere = Sphere()
center_sphere.Center = center
center_sphere.Radius = radius * 0.15
center_sphere.ThetaResolution = 16
center_sphere.PhiResolution = 16
Show(center_sphere, view)
center_display = GetDisplayProperties(center_sphere, view)
center_display.DiffuseColor = [1.0, 1.0, 0.0]
center_display.AmbientColor = [1.0, 1.0, 0.0]

Show(stream, view)
stream_display = GetDisplayProperties(stream, view)
stream_display.LineWidth = 2.5
ColorBy(stream_display, ('CELLS', 'B_magnitude'))

Show(slice1, view)
slice_display = GetDisplayProperties(slice1, view)
slice_display.Representation = 'Surface'
ColorBy(slice_display, ('CELLS', 'B_magnitude'))

b_lut = GetColorTransferFunction('B_magnitude')
b_lut.ApplyPreset('Cool to Warm (Extended)', True)

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

has_coords = False
points_cols = None

for col in df.columns:
    if 'Points' in col or 'points' in col.lower():
        has_coords = True
        coord_cols = [c for c in df.columns if 'Points' in c or 'points' in c.lower()]
        if len(coord_cols) >= 3:
            points_cols = coord_cols[:3]
            print(f"\nFound coordinate columns: {points_cols}")
            break

if not has_coords:

    cell_to_point = CellDatatoPointData(Input=slice1)
    UpdatePipeline()

    output_path2 = base_path + 'slice_data_local_v2.csv'
    SaveData(output_path2, cell_to_point,
             PointDataArrays=['b1', 'b2', 'b3', 'j1', 'j2', 'j3', 'B_magnitude'])

    print(f"Alternative export saved to: {output_path2}")

    if os.path.exists(output_path2):
        df = pd.read_csv(output_path2)
        print(f"\nSecond CSV loaded. Columns: {df.columns.tolist()}")

        for col in df.columns:
            if 'Points' in col or 'points' in col.lower():
                has_coords = True
                coord_cols = [c for c in df.columns if 'Points' in c or 'points' in c.lower()]
                if len(coord_cols) >= 3:
                    points_cols = coord_cols[:3]
                    print(f"\nFound coordinate columns in second file: {points_cols}")
                    break

if has_coords and points_cols and all(col in df.columns for col in ['b1', 'b2', 'b3', 'j1', 'j2', 'j3']):

    points = df[points_cols].values
    B_vectors = df[['b1', 'b2', 'b3']].values
    j_vectors = df[['j1', 'j2', 'j3']].values

    r, phi, z_cyl = compute_cylindrical_coordinates(points, center, B_norm, X_prime, Y_prime)
    B_r, B_phi, B_n = decompose_vector_cylindrical(points, B_vectors, center, B_norm, X_prime, Y_prime)

    curl_B_r, curl_B_phi, curl_B_n = decompose_curl_cylindrical(
        points, j_vectors, center, B_norm, X_prime, Y_prime
    )

    df['r'] = r
    df['phi_rad'] = phi
    df['phi_deg'] = np.degrees(phi)
    df['z_cyl'] = z_cyl
    df['B_r'] = B_r
    df['B_phi'] = B_phi
    df['B_n'] = B_n
    df['curl_B_r'] = curl_B_r
    df['curl_B_phi'] = curl_B_phi
    df['curl_B_n'] = curl_B_n
    df['x_prime'] = r * np.cos(phi)
    df['y_prime'] = r * np.sin(phi)

    cylindrical_output = base_path + 'slice_data_cylindrical.csv'
    df.to_csv(cylindrical_output, index=False)

    matplotlib.use('Agg')

    fig_task4, axes_task4 = plt.subplots(1, 3, figsize=(18, 5))

    sc1 = axes_task4[0].scatter(df['x_prime'], df['y_prime'], c=B_n,
                                cmap='coolwarm', s=8, alpha=0.8)
    axes_task4[0].set_xlabel('X\' (m)')
    axes_task4[0].set_ylabel('Y\' (m)')
    axes_task4[0].set_title('B_n distribution')
    axes_task4[0].set_aspect('equal')
    axes_task4[0].grid(True, alpha=0.3)
    cbar1 = plt.colorbar(sc1, ax=axes_task4[0])
    cbar1.set_label('B_n [T]', rotation=270, labelpad=15)

    sc2 = axes_task4[1].scatter(df['x_prime'], df['y_prime'], c=B_r,
                                cmap='coolwarm', s=8, alpha=0.8)
    axes_task4[1].set_xlabel('X\' (m)')
    axes_task4[1].set_ylabel('Y\' (m)')
    axes_task4[1].set_title('B_r distribution')
    axes_task4[1].set_aspect('equal')
    axes_task4[1].grid(True, alpha=0.3)
    cbar2 = plt.colorbar(sc2, ax=axes_task4[1])
    cbar2.set_label('B_r [T]', rotation=270, labelpad=15)

    sc3 = axes_task4[2].scatter(df['x_prime'], df['y_prime'], c=B_phi,
                                cmap='coolwarm', s=8, alpha=0.8)
    axes_task4[2].set_xlabel('X\' (m)')
    axes_task4[2].set_ylabel('Y\' (m)')
    axes_task4[2].set_title('B_φ distribution')
    axes_task4[2].set_aspect('equal')
    axes_task4[2].grid(True, alpha=0.3)
    cbar3 = plt.colorbar(sc3, ax=axes_task4[2])
    cbar3.set_label('B_φ [T]', rotation=270, labelpad=15)

    plt.suptitle(f'Magnetic Field Components (O={center}, R={radius}m)', fontsize=14, y=1.02)
    plt.tight_layout()

    task4_output = base_path + 'task4_B_components.png'
    plt.savefig(task4_output, dpi=150, bbox_inches='tight')
    plt.close(fig_task4)

    fig_task5, axes_task5 = plt.subplots(1, 3, figsize=(18, 5))

    sc4 = axes_task5[0].scatter(df['x_prime'], df['y_prime'], c=curl_B_n,
                                cmap='coolwarm', s=8, alpha=0.8)
    axes_task5[0].set_xlabel('X\' (m)')
    axes_task5[0].set_ylabel('Y\' (m)')
    axes_task5[0].set_title('curl_B_n distribution (≈ j_n)')
    axes_task5[0].set_aspect('equal')
    axes_task5[0].grid(True, alpha=0.3)
    cbar4 = plt.colorbar(sc4, ax=axes_task5[0])
    cbar4.set_label('curl_B_n [A/m²]', rotation=270, labelpad=15)

    sc5 = axes_task5[1].scatter(df['x_prime'], df['y_prime'], c=curl_B_r,
                                cmap='coolwarm', s=8, alpha=0.8)
    axes_task5[1].set_xlabel('X\' (m)')
    axes_task5[1].set_ylabel('Y\' (m)')
    axes_task5[1].set_title('curl_B_r distribution (≈ j_r)')
    axes_task5[1].set_aspect('equal')
    axes_task5[1].grid(True, alpha=0.3)
    cbar5 = plt.colorbar(sc5, ax=axes_task5[1])
    cbar5.set_label('curl_B_r [A/m²]', rotation=270, labelpad=15)

    sc6 = axes_task5[2].scatter(df['x_prime'], df['y_prime'], c=curl_B_phi,
                                cmap='coolwarm', s=8, alpha=0.8)
    axes_task5[2].set_xlabel('X\' (m)')
    axes_task5[2].set_ylabel('Y\' (m)')
    axes_task5[2].set_title('curl_B_φ distribution (≈ j_φ)')
    axes_task5[2].set_aspect('equal')
    axes_task5[2].grid(True, alpha=0.3)
    cbar6 = plt.colorbar(sc6, ax=axes_task5[2])
    cbar6.set_label('curl_B_φ [A/m²]', rotation=270, labelpad=15)

    plt.suptitle(f'Curl of B Components (approximated by current density j)\n' +
                 f'Note: ∇ × B ≈ μ₀·j, O={center}, R={radius}m',
                 fontsize=12, y=1.02)
    plt.tight_layout()

    task5_output = base_path + 'task5_curl_B_components.png'
    plt.savefig(task5_output, dpi=150, bbox_inches='tight')
    plt.close(fig_task5)
