from paraview.simple import *
import paraview
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gc
import time


source = GetActiveSource()
if source is None:
    raise RuntimeError("No active source found. Please load the file first.")

# AnnotateTime
annotate_time = AnnotateTime()
annotate_time.Format = 'Time: {time:.2f}'
RenameSource('AnnotateTime', annotate_time)
annotate_time.UpdatePipeline()

# V_km/s
calculator_V = Calculator(Input=source)
calculator_V.ResultArrayName = 'V'
calculator_V.Function = '116.45 * (iHat*v1 + jHat*v2 + kHat*v3)'
RenameSource('V_km/s', calculator_V)
calculator_V.UpdatePipeline()

# J_A/km^2
calculator_J = Calculator(Input=source)
calculator_J.ResultArrayName = 'j'
calculator_J.Function = '15.89619*(iHat*j1 + jHat*j2 + kHat*j3)'
RenameSource('J_A/km^2', calculator_J)
calculator_J.UpdatePipeline()

# B_Gs
calculator_B = Calculator(Input=source)
calculator_B.ResultArrayName = 'B'
calculator_B.Function = '1.99757357615242*(iHat*b1 + jHat*b2 + kHat*b3)'
RenameSource('B_Gs', calculator_B)
calculator_B.UpdatePipeline()

# B_RFP_TRACER
stream_tracer_B1 = StreamTracer(Input=calculator_B)
stream_tracer_B1.SeedType = 'Point Cloud'
stream_tracer_B1.SeedType.Center = [-1.5, 0.0, 0.3]
stream_tracer_B1.SeedType.Radius = 0.3
stream_tracer_B1.SeedType.NumberOfPoints = 25
stream_tracer_B1.MaximumStreamlineLength = 12.0
stream_tracer_B1.Vectors = ['POINTS', 'B']
RenameSource('B_RFP_TRACER', stream_tracer_B1)
stream_tracer_B1.UpdatePipeline()

tube_B1 = Tube(Input=stream_tracer_B1)
tube_B1.Scalars = ['POINTS', 'B']
tube_B1.Radius = 0.1
tube_B1.NumberofSides = 6
RenameSource('Tube3', tube_B1)
tube_B1.UpdatePipeline()

# B_LFP_tracer
stream_tracer_B2 = StreamTracer(Input=calculator_B)
stream_tracer_B2.SeedType = 'Point Cloud'
stream_tracer_B2.SeedType.Center = [-1.5, 0.0, 0.3]
stream_tracer_B2.SeedType.Radius = 0.3
stream_tracer_B2.SeedType.NumberOfPoints = 25
stream_tracer_B2.MaximumStreamlineLength = 24.0
stream_tracer_B2.Vectors = ['POINTS', 'B']
RenameSource('B_LFP_tracer', stream_tracer_B2)
stream_tracer_B2.UpdatePipeline()

# Jz_slice
slice_filter = Slice(Input=calculator_J)
slice_filter.SliceType = 'Plane'
slice_filter.SliceType.Origin = [0.0, 0.0, 0.01]
slice_filter.SliceType.Normal = [0.0, 0.0, 1.0]
slice_filter.Triangulatetheslice = 0
RenameSource('Jz_slice_z=0.1Mm', slice_filter)
slice_filter.UpdatePipeline()

calculator_Jz_slice = Calculator(Input=slice_filter)
calculator_Jz_slice.ResultArrayName = 'Jz'
calculator_Jz_slice.Function = 'jHat*j'
RenameSource('Jz_on_slice', calculator_Jz_slice)
calculator_Jz_slice.UpdatePipeline()

# J_RFP_tracer
stream_tracer_J1 = StreamTracer(Input=calculator_J)
stream_tracer_J1.SeedType = 'Point Cloud'
stream_tracer_J1.SeedType.Center = [1.5, 0.0, 0.3]
stream_tracer_J1.SeedType.Radius = 0.3
stream_tracer_J1.SeedType.NumberOfPoints = 50
stream_tracer_J1.MaximumStreamlineLength = 64.0
stream_tracer_J1.Vectors = ['POINTS', 'j']
stream_tracer_J1.IntegrationDirection = 'BOTH'
stream_tracer_J1.IntegratorType = 'Runge-Kutta 4-5'
RenameSource('J_RFP_tracer', stream_tracer_J1)
stream_tracer_J1.UpdatePipeline()

tube_J1 = Tube(Input=stream_tracer_J1)
tube_J1.Scalars = ['POINTS', 'j']
tube_J1.Radius = 0.1
tube_J1.NumberofSides = 6
RenameSource('Tube1', tube_J1)
tube_J1.UpdatePipeline()

# J_LFP_tracer
stream_tracer_J2 = StreamTracer(Input=calculator_J)
stream_tracer_J2.SeedType = 'Point Cloud'
stream_tracer_J2.SeedType.Center = [1.5, 0.0, 0.3]
stream_tracer_J2.SeedType.Radius = 0.3
stream_tracer_J2.SeedType.NumberOfPoints = 50
stream_tracer_J2.MaximumStreamlineLength = 256.0
stream_tracer_J2.Vectors = ['POINTS', 'j']
stream_tracer_J2.IntegrationDirection = 'BOTH'
stream_tracer_J2.IntegratorType = 'Runge-Kutta 4-5'
RenameSource('J_LFP_tracer', stream_tracer_J2)
stream_tracer_J2.UpdatePipeline()

tube_J2 = Tube(Input=stream_tracer_J2)
tube_J2.Scalars = ['POINTS', 'j']
tube_J2.Radius = 0.1
tube_J2.NumberofSides = 6
RenameSource('Tube4', tube_J2)
tube_J2.UpdatePipeline()

# J_Below_z=4Mm
stream_tracer_J3 = StreamTracer(Input=calculator_J)
stream_tracer_J3.SeedType = 'Point Cloud'
stream_tracer_J3.SeedType.Center = [0.0, 0.0, 0.4]
stream_tracer_J3.SeedType.Radius = 0.1
stream_tracer_J3.SeedType.NumberOfPoints = 50
stream_tracer_J3.MaximumStreamlineLength = 64.0
stream_tracer_J3.Vectors = ['POINTS', 'j']
stream_tracer_J3.IntegrationDirection = 'BOTH'
stream_tracer_J3.IntegratorType = 'Runge-Kutta 4-5'
RenameSource('J_Below_z=4Mm', stream_tracer_J3)
stream_tracer_J3.UpdatePipeline()

tube_J3 = Tube(Input=stream_tracer_J3)
tube_J3.Scalars = ['POINTS', 'j']
tube_J3.Radius = 0.1
tube_J3.NumberofSides = 6
RenameSource('Tube5', tube_J3)
tube_J3.UpdatePipeline()


animation_scene = GetAnimationScene()
timesteps = animation_scene.TimeKeeper.TimestepValues

output_base = 'C:/Users/user/Desktop/SummerPractice/time_series/'

os.makedirs(output_base, exist_ok=True)
os.makedirs(os.path.join(output_base, 'B_maps'), exist_ok=True)
os.makedirs(os.path.join(output_base, 'J_maps'), exist_ok=True)
os.makedirs(os.path.join(output_base, 'data'), exist_ok=True)

center_point = [0.0, 0.0, 0.0]
radius = 0.5

clip = Clip(Input=calculator_B)
clip.ClipType = 'Sphere'
clip.ClipType.Center = center_point
clip.ClipType.Radius = radius
clip.Invert = 1
UpdatePipeline()

integrate = IntegrateVariables(Input=clip)
UpdatePipeline()

data = servermanager.Fetch(integrate)
cell_data = data.GetCellData()

volume = 0
b1_total, b2_total, b3_total = 0, 0, 0

for i in range(cell_data.GetNumberOfArrays()):
    arr = cell_data.GetArray(i)
    name = arr.GetName()

    if arr.GetNumberOfComponents() == 1 and arr.GetNumberOfTuples() > 0:
        if name == 'Volume':
            volume = arr.GetValue(0)
        elif name == 'b1':
            b1_total = arr.GetValue(0)
        elif name == 'b2':
            b2_total = arr.GetValue(0)
        elif name == 'b3':
            b3_total = arr.GetValue(0)

if volume > 0:
    b1_mean = b1_total / volume
    b2_mean = b2_total / volume
    b3_mean = b3_total / volume
    B_mean = np.array([b1_mean, b2_mean, b3_mean])
    B_norm = B_mean / np.linalg.norm(B_mean)
    print(f"B_mean = [{b1_mean:.6f}, {b2_mean:.6f}, {b3_mean:.6f}]")
else:
    B_norm = np.array([0.0, 0.0, 1.0])

del clip, integrate, data, cell_data
gc.collect()


Z_prime = B_norm
arbitrary = np.array([0.0, 0.0, 1.0]) if abs(B_norm[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
X_prime = np.cross(arbitrary, Z_prime)
if np.linalg.norm(X_prime) > 0:
    X_prime = X_prime / np.linalg.norm(X_prime)
else:
    X_prime = np.array([1.0, 0.0, 0.0])
Y_prime = np.cross(Z_prime, X_prime)

print(f"X' = [{X_prime[0]:.4f}, {X_prime[1]:.4f}, {X_prime[2]:.4f}]")
print(f"Y' = [{Y_prime[0]:.4f}, {Y_prime[1]:.4f}, {Y_prime[2]:.4f}]")
print(f"Z' = [{Z_prime[0]:.4f}, {Z_prime[1]:.4f}, {Z_prime[2]:.4f}]")

slice_height = 0.01


def point_cylindrical_coords(px, py, pz, center, z_axis, x_axis, y_axis):
    r_vec = np.array([px, py, pz]) - np.array(center)
    z_cyl = np.dot(r_vec, z_axis)
    r_proj = r_vec - z_cyl * z_axis
    r = np.linalg.norm(r_proj)

    if r > 1e-10:
        e_plane = r_proj / r
        cos_phi = np.dot(e_plane, x_axis)
        sin_phi = np.dot(e_plane, y_axis)
        phi = np.arctan2(sin_phi, cos_phi)
    else:
        phi = 0.0

    x_prime = r * np.cos(phi)
    y_prime = r * np.sin(phi)

    return r, phi, x_prime, y_prime


def point_vector_decomposition(px, py, pz, vx, vy, vz, center, z_axis, x_axis, y_axis):
    r_vec = np.array([px, py, pz]) - np.array(center)
    vector = np.array([vx, vy, vz])

    vz_comp = np.dot(vector, z_axis)

    z_val = np.dot(r_vec, z_axis)
    r_proj = r_vec - z_val * z_axis
    r_dist = np.linalg.norm(r_proj)

    if r_dist > 1e-10:
        e_r = r_proj / r_dist
        e_phi = np.cross(z_axis, e_r)
        vr_comp = np.dot(vector, e_r)
        vphi_comp = np.dot(vector, e_phi)
    else:
        vr_comp = np.dot(vector, x_axis)
        vphi_comp = np.dot(vector, y_axis)

    return vr_comp, vphi_comp, vz_comp


def process_slice_directly(data_file, output_prefix, t_idx_, field_type='B'):
    line_count = 0
    chunk_size = 10000

    if not os.path.exists(data_file):
        return

    with open(data_file, 'r') as f:
        header_line = f.readline().strip()
        header = [col.strip('"') for col in header_line.split(',')]
        px_idx = header.index('Points:0')
        py_idx = header.index('Points:1')
        pz_idx = header.index('Points:2')

        if field_type == 'B':
            vx_idx = header.index('B:0')
            vy_idx = header.index('B:1')
            vz_idx = header.index('B:2')

        else:  # J
            vx_idx = header.index('j:0')
            vy_idx = header.index('j:1')
            vz_idx = header.index('j:2')

        with open(f'{output_prefix}_r_t{t_idx_:03d}.txt', 'w') as f_r, \
                open(f'{output_prefix}_phi_t{t_idx_:03d}.txt', 'w') as f_phi, \
                open(f'{output_prefix}_vr_t{t_idx_:03d}.txt', 'w') as f_vr, \
                open(f'{output_prefix}_vphi_t{t_idx_:03d}.txt', 'w') as f_vphi, \
                open(f'{output_prefix}_vz_t{t_idx_:03d}.txt', 'w') as f_vz:

            f_r.write('x_prime,y_prime,r\n')
            f_phi.write('x_prime,y_prime,phi_rad,phi_deg\n')
            f_vr.write('x_prime,y_prime,vr\n')
            f_vphi.write('x_prime,y_prime,vphi\n')
            f_vz.write('x_prime,y_prime,vz\n')

            for line in f:
                line = line.strip()
                if not line:
                    continue

                values = [val.strip('"') for val in line.split(',')]

                if len(values) <= max(px_idx, py_idx, pz_idx, vx_idx, vy_idx, vz_idx):
                    continue

                px = float(values[px_idx])
                py = float(values[py_idx])
                pz = float(values[pz_idx])

                vx = float(values[vx_idx])
                vy = float(values[vy_idx])
                vz = float(values[vz_idx])

                r, phi, x_prime, y_prime = point_cylindrical_coords(
                    px, py, pz, center_point, Z_prime, X_prime, Y_prime
                )

                vr, vphi, vz_comp = point_vector_decomposition(
                    px, py, pz, vx, vy, vz, center_point, Z_prime, X_prime, Y_prime
                )

                f_r.write(f'{x_prime:.6f},{y_prime:.6f},{r:.6f}\n')
                f_phi.write(f'{x_prime:.6f},{y_prime:.6f},{phi:.6f},{np.degrees(phi):.6f}\n')
                f_vr.write(f'{x_prime:.6f},{y_prime:.6f},{vr:.6f}\n')
                f_vphi.write(f'{x_prime:.6f},{y_prime:.6f},{vphi:.6f}\n')
                f_vz.write(f'{x_prime:.6f},{y_prime:.6f},{vz_comp:.6f}\n')

    gc.collect()


total_start = time.time()

for t_idx, time_val in enumerate(timesteps):

    animation_scene.TimeKeeper.Time = time_val
    Render()

    slice_B = Slice(Input=calculator_B)
    slice_B.SliceType = 'Plane'
    slice_B.SliceType.Origin = [center_point[0], center_point[1], slice_height]
    slice_B.SliceType.Normal = Z_prime.tolist()
    slice_B.Triangulatetheslice = 0
    UpdatePipeline()

    data_file_B = os.path.join(output_base, 'data', f'B_slice_t{t_idx:03d}.csv')
    SaveData(data_file_B, slice_B,
             PointDataArrays=['B', 'Points'],
             Precision=6, FieldAssociation='Point Data')

    Delete(slice_B)
    gc.collect()

    process_slice_directly(data_file_B,
                           os.path.join(output_base, 'B_maps', 'B'),
                           t_idx, 'B')

    slice_J = Slice(Input=calculator_J)
    slice_J.SliceType = 'Plane'
    slice_J.SliceType.Origin = [center_point[0], center_point[1], slice_height]
    slice_J.SliceType.Normal = Z_prime.tolist()
    slice_J.Triangulatetheslice = 0
    UpdatePipeline()

    data_file_J = os.path.join(output_base, 'data', f'J_slice_t{t_idx:03d}.csv')
    SaveData(data_file_J, slice_J,
             PointDataArrays=['j', 'Points'],
             Precision=6, FieldAssociation='Point Data')

    Delete(slice_J)
    gc.collect()

    process_slice_directly(data_file_J,
                           os.path.join(output_base, 'J_maps', 'J'),
                           t_idx, 'J')

    elapsed = time.time() - total_start
    print(f"{elapsed / 60:.1f} min")

t_idx = 0

view = GetActiveViewOrCreate('RenderView')

annotate_display = GetDisplayProperties(annotate_time)
if annotate_display:
    annotate_display.WindowLocation = 'Lower Left Corner'
    annotate_display.FontSize = 16

Show(stream_tracer_B1, view)
Show(stream_tracer_B2, view)
Show(tube_B1, view)
Show(calculator_Jz_slice, view)
Show(stream_tracer_J1, view)
Show(tube_J1, view)
Show(stream_tracer_J2, view)
Show(tube_J2, view)
Show(stream_tracer_J3, view)
Show(tube_J3, view)

GetColorTransferFunction('B')
GetColorTransferFunction('j')
GetColorTransferFunction('Jz')

for tracer in [stream_tracer_B1, stream_tracer_B2]:
    display = GetDisplayProperties(tracer)
    if display:
        display.ColorArrayName = ['POINTS', 'B']
        display.LookupTable = GetColorTransferFunction('B')

for tracer in [stream_tracer_J1, stream_tracer_J2, stream_tracer_J3]:
    display = GetDisplayProperties(tracer)
    if display:
        display.ColorArrayName = ['POINTS', 'j']
        display.LookupTable = GetColorTransferFunction('j')

slice_display = GetDisplayProperties(calculator_Jz_slice)
if slice_display:
    slice_display.Representation = 'Surface'
    slice_display.ColorArrayName = ['POINTS', 'Jz']
    slice_display.LookupTable = GetColorTransferFunction('Jz')

Render()

total_time = time.time() - total_start
print(f"Total time: {total_time / 60:.1f} min")
