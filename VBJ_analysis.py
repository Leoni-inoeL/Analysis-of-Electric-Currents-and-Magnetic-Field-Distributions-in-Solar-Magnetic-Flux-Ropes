from paraview.simple import *
import paraview
import sys
import os
import numpy as np
import gc
import time
import signal
from paraview.simple import CellDatatoPointData


def signal_handler(sig, frame):
    plt.close('all')
    gc.collect()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

source = GetActiveSource()
if source is None:
    raise RuntimeError("No active source found. Please load the file first.")

annotate_time = AnnotateTime()
annotate_time.Format = 'Time: {time:.2f}'
RenameSource('AnnotateTime', annotate_time)
annotate_time.UpdatePipeline()

calculator_V = Calculator(Input=source)
calculator_V.ResultArrayName = 'V'
calculator_V.Function = '116.45 * (iHat*v1 + jHat*v2 + kHat*v3)'
RenameSource('V_km/s', calculator_V)
calculator_V.UpdatePipeline()

calculator_J = Calculator(Input=source)
calculator_J.ResultArrayName = 'j'
calculator_J.Function = '15.89619*(iHat*j1 + jHat*j2 + kHat*j3)'
RenameSource('J_A/km^2', calculator_J)
calculator_J.UpdatePipeline()

calculator_B = Calculator(Input=source)
calculator_B.ResultArrayName = 'B'
calculator_B.Function = '1.99757357615242*(iHat*b1 + jHat*b2 + kHat*b3)'
RenameSource('B_Gs', calculator_B)
calculator_B.UpdatePipeline()

left_foot_coordinates = [-1.5, 0.0, 0.3]
right_foot_coordinates = [1.5, 0.0, 0.3]
radius = 0.3
n_points = 25
tube_radius = 0.1
n_sides = 6

stream_tracer_B1 = StreamTracer(Input=calculator_B)
stream_tracer_B1.SeedType = 'Point Cloud'
stream_tracer_B1.SeedType.Center = left_foot_coordinates
stream_tracer_B1.SeedType.Radius = radius
stream_tracer_B1.SeedType.NumberOfPoints = n_points
stream_tracer_B1.MaximumStreamlineLength = 12.0
stream_tracer_B1.Vectors = ['POINTS', 'B']
RenameSource('B_RFP_TRACER', stream_tracer_B1)
stream_tracer_B1.UpdatePipeline()

tube_B1 = Tube(Input=stream_tracer_B1)
tube_B1.Scalars = ['POINTS', 'B']
tube_B1.Radius = tube_radius
tube_B1.NumberofSides = n_sides
RenameSource('Tube3', tube_B1)
tube_B1.UpdatePipeline()

stream_tracer_B2 = StreamTracer(Input=calculator_B)
stream_tracer_B2.SeedType = 'Point Cloud'
stream_tracer_B2.SeedType.Center = left_foot_coordinates
stream_tracer_B2.SeedType.Radius = radius
stream_tracer_B2.SeedType.NumberOfPoints = n_points
stream_tracer_B2.MaximumStreamlineLength = 24.0
stream_tracer_B2.Vectors = ['POINTS', 'B']
RenameSource('B_LFP_tracer', stream_tracer_B2)
stream_tracer_B2.UpdatePipeline()

slice_filter = Slice(Input=calculator_J)
slice_filter.SliceType = 'Plane'
slice_filter.SliceType.Origin = [0.0, 0.0, 0.01]
slice_filter.SliceType.Normal = [0.0, 0.0, 1.0]
slice_filter.Triangulatetheslice = 0
RenameSource('Jz_slice_z=0.01Mm', slice_filter)
slice_filter.UpdatePipeline()

calculator_Jz_slice = Calculator(Input=slice_filter)
calculator_Jz_slice.ResultArrayName = 'Jz'
calculator_Jz_slice.Function = '15.89619*j3'
RenameSource('Jz_on_slice', calculator_Jz_slice)
calculator_Jz_slice.UpdatePipeline()

cell_to_point = CellDatatoPointData(Input=calculator_Jz_slice)
cell_to_point.CellDataArraytoprocess = ['Jz']
cell_to_point.UpdatePipeline()

stream_tracer_J1 = StreamTracer(Input=calculator_J)
stream_tracer_J1.SeedType = 'Point Cloud'
stream_tracer_J1.SeedType.Center = right_foot_coordinates
stream_tracer_J1.SeedType.Radius = radius
stream_tracer_J1.SeedType.NumberOfPoints = 2 * n_points
stream_tracer_J1.MaximumStreamlineLength = 64.0
stream_tracer_J1.Vectors = ['POINTS', 'j']
stream_tracer_J1.IntegrationDirection = 'BOTH'
stream_tracer_J1.IntegratorType = 'Runge-Kutta 4-5'
RenameSource('J_RFP_tracer', stream_tracer_J1)
stream_tracer_J1.UpdatePipeline()

tube_J1 = Tube(Input=stream_tracer_J1)
tube_J1.Scalars = ['POINTS', 'j']
tube_J1.Radius = tube_radius
tube_J1.NumberofSides = n_sides
RenameSource('Tube1', tube_J1)
tube_J1.UpdatePipeline()

stream_tracer_J2 = StreamTracer(Input=calculator_J)
stream_tracer_J2.SeedType = 'Point Cloud'
stream_tracer_J2.SeedType.Center = right_foot_coordinates
stream_tracer_J2.SeedType.Radius = radius
stream_tracer_J2.SeedType.NumberOfPoints = 2 * n_points
stream_tracer_J2.MaximumStreamlineLength = 256.0
stream_tracer_J2.Vectors = ['POINTS', 'j']
stream_tracer_J2.IntegrationDirection = 'BOTH'
stream_tracer_J2.IntegratorType = 'Runge-Kutta 4-5'
RenameSource('J_LFP_tracer', stream_tracer_J2)
stream_tracer_J2.UpdatePipeline()

tube_J2 = Tube(Input=stream_tracer_J2)
tube_J2.Scalars = ['POINTS', 'j']
tube_J2.Radius = tube_radius
tube_J2.NumberofSides = n_sides
RenameSource('Tube4', tube_J2)
tube_J2.UpdatePipeline()

stream_tracer_J3 = StreamTracer(Input=calculator_J)
stream_tracer_J3.SeedType = 'Point Cloud'
stream_tracer_J3.SeedType.Center = [0.0, 0.0, 0.4]
stream_tracer_J3.SeedType.Radius = tube_radius
stream_tracer_J3.SeedType.NumberOfPoints = 2 * n_points
stream_tracer_J3.MaximumStreamlineLength = 64.0
stream_tracer_J3.Vectors = ['POINTS', 'j']
stream_tracer_J3.IntegrationDirection = 'BOTH'
stream_tracer_J3.IntegratorType = 'Runge-Kutta 4-5'
RenameSource('J_Below_z=4Mm', stream_tracer_J3)
stream_tracer_J3.UpdatePipeline()

tube_J3 = Tube(Input=stream_tracer_J3)
tube_J3.Scalars = ['POINTS', 'j']
tube_J3.Radius = tube_radius
tube_J3.NumberofSides = n_sides
RenameSource('Tube5', tube_J3)
tube_J3.UpdatePipeline()

regions = {
    'footpoint': {
        'center': [-1.5, 0.0, 0.3],
        'radius': 0.3,
        'description': 'Left footpoint of the loop'
    },
    'apex': {
        'center': [0.0, 0.0, 1.5],
        'radius': 0.3,
        'description': 'Apex of the loop'
    }
}

output_base = 'C:/Users/user/Desktop/SummerPractice/time_series/'

for region_name in regions.keys():
    os.makedirs(os.path.join(output_base, region_name, 'data'), exist_ok=True)
    os.makedirs(os.path.join(output_base, region_name, 'B_maps'), exist_ok=True)
    os.makedirs(os.path.join(output_base, region_name, 'J_maps'), exist_ok=True)
    os.makedirs(os.path.join(output_base, region_name, 'spheres'), exist_ok=True)
    os.makedirs(os.path.join(output_base, region_name, 'profiles'), exist_ok=True)
    os.makedirs(os.path.join(output_base, region_name, 'model_fits'), exist_ok=True)
    os.makedirs(os.path.join(output_base, region_name, 'parameters'), exist_ok=True)


def calculate_local_axes(calculator, center_, radius_):
    clip = Clip(Input=calculator)
    clip.ClipType = 'Sphere'
    clip.ClipType.Center = center_
    clip.ClipType.Radius = radius_
    clip.Invert = 0
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

    del clip, integrate, data, cell_data
    gc.collect()

    if volume > 0:
        b1_mean = b1_total / volume
        b2_mean = b2_total / volume
        b3_mean = b3_total / volume
        b_mean = np.array([b1_mean, b2_mean, b3_mean])
        b_norm = b_mean / np.linalg.norm(b_mean)
        print(f"  B_mean = [{b1_mean:.6f}, {b2_mean:.6f}, {b3_mean:.6f}], |B| = {np.linalg.norm(b_mean):.6f}")
    else:
        print(f"  WARNING: Zero volume, using default Z axis")
        b_norm = np.array([0.0, 0.0, 1.0])

    z_prime_ = b_norm

    if abs(z_prime_[2]) < 0.9:
        arbitrary = np.array([0.0, 0.0, 1.0])
    else:
        arbitrary = np.array([1.0, 0.0, 0.0])

    x_prime_ = np.cross(arbitrary, z_prime_)
    if np.linalg.norm(x_prime_) > 1e-10:
        x_prime_ = x_prime_ / np.linalg.norm(x_prime_)
    else:
        x_prime_ = np.array([1.0, 0.0, 0.0])

    y_prime_ = np.cross(z_prime_, x_prime_)

    print(f"  X' = [{x_prime_[0]:.4f}, {x_prime_[1]:.4f}, {x_prime_[2]:.4f}]")
    print(f"  Y' = [{y_prime_[0]:.4f}, {y_prime_[1]:.4f}, {y_prime_[2]:.4f}]")
    print(f"  Z' = [{z_prime_[0]:.4f}, {z_prime_[1]:.4f}, {z_prime_[2]:.4f}]")

    return x_prime_, y_prime_, z_prime_


def point_cylindrical_coords(px, py, pz, center_, z_axis, x_axis, y_axis):
    r_vec = np.array([px, py, pz]) - np.array(center_)
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


def point_vector_decomposition(px, py, pz, vx, vy, vz, center_, z_axis, x_axis, y_axis):
    r_vec = np.array([px, py, pz]) - np.array(center_)
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


def save_radial_profile_b(region_name_, t_idx_, axes_, center_):
    x_axis, y_axis, z_axis = axes_  

    slice_filter = Slice(Input=calculator_B)
    slice_filter.SliceType = 'Plane'
    slice_filter.SliceType.Origin = center_
    slice_filter.SliceType.Normal = z_axis.tolist()
    slice_filter.Triangulatetheslice = 0
    UpdatePipeline()

    data = servermanager.Fetch(slice_filter)
    points = data.GetPoints()
    point_data = data.GetPointData()

    b_array = point_data.GetArray('B')
    if b_array is None:
        Delete(slice_filter)
        gc.collect()
        return None, None, None

    radii = []
    b_z_vals = []
    b_phi_vals = []
    b_r_vals = []
    b_mag_vals = []

    for i in range(points.GetNumberOfPoints()):
        px, py, pz = points.GetPoint(i)
        bx, by, bz = b_array.GetTuple(i)

        r, phi, x_prime_local, y_prime_local = point_cylindrical_coords(
            px, py, pz, center_, z_axis, x_axis, y_axis
        )

        br, bphi, bz_comp = point_vector_decomposition(
            px, py, pz, bx, by, bz, center_, z_axis, x_axis, y_axis
        )

        radii.append(r)
        b_r_vals.append(br)
        b_phi_vals.append(bphi)
        b_z_vals.append(bz_comp)
        b_mag_vals.append(np.sqrt(bx * bx + by * by + bz * bz))

    profile_file = os.path.join(output_base, region_name_, 'profiles',
                                f'B_profile_t{t_idx_:03d}.txt')
    with open(profile_file, 'w') as f_:
        f_.write('r,B_r,B_phi,B_z,B_mag\n')
        for i in range(len(radii)):
            f_.write(f'{radii[i]:.6f},{b_r_vals[i]:.6f},{b_phi_vals[i]:.6f},{b_z_vals[i]:.6f},{b_mag_vals[i]:.6f}\n')

    Delete(slice_filter)
    gc.collect()

    return radii, b_z_vals, b_phi_vals


def save_radial_profile_j(region_name_, t_idx_, axes_, center_):
    x_axis, y_axis, z_axis = axes_

    slice_filter = Slice(Input=calculator_J)
    slice_filter.SliceType = 'Plane'
    slice_filter.SliceType.Origin = center_
    slice_filter.SliceType.Normal = z_axis.tolist()
    slice_filter.Triangulatetheslice = 0
    UpdatePipeline()

    data = servermanager.Fetch(slice_filter)
    points = data.GetPoints()
    point_data = data.GetPointData()

    j_array = point_data.GetArray('j')
    if j_array is None:
        Delete(slice_filter)
        gc.collect()
        return None, None, None

    radii = []
    j_z_vals = []
    j_phi_vals = []
    j_r_vals = []
    j_mag_vals = []

    for i in range(points.GetNumberOfPoints()):
        px, py, pz = points.GetPoint(i)
        jx, jy, jz = j_array.GetTuple(i)

        r, phi, _, _ = point_cylindrical_coords(
            px, py, pz, center_, z_axis, x_axis, y_axis
        )

        jr, jphi, jz_comp = point_vector_decomposition(
            px, py, pz, jx, jy, jz, center_, z_axis, x_axis, y_axis
        )

        radii.append(r)
        j_r_vals.append(jr)
        j_phi_vals.append(jphi)
        j_z_vals.append(jz_comp)
        j_mag_vals.append(np.sqrt(jx * jx + jy * jy + jz * jz))

    profile_file = os.path.join(output_base, region_name_, 'profiles',
                                f'J_profile_t{t_idx_:03d}.txt')
    with open(profile_file, 'w') as f_:
        f_.write('r,J_r,J_phi,J_z,J_mag\n')
        for i in range(len(radii)):
            f_.write(f'{radii[i]:.6f},{j_r_vals[i]:.6f},{j_phi_vals[i]:.6f},{j_z_vals[i]:.6f},{j_mag_vals[i]:.6f}\n')

    Delete(slice_filter)
    gc.collect()

    return radii, j_z_vals, j_phi_vals


def calculate_g_parameter(radii, b_z_vals):
    radii = np.array(radii)
    b_z_vals = np.array(b_z_vals)
    
    idx_center = np.argmin(np.abs(radii))
    b_0 = np.abs(b_z_vals[idx_center])
    
    idx_edge = np.argmin(np.abs(radii - np.max(radii)))
    bz_ex = np.abs(b_z_vals[idx_edge])
    
    g = (bz_ex / b_0)**2 if b_0 > 0 else 0
    
    return g, b_0, bz_ex


def find_critical_surface(radii, b_z_vals):
    radii = np.array(radii)
    b_z_vals = np.array(b_z_vals)
    
    sign_changes = np.where(np.diff(np.sign(b_z_vals)))[0]
    
    critical_radii = []
    for idx in sign_changes:
        if idx < len(radii) - 1:
            r1, r2 = radii[idx], radii[idx + 1]
            bz1, bz2 = b_z_vals[idx], b_z_vals[idx + 1]
            if bz2 - bz1 != 0:
                r_crit = r1 - bz1 * (r2 - r1) / (bz2 - bz1)
                critical_radii.append(r_crit)
    
    return critical_radii


def save_alpha_parameter(region_name_, t_idx_, radii, b_z_vals, b_phi_vals):
    radii = np.array(radii)
    b_z_vals = np.array(b_z_vals)
    b_phi_vals = np.array(b_phi_vals)
    
    alpha_vals = []
    valid_radii = []
    
    for i in range(len(radii)):
        r = radii[i]
        bz = b_z_vals[i]
        
        if r > 0.01 and abs(bz) > 1e-6:
            dr = 0.01
            idx_plus = np.argmin(np.abs(radii - (r + dr)))
            idx_minus = np.argmin(np.abs(radii - (r - dr)))
            
            if idx_plus != idx_minus and idx_plus < len(radii) and idx_minus < len(radii):
                dbphi_dr = (b_phi_vals[idx_plus] - b_phi_vals[idx_minus]) / (radii[idx_plus] - radii[idx_minus])
                alpha = -dbphi_dr / bz
                alpha_vals.append(alpha)
                valid_radii.append(r)
    
    alpha_file = os.path.join(output_base, region_name_, 'parameters',
                              f'alpha_t{t_idx_:03d}.txt')
    with open(alpha_file, 'w') as f_:
        f_.write('r,alpha\n')
        for i in range(len(valid_radii)):
            f_.write(f'{valid_radii[i]:.6f},{alpha_vals[i]:.6f}\n')
    
    return valid_radii, alpha_vals


def process_slice(region_name_, t_idx_, axes_, center_):
    x_axis, y_axis, z_axis = axes_

    slice_filter = Slice(Input=calculator_B)
    slice_filter.SliceType = 'Plane'
    slice_filter.SliceType.Origin = center_
    slice_filter.SliceType.Normal = z_axis.tolist()
    slice_filter.Triangulatetheslice = 0
    UpdatePipeline()

    data_file = os.path.join(output_base, region_name_, 'data',
                             f'B_slice_t{t_idx_:03d}.csv')
    SaveData(data_file, slice_filter,
             PointDataArrays=['B', 'Points'],
             Precision=6, FieldAssociation='Point Data')

    with open(data_file, 'r') as f_:
        header_line = f_.readline().strip()
        header = [col.strip('"') for col in header_line.split(',')]
        px_idx = header.index('Points:0')
        py_idx = header.index('Points:1')
        pz_idx = header.index('Points:2')
        vx_idx = header.index('B:0')
        vy_idx = header.index('B:1')
        vz_idx = header.index('B:2')

        output_prefix = os.path.join(output_base, region_name_, 'B_maps', 'B')

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

            for line in f_:
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

                r, phi, x_prime_local, y_prime_local = point_cylindrical_coords(
                    px, py, pz, center_, z_axis, x_axis, y_axis
                )

                vr, vphi, vz_comp = point_vector_decomposition(
                    px, py, pz, vx, vy, vz, center_, z_axis, x_axis, y_axis
                )

                f_r.write(f'{x_prime_local:.6f},{y_prime_local:.6f},{r:.6f}\n')
                f_phi.write(f'{x_prime_local:.6f},{y_prime_local:.6f},{phi:.6f},{np.degrees(phi):.6f}\n')
                f_vr.write(f'{x_prime_local:.6f},{y_prime_local:.6f},{vr:.6f}\n')
                f_vphi.write(f'{x_prime_local:.6f},{y_prime_local:.6f},{vphi:.6f}\n')
                f_vz.write(f'{x_prime_local:.6f},{y_prime_local:.6f},{vz_comp:.6f}\n')

    Delete(slice_filter)
    gc.collect()


def save_sphere_data(region_name_, t_idx_):
    region = regions[region_name_]
    center_ = region['center']
    radius_ = region['radius']

    clip_B = Clip(Input=calculator_B)
    clip_B.ClipType = 'Sphere'
    clip_B.ClipType.Center = center_
    clip_B.ClipType.Radius = radius_
    clip_B.Invert = 0
    UpdatePipeline()

    csv_file = os.path.join(output_base, region_name_, 'spheres',
                            f'{region_name_}_B_t{t_idx_:03d}.csv')
    SaveData(csv_file, clip_B, PointDataArrays=['B', 'Points'], Precision=6)
    Delete(clip_B)

    gc.collect()


animation_scene = GetAnimationScene()
timesteps = animation_scene.TimeKeeper.TimestepValues

total_start = time.time()

region_axes = {region_name: {} for region_name in regions.keys()}

for t_idx, time_val in enumerate(timesteps):


    animation_scene.TimeKeeper.Time = time_val
    Render()

    for region_name, region_info in regions.items():
        print(f"\n--- {region_name.upper()} ({region_info['description']}) ---")
        print(f"Center: {region_info['center']}")

        X_prime, Y_prime, z_prime = calculate_local_axes(
            calculator_B,
            region_info['center'],
            region_info['radius']
        )

        region_axes[region_name][t_idx] = (X_prime, Y_prime, z_prime)

        save_sphere_data(region_name, t_idx)

        axes = (X_prime, Y_prime, z_prime)
        center = region_info['center']

        process_slice(region_name, t_idx, axes, center)
        
        radii_b, b_z, b_phi = save_radial_profile_b(region_name, t_idx, axes, center)
        save_radial_profile_j(region_name, t_idx, axes, center)
        
        if radii_b and len(radii_b) > 0:
            G, B0, Bz_ex = calculate_g_parameter(radii_b, b_z)
            print(f"  G = {G:.6f}, B0 = {B0:.6f}, Bz_ex = {Bz_ex:.6f}")
            
            params_file = os.path.join(output_base, region_name, 'parameters',
                                        f'parameters_t{t_idx:03d}.txt')
            with open(params_file, 'w') as f:
                f.write(f'time,{time_val}\n')
                f.write(f'G,{G}\n')
                f.write(f'B0,{B0}\n')
                f.write(f'Bz_ex,{Bz_ex}\n')
                f.write(f'region_center_x,{center[0]}\n')
                f.write(f'region_center_y,{center[1]}\n')
                f.write(f'region_center_z,{center[2]}\n')
                f.write(f'Z_prime_x,{z_prime[0]}\n')
                f.write(f'Z_prime_y,{z_prime[1]}\n')
                f.write(f'Z_prime_z,{z_prime[2]}\n')
            
            critical_surfaces = find_critical_surface(radii_b, b_z)
            if critical_surfaces:
                print(f"  Critical surface(s) at r = {critical_surfaces}")
                with open(params_file, 'a') as f:
                    f.write(f'critical_surfaces,{critical_surfaces}\n')
            
            save_alpha_parameter(region_name, t_idx, radii_b, b_z, b_phi)

    elapsed = time.time() - total_start
    print(f"\nTime elapsed for step {t_idx}: {(time.time() - total_start):.1f} sec")
    print(f"Total elapsed: {elapsed / 60:.1f} min")

    gc.collect()

view = GetActiveViewOrCreate('RenderView')

annotate_display = GetDisplayProperties(annotate_time)
if annotate_display:
    annotate_display.WindowLocation = 'Lower Left Corner'
    annotate_display.FontSize = 16

Show(stream_tracer_B1, view)
Show(stream_tracer_B2, view)

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

SetActiveSource(cell_to_point)
cell_to_point.UpdatePipeline()

Jz_lut = GetColorTransferFunction('Jz')
Jz_lut.RescaleTransferFunction(-10, 10)

slice_display = Show(cell_to_point, view)
if slice_display:
    slice_display.Representation = 'Surface'
    slice_display.ColorArrayName = ['POINTS', 'Jz']
    slice_display.LookupTable = Jz_lut

B_lut = GetColorTransferFunction('B')
B_lut.RescaleTransferFunction(0, 500)

j_lut = GetColorTransferFunction('j')
j_lut.RescaleTransferFunction(0, 30)

source_display = GetDisplayProperties(source)
if source_display:
    source_display.Visibility = 0

slice_disp = GetDisplayProperties(slice_filter)
if slice_disp:
    slice_disp.Visibility = 0

orig_disp = GetDisplayProperties(calculator_Jz_slice)
if orig_disp:
    orig_disp.Visibility = 0

Render()

total_time = time.time() - total_start
print(f"Total processing time: {total_time / 60:.1f} min")
