from paraview.simple import *
from paraview import servermanager
import sys
import os
import numpy as np
import h5py
import gc
import signal

def signal_handler(sig, frame):
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

OUTPUT_BASE = 'C:/Users/user/Desktop/SummerPractice/AnalysisData'
os.makedirs(OUTPUT_BASE, exist_ok=True)

REGIONS = {
    'footpoint': {
        'center': [-1.5, 0.0, 0.3],
        'radius': 0.3,
        'dynamic_center': True
    },
    'apex': {
        'center': [0.0, 0.0, 1.5],
        'radius': 0.3,
        'dynamic_center': True
    }
}

N_BINS = 50
PLANE_RESOLUTION = 128

B_FACTOR = 1.99757357615242
J_FACTOR = 15.89619

START_STEP = 0
END_STEP = 95

source = GetActiveSource()
if source is None:
    raise RuntimeError("No active source found. Please load the file first.")

print("Converting CellData to PointData...")
cell_to_point_main = CellDatatoPointData(Input=source)
cell_to_point_main.PassCellData = 1
cell_to_point_main.UpdatePipeline()

print("PointData arrays in converted source:")
point_arrays = []
for arr in cell_to_point_main.PointData.keys():
    print(f"  - {arr}")
    point_arrays.append(arr)

if len(point_arrays) == 0:
    print("ERROR: No PointData arrays after conversion!")
    sys.exit(1)

annotate_time = AnnotateTime()
annotate_time.Format = 'Time: {time:.2f}'
RenameSource('AnnotateTime', annotate_time)
annotate_time.UpdatePipeline()

calculator_B = Calculator(Input=cell_to_point_main)
calculator_B.ResultArrayName = 'B'
calculator_B.Function = f'{B_FACTOR} * (iHat*b1 + jHat*b2 + kHat*b3)'
calculator_B.UpdatePipeline()

calculator_J = Calculator(Input=calculator_B)
calculator_J.ResultArrayName = 'J'
calculator_J.Function = f'{J_FACTOR} * (iHat*j1 + jHat*j2 + kHat*j3)'
calculator_J.UpdatePipeline()

previous_axes = {
    'footpoint': None,
    'apex': None
}

def compute_global_bmax(calculator, time_val):
    calculator.UpdatePipeline(time_val)
    data_tmp = servermanager.Fetch(calculator)
    point_data = data_tmp.GetPointData()

    b_array_tmp = point_data.GetArray('B')

    if b_array_tmp is None:
        print(f"  WARNING: B array not found!")
        return 1.0

    bmax = 0.0
    for i in range(b_array_tmp.GetNumberOfTuples()):
        bx, by, bz = b_array_tmp.GetTuple(i)
        bmag = np.sqrt(bx*bx + by*by + bz*bz)
        if bmag > bmax:
            bmax = bmag

    del data_tmp
    return bmax

def find_dynamic_center(calculator, approximate_center, radius, time_val):
    calculator.UpdatePipeline(time_val)

    clip = Clip(Input=calculator)
    clip.ClipType = 'Sphere'
    clip.ClipType.Center = approximate_center
    clip.ClipType.Radius = radius
    clip.Invert = 0
    clip.UpdatePipeline()

    data = servermanager.Fetch(clip)
    points = data.GetPoints()
    point_data = data.GetPointData()

    field_array = point_data.GetArray('B')

    if field_array is None:
        Delete(clip)
        del data
        return np.array(approximate_center)

    total_weight = 0.0
    weighted_center = np.zeros(3)
    npts = points.GetNumberOfPoints()

    for i in range(npts):
        px, py, pz = points.GetPoint(i)
        fx, fy, fz = field_array.GetTuple(i)
        
        if not np.isfinite(fx + fy + fz):
            continue
        
        fmag2 = fx*fx + fy*fy + fz*fz
        
        weighted_center[0] += fmag2 * px
        weighted_center[1] += fmag2 * py
        weighted_center[2] += fmag2 * pz
        total_weight += fmag2

    Delete(clip)
    del data

    if total_weight > 0:
        weighted_center /= total_weight
        return weighted_center

    return np.array(approximate_center)

def calculate_local_axes(calculator, center, radius, bmax_global, previous_axis, time_val):
    calculator.UpdatePipeline(time_val)

    clip = Clip(Input=calculator)
    clip.ClipType = 'Sphere'
    clip.ClipType.Center = center
    clip.ClipType.Radius = radius
    clip.Invert = 0
    clip.UpdatePipeline()

    point_to_cell = PointDatatoCellData(Input=clip)
    point_to_cell.PassPointData = 1

    integrate = IntegrateVariables(Input=point_to_cell)
    integrate.UpdatePipeline()

    data = servermanager.Fetch(integrate)
    cell_data = data.GetCellData()

    volume_array = cell_data.GetArray('Volume')
    if volume_array is not None:
        volume = volume_array.GetValue(0)
    else:
        volume = 0.0

    b_array = cell_data.GetArray('B')

    if b_array is not None and b_array.GetNumberOfTuples() > 0:
        b_integrated = np.array(b_array.GetTuple(0))
    else:
        b_integrated = np.array([0.0, 0.0, 0.0])

    Delete(point_to_cell)
    Delete(clip)
    Delete(integrate)
    del data

    if volume > 0:
        b_mean = b_integrated / volume
        b_norm = np.linalg.norm(b_mean)
        
        if b_norm < 1e-3 * bmax_global:
            if previous_axis is not None:
                return previous_axis
            else:
                return np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])
        
        z_prime = b_mean / b_norm
    else:
        if previous_axis is not None:
            return previous_axis
        z_prime = np.array([0.0, 0.0, 1.0])

    if abs(z_prime[2]) < 0.9:
        arbitrary = np.array([0.0, 0.0, 1.0])
    else:
        arbitrary = np.array([1.0, 0.0, 0.0])

    x_prime = np.cross(arbitrary, z_prime)
    x_norm = np.linalg.norm(x_prime)
    if x_norm > 1e-12:
        x_prime /= x_norm
    else:
        x_prime = np.array([1.0, 0.0, 0.0])

    y_prime = np.cross(z_prime, x_prime)
    y_norm = np.linalg.norm(y_prime)
    if y_norm > 1e-12:
        y_prime /= y_norm

    handedness = np.dot(np.cross(x_prime, y_prime), z_prime)
    if handedness < 0:
        y_prime = -y_prime

    if previous_axis is not None:
        alpha = 0.8
        z_prev = previous_axis[2]
        
        if np.dot(z_prime, z_prev) < 0:
            z_prime = -z_prime
        
        z_prime = alpha * z_prev + (1 - alpha) * z_prime
        z_norm = np.linalg.norm(z_prime)
        
        if z_norm > 1e-12:
            z_prime /= z_norm
        else:
            z_prime = z_prev
        
        if abs(z_prime[2]) < 0.9:
            arbitrary = np.array([0.0, 0.0, 1.0])
        else:
            arbitrary = np.array([1.0, 0.0, 0.0])
        
        x_prime = np.cross(arbitrary, z_prime)
        x_norm = np.linalg.norm(x_prime)
        if x_norm > 1e-12:
            x_prime /= x_norm
        elif previous_axis is not None:
            x_prime = previous_axis[0]
        
        y_prime = np.cross(z_prime, x_prime)
        y_norm = np.linalg.norm(y_prime)
        if y_norm > 1e-12:
            y_prime /= y_norm
        elif previous_axis is not None:
            y_prime = previous_axis[1]
        
        handedness = np.dot(np.cross(x_prime, y_prime), z_prime)
        if handedness < 0:
            y_prime = -y_prime

    return x_prime, y_prime, z_prime

def point_vector_decomposition(px, py, pz, vx, vy, vz, center, z_axis, x_axis, y_axis):
    r_vec = np.array([px, py, pz]) - np.array(center)
    vector = np.array([vx, vy, vz])

    vz_comp = np.dot(vector, z_axis)
    z_val = np.dot(r_vec, z_axis)
    r_proj = r_vec - z_val * z_axis
    r_dist = np.linalg.norm(r_proj)

    if r_dist > 1e-12:
        e_r = r_proj / r_dist
        e_phi = np.cross(z_axis, e_r)
        e_phi -= np.dot(e_phi, z_axis) * z_axis
        e_phi_norm = np.linalg.norm(e_phi)
        if e_phi_norm > 1e-12:
            e_phi /= e_phi_norm
        else:
            e_phi = np.array([0.0, 0.0, 0.0])
        vr_comp = np.dot(vector, e_r)
        vphi_comp = np.dot(vector, e_phi)
    else:
        vr_comp = 0.0
        vphi_comp = 0.0

    return vr_comp, vphi_comp, vz_comp

def create_sampling_plane(center, x_axis, y_axis, plane_radius, resolution):
    plane = Plane()
    plane.XResolution = resolution - 1
    plane.YResolution = resolution - 1

    origin = np.array(center) - plane_radius * np.array(x_axis) - plane_radius * np.array(y_axis)
    point1 = np.array(center) + plane_radius * np.array(x_axis) - plane_radius * np.array(y_axis)
    point2 = np.array(center) - plane_radius * np.array(x_axis) + plane_radius * np.array(y_axis)

    plane.Origin = origin.tolist()
    plane.Point1 = point1.tolist()
    plane.Point2 = point2.tolist()
    plane.UpdatePipeline()

    return plane

def sample_on_plane(calculator, center, x_axis, y_axis, plane_radius, resolution, time_val):
    calculator.UpdatePipeline(time_val)

    plane = create_sampling_plane(center, x_axis, y_axis, plane_radius, resolution)

    sampled = ResampleWithDataset(
        SourceDataArrays=calculator,
        DestinationMesh=plane
    )
    sampled.UpdatePipeline()

    Delete(plane)

    return sampled

def compute_profiles_from_maps(r_vals, br_vals, bphi_vals, bz_vals, bmag_vals, jr_vals, jphi_vals, jz_vals, jmag_vals, alpha_vals, forcefree_vals, sigma_vals, n_bins):
    valid = np.isfinite(r_vals) & (r_vals > 0.01)

    r_vals = r_vals[valid]
    br_vals = br_vals[valid]
    bphi_vals = bphi_vals[valid]
    bz_vals = bz_vals[valid]
    bmag_vals = bmag_vals[valid]
    jr_vals = jr_vals[valid]
    jphi_vals = jphi_vals[valid]
    jz_vals = jz_vals[valid]
    jmag_vals = jmag_vals[valid]
    alpha_vals = alpha_vals[valid]
    forcefree_vals = forcefree_vals[valid]
    sigma_vals = sigma_vals[valid]

    if len(r_vals) == 0:
        return None

    sort_idx = np.argsort(r_vals)
    r_vals = r_vals[sort_idx]
    br_vals = br_vals[sort_idx]
    bphi_vals = bphi_vals[sort_idx]
    bz_vals = bz_vals[sort_idx]
    bmag_vals = bmag_vals[sort_idx]
    jr_vals = jr_vals[sort_idx]
    jphi_vals = jphi_vals[sort_idx]
    jz_vals = jz_vals[sort_idx]
    jmag_vals = jmag_vals[sort_idx]
    alpha_vals = alpha_vals[sort_idx]
    forcefree_vals = forcefree_vals[sort_idx]
    sigma_vals = sigma_vals[sort_idx]

    r_max = np.percentile(r_vals, 95)
    r_bins = np.sqrt(np.linspace(0, r_max**2, n_bins + 1))
    bin_centers = 0.5 * (r_bins[:-1] + r_bins[1:])

    bz_mean = np.zeros(n_bins)
    bz_std = np.zeros(n_bins)
    bphi_mean = np.zeros(n_bins)
    bphi_std = np.zeros(n_bins)
    br_mean = np.zeros(n_bins)
    br_std = np.zeros(n_bins)
    bmag_mean = np.zeros(n_bins)
    bmag_std = np.zeros(n_bins)
    jz_mean = np.zeros(n_bins)
    jz_std = np.zeros(n_bins)
    jphi_mean = np.zeros(n_bins)
    jphi_std = np.zeros(n_bins)
    jr_mean = np.zeros(n_bins)
    jr_std = np.zeros(n_bins)
    jmag_mean = np.zeros(n_bins)
    jmag_std = np.zeros(n_bins)
    alpha_mean = np.zeros(n_bins)
    alpha_std = np.zeros(n_bins)
    forcefree_mean = np.zeros(n_bins)
    forcefree_std = np.zeros(n_bins)
    sigma_mean = np.zeros(n_bins)
    sigma_std = np.zeros(n_bins)
    n_per_bin = np.zeros(n_bins)

    twist_mean = np.full(n_bins, np.nan)
    twist_std = np.full(n_bins, np.nan)

    for i in range(n_bins):
        mask = (r_vals >= r_bins[i]) & (r_vals < r_bins[i+1])
        n_in_bin = np.sum(mask)
        if n_in_bin > 0:
            n_per_bin[i] = n_in_bin
            bz_mean[i] = np.mean(bz_vals[mask])
            bz_std[i] = np.std(bz_vals[mask])
            bphi_mean[i] = np.mean(bphi_vals[mask])
            bphi_std[i] = np.std(bphi_vals[mask])
            br_mean[i] = np.mean(br_vals[mask])
            br_std[i] = np.std(br_vals[mask])
            bmag_mean[i] = np.mean(bmag_vals[mask])
            bmag_std[i] = np.std(bmag_vals[mask])
            jz_mean[i] = np.mean(jz_vals[mask])
            jz_std[i] = np.std(jz_vals[mask])
            jphi_mean[i] = np.mean(jphi_vals[mask])
            jphi_std[i] = np.std(jphi_vals[mask])
            jr_mean[i] = np.mean(jr_vals[mask])
            jr_std[i] = np.std(jr_vals[mask])
            jmag_mean[i] = np.mean(jmag_vals[mask])
            jmag_std[i] = np.std(jmag_vals[mask])
            alpha_mean[i] = np.mean(alpha_vals[mask])
            alpha_std[i] = np.std(alpha_vals[mask])
            forcefree_mean[i] = np.mean(forcefree_vals[mask])
            forcefree_std[i] = np.std(forcefree_vals[mask])
            sigma_mean[i] = np.mean(sigma_vals[mask])
            sigma_std[i] = np.std(sigma_vals[mask])

    bz_ref = max(np.nanmax(np.abs(bz_mean)), 1e-12)

    valid_twist = (
        (n_per_bin > 5) &
        (bin_centers > 0.01) &
        (np.abs(bz_mean) > 0.05 * bz_ref)
    )

    twist_mean[valid_twist] = (
        bphi_mean[valid_twist] /
        (bin_centers[valid_twist] * bz_mean[valid_twist])
    )

    twist_std[valid_twist] = (
        np.abs(twist_mean[valid_twist]) *
        np.sqrt(
            (bphi_std[valid_twist] / (np.abs(bphi_mean[valid_twist]) + 1e-12))**2 +
            (bz_std[valid_twist] / (np.abs(bz_mean[valid_twist]) + 1e-12))**2
        )
    )

    bz0 = max(np.max(np.abs(bz_mean)), 1e-12)
    bphi0 = max(np.max(np.abs(bphi_mean)), 1e-12)
    bz_norm = bz_mean / bz0
    bphi_norm = bphi_mean / bphi0

    return {
        'r': bin_centers,
        'Bz_mean': bz_mean, 'Bz_std': bz_std,
        'Bphi_mean': bphi_mean, 'Bphi_std': bphi_std,
        'Br_mean': br_mean, 'Br_std': br_std,
        'Bmag_mean': bmag_mean, 'Bmag_std': bmag_std,
        'Jz_mean': jz_mean, 'Jz_std': jz_std,
        'Jphi_mean': jphi_mean, 'Jphi_std': jphi_std,
        'Jr_mean': jr_mean, 'Jr_std': jr_std,
        'Jmag_mean': jmag_mean, 'Jmag_std': jmag_std,
        'alpha_mean': alpha_mean, 'alpha_std': alpha_std,
        'forcefree_mean': forcefree_mean, 'forcefree_std': forcefree_std,
        'sigma_mean': sigma_mean, 'sigma_std': sigma_std,
        'Bz_norm': bz_norm, 'Bphi_norm': bphi_norm,
        'twist_mean': twist_mean, 'twist_std': twist_std,
        'n_per_bin': n_per_bin
    }

animation_scene = GetAnimationScene()
timesteps = animation_scene.TimeKeeper.TimestepValues

print(f"\nTotal timesteps: {len(timesteps)}")
print(f"Processing steps {START_STEP} to {END_STEP - 1}\n")

plane_radius_fixed = 0.6

for t_idx, time_val in enumerate(timesteps):
    if t_idx < START_STEP:
        continue

    if t_idx >= END_STEP:
        break

    output_file = os.path.join(OUTPUT_BASE, f'analysis_t{t_idx:04d}.h5')

    print(f"\n{'='*60}")
    print(f"START timestep {t_idx}: t = {time_val:.2f}")
    print(f"Output: {output_file}")
    print('='*60)

    try:
        animation_scene.TimeKeeper.Time = time_val
        
        calculator_B.UpdatePipeline(time_val)
        calculator_J.UpdatePipeline(time_val)
        
        bmax_global = compute_global_bmax(calculator_B, time_val)
        print(f"  Global Bmax = {bmax_global:.6f} G")
        
        with h5py.File(output_file, 'w') as h5file:
            h5file.attrs['format_version'] = '1.0'
            h5file.attrs['time_index'] = t_idx
            h5file.attrs['physical_time'] = time_val
            h5file.attrs['B_FACTOR'] = B_FACTOR
            h5file.attrs['J_FACTOR'] = J_FACTOR
            h5file.attrs['N_BINS'] = N_BINS
            h5file.attrs['PLANE_RESOLUTION'] = PLANE_RESOLUTION
            h5file.attrs['plane_radius_fixed'] = plane_radius_fixed
            h5file.attrs['B_units'] = 'G'
            h5file.attrs['J_units'] = 'A/km^2 (converted from code units)'
            h5file.attrs['length_units'] = 'Mm'
            h5file.attrs['Bz_definition'] = 'local axial component along z_axis'
            h5file.attrs['Bphi_definition'] = 'azimuthal component around local z_axis'
            h5file.attrs['twist_definition'] = 'Bphi/(r*Bz), local cylindrical proxy, not field-line turns'
            
            for region_name, region_info in REGIONS.items():
                h5file.attrs[f'{region_name}_center'] = region_info['center']
                h5file.attrs[f'{region_name}_radius'] = region_info['radius']
            
            for region_name, region_info in REGIONS.items():
                static_center = region_info['center']
                use_dynamic = region_info.get('dynamic_center', False)
                
                print(f"\n  Processing {region_name}...")
                
                if use_dynamic:
                    center = find_dynamic_center(calculator_B, static_center, region_info['radius'], time_val)
                    center_shift = np.linalg.norm(np.array(center) - np.array(static_center))
                    if center_shift > region_info['radius']:
                        center = static_center
                        dynamic_used = False
                        center_shift = 0.0
                    else:
                        dynamic_used = True
                else:
                    center = static_center
                    dynamic_used = False
                    center_shift = 0.0
                
                X_prime, Y_prime, Z_prime = calculate_local_axes(
                    calculator_B, center, region_info['radius'], bmax_global,
                    previous_axes[region_name], time_val
                )
                previous_axes[region_name] = (X_prime, Y_prime, Z_prime)
                
                region_group = h5file.create_group(region_name)
                region_group.attrs['plane_radius'] = plane_radius_fixed
                region_group.attrs['plane_resolution'] = PLANE_RESOLUTION
                
                axes_group = region_group.create_group('axes')
                axes_group.create_dataset('center', data=center)
                axes_group.create_dataset('x_axis', data=X_prime)
                axes_group.create_dataset('y_axis', data=Y_prime)
                axes_group.create_dataset('z_axis', data=Z_prime)
                axes_group.attrs['dynamic_center_used'] = dynamic_used
                axes_group.attrs['center_shift'] = center_shift
                
                sampled = sample_on_plane(calculator_J, center, X_prime, Y_prime, plane_radius_fixed, PLANE_RESOLUTION, time_val)
                
                data = servermanager.Fetch(sampled)
                points = data.GetPoints()
                point_data = data.GetPointData()
                
                b_array = point_data.GetArray('B')
                j_array = point_data.GetArray('J')
                
                if b_array is not None and j_array is not None:
                    n_points = points.GetNumberOfPoints()
                    resolution = PLANE_RESOLUTION
                    
                    X_map = np.full((resolution, resolution), np.nan)
                    Y_map = np.full((resolution, resolution), np.nan)
                    R_map = np.full((resolution, resolution), np.nan)
                    Phi_map = np.full((resolution, resolution), np.nan)
                    
                    Bx_local_map = np.full((resolution, resolution), np.nan)
                    By_local_map = np.full((resolution, resolution), np.nan)
                    Bz_local_map = np.full((resolution, resolution), np.nan)
                    Bphi_map = np.full((resolution, resolution), np.nan)
                    Br_map = np.full((resolution, resolution), np.nan)
                    Bmag_map = np.full((resolution, resolution), np.nan)
                    
                    Jx_local_map = np.full((resolution, resolution), np.nan)
                    Jy_local_map = np.full((resolution, resolution), np.nan)
                    Jz_local_map = np.full((resolution, resolution), np.nan)
                    Jphi_map = np.full((resolution, resolution), np.nan)
                    Jr_map = np.full((resolution, resolution), np.nan)
                    Jmag_map = np.full((resolution, resolution), np.nan)
                    
                    alpha_map = np.full((resolution, resolution), np.nan)
                    forcefree_map = np.full((resolution, resolution), np.nan)
                    sigma_map = np.full((resolution, resolution), np.nan)
                    
                    r_vals = []
                    br_vals = []
                    bphi_vals = []
                    bz_vals = []
                    bmag_vals = []
                    jr_vals = []
                    jphi_vals = []
                    jz_vals = []
                    jmag_vals = []
                    alpha_vals = []
                    forcefree_vals = []
                    sigma_vals = []
                    
                    for idx in range(n_points):
                        px, py, pz = points.GetPoint(idx)
                        bx, by, bz = b_array.GetTuple(idx)
                        jx, jy, jz = j_array.GetTuple(idx)
                        
                        if not np.isfinite(bx + by + bz):
                            continue
                        if not np.isfinite(jx + jy + jz):
                            continue
                        
                        iy = idx // resolution
                        ix = idx % resolution
                        
                        if ix < 0 or ix >= resolution or iy < 0 or iy >= resolution:
                            continue
                        
                        r_vec = np.array([px, py, pz]) - np.array(center)
                        xp = np.dot(r_vec, X_prime)
                        yp = np.dot(r_vec, Y_prime)
                        r_cyl = np.sqrt(xp*xp + yp*yp)
                        phi = np.arctan2(yp, xp)
                        
                        br, bphi, bz_comp = point_vector_decomposition(
                            px, py, pz, bx, by, bz, center, Z_prime, X_prime, Y_prime
                        )
                        
                        jr, jphi, jz_comp = point_vector_decomposition(
                            px, py, pz, jx, jy, jz, center, Z_prime, X_prime, Y_prime
                        )
                        
                        B_vec = np.array([bx, by, bz])
                        J_vec = np.array([jx, jy, jz])
                        
                        Bx_local = np.dot(B_vec, X_prime)
                        By_local = np.dot(B_vec, Y_prime)
                        
                        Jx_local = np.dot(J_vec, X_prime)
                        Jy_local = np.dot(J_vec, Y_prime)
                        
                        bmag = np.sqrt(bx*bx + by*by + bz*bz)
                        jmag = np.sqrt(jx*jx + jy*jy + jz*jz)
                        
                        Bmag2 = np.dot(B_vec, B_vec)
                        Jmag_norm = np.linalg.norm(J_vec)
                        
                        if Bmag2 > 1e-20 and Jmag_norm > 1e-20:
                            alpha_val = np.dot(J_vec, B_vec) / Bmag2
                            JcrossB = np.cross(J_vec, B_vec)
                            forcefree_val = np.linalg.norm(JcrossB) / (Jmag_norm * np.sqrt(Bmag2))
                            sigma_val = np.dot(J_vec, B_vec) / (Jmag_norm * np.sqrt(Bmag2) + 1e-12)
                        else:
                            alpha_val = 0.0
                            forcefree_val = 1.0
                            sigma_val = 0.0
                        
                        X_map[iy, ix] = xp
                        Y_map[iy, ix] = yp
                        R_map[iy, ix] = r_cyl
                        Phi_map[iy, ix] = phi
                        
                        Bx_local_map[iy, ix] = Bx_local
                        By_local_map[iy, ix] = By_local
                        Bz_local_map[iy, ix] = bz_comp
                        Bphi_map[iy, ix] = bphi
                        Br_map[iy, ix] = br
                        Bmag_map[iy, ix] = bmag
                        
                        Jx_local_map[iy, ix] = Jx_local
                        Jy_local_map[iy, ix] = Jy_local
                        Jz_local_map[iy, ix] = jz_comp
                        Jphi_map[iy, ix] = jphi
                        Jr_map[iy, ix] = jr
                        Jmag_map[iy, ix] = jmag
                        
                        alpha_map[iy, ix] = alpha_val
                        forcefree_map[iy, ix] = forcefree_val
                        sigma_map[iy, ix] = sigma_val
                        
                        if r_cyl > 0.01 and bmag > 0.01 * bmax_global:
                            r_vals.append(r_cyl)
                            br_vals.append(br)
                            bphi_vals.append(bphi)
                            bz_vals.append(bz_comp)
                            bmag_vals.append(bmag)
                            jr_vals.append(jr)
                            jphi_vals.append(jphi)
                            jz_vals.append(jz_comp)
                            jmag_vals.append(jmag)
                            alpha_vals.append(alpha_val)
                            forcefree_vals.append(forcefree_val)
                            sigma_vals.append(sigma_val)
                    
                    maps_group = region_group.create_group('maps')
                    
                    maps_group.create_dataset('X', data=X_map, compression='gzip', compression_opts=1)
                    maps_group.create_dataset('Y', data=Y_map, compression='gzip', compression_opts=1)
                    maps_group.create_dataset('R', data=R_map, compression='gzip', compression_opts=1)
                    maps_group.create_dataset('Phi', data=Phi_map, compression='gzip', compression_opts=1)
                    
                    maps_group.create_dataset('Bx_local', data=Bx_local_map, compression='gzip', compression_opts=1)
                    maps_group.create_dataset('By_local', data=By_local_map, compression='gzip', compression_opts=1)
                    maps_group.create_dataset('Bz_local', data=Bz_local_map, compression='gzip', compression_opts=1)
                    maps_group.create_dataset('Bphi', data=Bphi_map, compression='gzip', compression_opts=1)
                    maps_group.create_dataset('Br', data=Br_map, compression='gzip', compression_opts=1)
                    maps_group.create_dataset('Bmag', data=Bmag_map, compression='gzip', compression_opts=1)
                    
                    maps_group.create_dataset('Jx_local', data=Jx_local_map, compression='gzip', compression_opts=1)
                    maps_group.create_dataset('Jy_local', data=Jy_local_map, compression='gzip', compression_opts=1)
                    maps_group.create_dataset('Jz_local', data=Jz_local_map, compression='gzip', compression_opts=1)
                    maps_group.create_dataset('Jphi', data=Jphi_map, compression='gzip', compression_opts=1)
                    maps_group.create_dataset('Jr', data=Jr_map, compression='gzip', compression_opts=1)
                    maps_group.create_dataset('Jmag', data=Jmag_map, compression='gzip', compression_opts=1)
                    
                    maps_group.create_dataset('alpha', data=alpha_map, compression='gzip', compression_opts=1)
                    maps_group.create_dataset('forcefree', data=forcefree_map, compression='gzip', compression_opts=1)
                    maps_group.create_dataset('sigma', data=sigma_map, compression='gzip', compression_opts=1)
                    
                    if len(r_vals) > 0:
                        profile_dict = compute_profiles_from_maps(
                            np.array(r_vals), np.array(br_vals), np.array(bphi_vals),
                            np.array(bz_vals), np.array(bmag_vals), np.array(jr_vals),
                            np.array(jphi_vals), np.array(jz_vals), np.array(jmag_vals),
                            np.array(alpha_vals), np.array(forcefree_vals), np.array(sigma_vals),
                            N_BINS
                        )
                        
                        if profile_dict is not None:
                            profiles_group = region_group.create_group('profiles')
                            for key, value in profile_dict.items():
                                profiles_group.create_dataset(key, data=value, compression='gzip', compression_opts=1)
                            
                            valid_bins = profile_dict['n_per_bin'] > 5
                            
                            if np.any(valid_bins):
                                r_valid = profile_dict['r'][valid_bins]
                                bphi_valid = profile_dict['Bphi_mean'][valid_bins]
                                bmag_valid = profile_dict['Bmag_mean'][valid_bins]
                                
                                twist_fraction = np.abs(bphi_valid) / (bmag_valid + 1e-12)
                                idx_peak = np.argmax(twist_fraction)
                                peak_value = twist_fraction[idx_peak]
                                threshold = 0.1 * peak_value
                                after_peak = twist_fraction[idx_peak:]
                                below = np.where(after_peak < threshold)[0]
                                
                                if len(below) > 0:
                                    r_boundary = r_valid[idx_peak + below[0]]
                                else:
                                    r_boundary = r_valid[-1]
                                
                                valid_twist = valid_bins & np.isfinite(profile_dict['twist_mean'])
                                mean_twist_proxy = np.nanmean(profile_dict['twist_mean'][valid_twist]) if np.any(valid_twist) else np.nan
                                median_twist_proxy = np.nanmedian(profile_dict['twist_mean'][valid_twist]) if np.any(valid_twist) else np.nan
                                
                                mean_forcefree = np.nanmean(profile_dict['forcefree_mean'][valid_bins])
                                mean_alpha = np.nanmean(profile_dict['alpha_mean'][valid_bins])
                                mean_sigma = np.nanmean(profile_dict['sigma_mean'][valid_bins])
                                
                                diagnostics_group = region_group.create_group('diagnostics')
                                diagnostics_group.attrs['rope_radius'] = r_boundary
                                diagnostics_group.attrs['Bmax_global'] = bmax_global
                                diagnostics_group.attrs['mean_forcefree'] = mean_forcefree
                                diagnostics_group.attrs['mean_alpha'] = mean_alpha
                                diagnostics_group.attrs['mean_sigma'] = mean_sigma
                                diagnostics_group.attrs['mean_twist_proxy'] = mean_twist_proxy
                                diagnostics_group.attrs['median_twist_proxy'] = median_twist_proxy
                                
                                print(f"    r_boundary={r_boundary:.3f} Mm, mean_twist_proxy={mean_twist_proxy:.4f}")
                
                Delete(sampled)
                del data
                
                print(f"  Completed {region_name}")
            
            h5file.flush()
        
        print(f"\nDONE timestep {t_idx}")

    except Exception as e:
        print(f"\nERROR timestep {t_idx}: {e}")
        import traceback
        traceback.print_exc()

    finally:
        gc.collect()
