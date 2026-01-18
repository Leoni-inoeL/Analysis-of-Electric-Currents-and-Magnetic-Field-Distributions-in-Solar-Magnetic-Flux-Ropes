import pyvista as pv
import numpy as np


file_path = "file_path.vtu"  # insert path to your .vtu file here
mesh = pv.read(file_path)


print("\n")
print(f"   Grid type: {type(mesh).__name__}")
print(f"   n_points: {mesh.n_points}")
print(f"   n_cells: {mesh.n_cells}")
print(f"   n_arrays: {mesh.n_arrays}")


print("\n")
print("\n   ___Point data___")
if mesh.point_data:
    for i, name in enumerate(mesh.point_data.keys()):
        arr = mesh.point_data[name]
        print(f"   [{i}] '{name}': {arr.shape} {arr.dtype}")
else:
    print("   No point data")

print("\n   ___Cell data___")
if mesh.cell_data:
    for i, name in enumerate(mesh.cell_data.keys()):
        arr = mesh.cell_data[name]
        print(f"   [{i}] '{name}': {arr.shape} {arr.dtype}")
else:
    print("   N0 cell data")

print("\n   Field (global) data")
if mesh.field_data:
    for i, name in enumerate(mesh.field_data.keys()):
        arr = mesh.field_data[name]
        print(f"   [{i}] '{name}': {arr.shape} {arr.dtype}")
else:
    print("   No field data")


print("\n")
b_components = ['b1', 'b2', 'b3']

for b_name in b_components:
    print(f"\n '{b_name} component':")

    location = None
    if b_name in mesh.point_data:
        location = "POINT DATA"
        arr = mesh.point_data[b_name]
    elif b_name in mesh.cell_data:
        location = "CELL DATA"
        arr = mesh.cell_data[b_name]
    else:
        print(f"      No data")
        continue

    print(f"      location: {location}")
    print(f"      arr_shape: {arr.shape}")
    print(f"      arr_dtype: {arr.dtype}")
    print(f"      min: {arr.min():.4e}, Max: {arr.max():.4e}, Mean: {arr.mean():.4e}")


print("\n")
if 'TIME' in mesh.point_data:
    time_data = mesh.point_data['TIME']
    print(f"   no 'TIME' in point data")
    print(f"   time_data: {time_data[0] if len(time_data) > 0 else 'N/A'}")
    print(f"   identical content: {np.allclose(time_data, time_data[0])}")
elif 'TIME' in mesh.cell_data:
    time_data = mesh.cell_data['TIME']
    print(f"   no 'TIME' in cell data")
    print(f"   time_data: {time_data[0] if len(time_data) > 0 else 'N/A'}")
    print(f"   identical content: {np.allclose(time_data, time_data[0])}")
else:
    print("   No data")


print("\n")
if mesh.point_data:
    print(f"   Point Data arrays: {len(mesh.point_data)}")
    print(f"   names: {list(mesh.point_data.keys())}")
if mesh.cell_data:
    print(f"   Cell Data arrays: {len(mesh.cell_data)}")
    print(f"   names: {list(mesh.cell_data.keys())}")


if all(b in mesh.point_data for b in b_components):
    b_location = "POINT DATA"
elif all(b in mesh.cell_data for b in b_components):
    b_location = "CELL DATA"
else:
    b_location = None
    print("Something's wrong")


if b_location:
    print(f"\n(b1,b2,b3) is in {b_location}")
