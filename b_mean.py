from paraview.simple import *
import numpy as np
import os

vtu_file = r"path_to_file\solar_bipolar_atmb0050.vtu"

print("\nLoading file...")
reader = XMLUnstructuredGridReader(FileName=[vtu_file])

view = GetActiveViewOrCreate('RenderView')
Show(reader, view)
Render()

info = reader.GetDataInformation()
print(f"\nFile info:")
print(f"  Cells: {info.GetNumberOfCells():,}")
print(f"  Points: {info.GetNumberOfPoints():,}")

bounds = info.GetBounds()
print(f"\nData bounds:")
print(f"  X: [{bounds[0]:.3f}, {bounds[1]:.3f}]")
print(f"  Y: [{bounds[2]:.3f}, {bounds[3]:.3f}]")
print(f"  Z: [{bounds[4]:.3f}, {bounds[5]:.3f}]")

cell_data_info = info.GetCellDataInformation()

print("\nAvailable cell data arrays:")
for i in range(cell_data_info.GetNumberOfArrays()):
    arr_info = cell_data_info.GetArrayInformation(i)
    print(f"  {arr_info.GetName()} ({arr_info.GetNumberOfComponents()} components)")

print("\n")

center_x = (bounds[0] + bounds[1]) / 2
center_y = (bounds[2] + bounds[3]) / 2
center_z = (bounds[4] + bounds[5]) / 2
center = [center_x, center_y, center_z]

size_x = bounds[1] - bounds[0]
size_y = bounds[3] - bounds[2]
size_z = bounds[5] - bounds[4]
radius = min(size_x, size_y, size_z) / 4

print(f"Data center: {center}")
print(f"Data sizes: X={size_x:.3f}, Y={size_y:.3f}, Z={size_z:.3f}")
print(f"Using sphere with center={center}, radius={radius:.3f}")

clip = Clip(Input=reader)
clip.ClipType = 'Sphere'
clip.ClipType.Center = center
clip.ClipType.Radius = radius
clip.Invert = 1
clip.Crinkleclip = 0

UpdatePipeline()

clip_info = clip.GetDataInformation()
print(f"\nAfter clipping:")
print(f"  Cells remaining: {clip_info.GetNumberOfCells():,}")
print(f"  Points remaining: {clip_info.GetNumberOfPoints():,}")

integrate = IntegrateVariables(Input=clip)
UpdatePipeline()

integrate_data = servermanager.Fetch(integrate)

if integrate_data and integrate_data.GetNumberOfCells() > 0:
    cell_data = integrate_data.GetCellData()

    print("\nIntegration results:")

    results = {}
    num_arrays = cell_data.GetNumberOfArrays()
    print(f"Number of data arrays: {num_arrays}")

    for i in range(num_arrays):
        arr = cell_data.GetArray(i)
        name = arr.GetName()
        num_tuples = arr.GetNumberOfTuples()
        num_components = arr.GetNumberOfComponents()

        if num_tuples > 0:
            if num_components == 1:
                value = arr.GetValue(0)
                results[name] = value
                print(f"{name:10}: {value:.6e}")
            else:
                for j in range(num_components):
                    comp_name = f"{name}[{j}]"
                    value = arr.GetComponent(0, j)
                    results[comp_name] = value
                    print(f"{comp_name:10}: {value:.6e}")

    volume = results.get('Volume', 0)

    if volume > 0:
        theoretical = 4.0 / 3.0 * np.pi * (radius ** 3)

        print(f"\nGeometry analysis:")
        print(f"  Sphere radius: {radius:.6f}")
        print(f"  Integrated volume: {volume:.6e}")
        print(f"  Theoretical sphere volume: {theoretical:.6e}")

        if theoretical > 0:
            diff = abs(volume - theoretical)
            rel_diff = diff / theoretical * 100
            print(f"  Absolute difference: {diff:.6e}")
            print(f"  Relative deviation:  {rel_diff:.2f}%")

        print(f"\nMagnetic field analysis:")

        b1 = results.get('b1', 0)
        b2 = results.get('b2', 0)
        b3 = results.get('b3', 0)

        print(f"  Integrals of magnetic field components:")
        print(f"    Int(b1) dV = {b1:.6e}")
        print(f"    Int(b2) dV = {b2:.6e}")
        print(f"    Int(b3) dV = {b3:.6e}")

        b1_avg = b1 / volume
        b2_avg = b2 / volume
        b3_avg = b3 / volume

        print(f"\n  Average magnetic field (∫B dV / ∫dV):")
        print(f"    <b1> = {b1_avg:.6e}")
        print(f"    <b2> = {b2_avg:.6e}")
        print(f"    <b3> = {b3_avg:.6e}")

        magnitude = np.sqrt(b1_avg ** 2 + b2_avg ** 2 + b3_avg ** 2)
        print(f"    |<B>| = {magnitude:.6e}")

        print(f"\nOther integrated quantities:")
        for name, value in results.items():
            if name not in ['b1', 'b2', 'b3', 'Volume']:
                print(f"  {name:10}: {value:.6e}")

        with open("paraview_results.txt", "w", encoding="utf-8") as f:
            f.write("Paraview results - sphere clip\n")
            f.write("\n\n")
            f.write(f"Input file: {vtu_file}\n")
            f.write(f"Data bounds:\n")
            f.write(f"  X: [{bounds[0]:.6f}, {bounds[1]:.6f}]\n")
            f.write(f"  Y: [{bounds[2]:.6f}, {bounds[3]:.6f}]\n")
            f.write(f"  Z: [{bounds[4]:.6f}, {bounds[5]:.6f}]\n\n")

            f.write(f"Clip parameters:\n")
            f.write(f"  Type: Sphere\n")
            f.write(f"  Center: {clip.ClipType.Center}\n")
            f.write(f"  Radius: {clip.ClipType.Radius}\n")
            f.write(f"  Invert: {clip.Invert} (keep inside)\n\n")

            f.write("Data statistics:\n")
            f.write(f"  Original cells: {info.GetNumberOfCells():,}\n")
            f.write(f"  After clipping: {clip_info.GetNumberOfCells():,}\n")
            if info.GetNumberOfCells() > 0:
                reduction = 100 * (1 - clip_info.GetNumberOfCells() / info.GetNumberOfCells())
                f.write(f"  Reduction: {reduction:.1f}%\n\n")

            f.write("Integration results:\n")
            for name, value in sorted(results.items()):
                f.write(f"  {name:15}: {value:.6e}\n")

            f.write(f"\nVolume analysis:\n")
            f.write(f"  Sphere radius: {radius:.6f}\n")
            f.write(f"  Integrated volume: {volume:.6e}\n")
            f.write(f"  Theoretical volume: {theoretical:.6e}\n")
            f.write(f"  Deviation: {rel_diff:.2f}%\n")

            f.write(f"\nAverage magnetic field:\n")
            f.write(f"  <b1> = {b1_avg:.6e}\n")
            f.write(f"  <b2> = {b2_avg:.6e}\n")
            f.write(f"  <b3> = {b3_avg:.6e}\n")
            f.write(f"  |<B>| = {magnitude:.6e}\n")

        print(f"\nResults saved to paraview_results.txt")
    else:
        print(f"\nError: Volume is zero")
        print("Available results:")
        for name, value in results.items():
            print(f"  {name}: {value:.6e}")

else:
    print("\nError: No integration data available")

