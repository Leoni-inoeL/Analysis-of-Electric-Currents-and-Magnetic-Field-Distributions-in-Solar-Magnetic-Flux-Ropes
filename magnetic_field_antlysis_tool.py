from paraview.simple import *
import numpy as np
from paraview import vtk

source = GetActiveSource()

center = [1.5, 0.0, 0.2]
radius = 0.2

print(f"\nParameters:")
print(f"Center O: {center}")
print(f"Radius R: {radius}")

clip = Clip(Input=source)
clip.ClipType = 'Sphere'
clip.ClipType.Center = center
clip.ClipType.Radius = radius
clip.Invert = 1

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

    print(f"\nMean field results:")
    print(f"  B_mean: [{b1_mean:.6e}, {b2_mean:.6e}, {b3_mean:.6e}]")
    print(f"  |B_mean|: {np.linalg.norm(B_mean):.6e}")
    print(f"  B_norm (Z' axis): {B_norm}")

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

print(f"Local coordinate axes:")
print(f"  X' axis: {X_prime}")
print(f"  Y' axis: {Y_prime}")
print(f"  Z' axis: {B_norm}")

calc_mag = Calculator(Input=source)
calc_mag.AttributeType = 'Cell Data'
calc_mag.ResultArrayName = 'B_magnitude'
calc_mag.Function = 'sqrt(b1*b1 + b2*b2 + b3*b3)'

calc_vector = Calculator(Input=calc_mag)
calc_vector.AttributeType = 'Cell Data'
calc_vector.ResultArrayName = 'B_vector'
calc_vector.Function = 'b1*iHat + b2*jHat + b3*kHat'

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

slice1 = Slice(Input=calc_mag)
slice1.SliceType = 'Plane'
slice1.SliceType.Origin = center
slice1.SliceType.Normal = B_norm.tolist()

arrow_length_Z = radius * 8.0
arrow_length_XY = radius * 4.0

line_X = Line()
line_X.Point1 = center
line_X.Point2 = [center[0] + X_prime[0] * arrow_length_XY,
                 center[1] + X_prime[1] * arrow_length_XY,
                 center[2] + X_prime[2] * arrow_length_XY]

line_Y = Line()
line_Y.Point1 = center
line_Y.Point2 = [center[0] + Y_prime[0] * arrow_length_XY,
                 center[1] + Y_prime[1] * arrow_length_XY,
                 center[2] + Y_prime[2] * arrow_length_XY]

line_Z = Line()
line_Z.Point1 = center
line_Z.Point2 = [center[0] + B_norm[0] * arrow_length_Z,
                 center[1] + B_norm[1] * arrow_length_Z,
                 center[2] + B_norm[2] * arrow_length_Z]

view = GetActiveView()
if not view:
    view = CreateRenderView()

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

text_X = Text(registrationName='Text_X')
text_X.Text = "X'"
text_X_display = Show(text_X, view)
text_X_display.WindowLocation = 'Upper Left Corner'

text_Y = Text(registrationName='Text_Y')
text_Y.Text = "Y'"
text_Y_display = Show(text_Y, view)
text_Y_display.WindowLocation = 'Upper Left Corner'

text_Z = Text(registrationName='Text_Z')
text_Z.Text = "Z' (B_norm)"
text_Z_display = Show(text_Z, view)
text_Z_display.WindowLocation = 'Upper Left Corner'

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

print(f"  Red line (X'): length = {arrow_length_XY:.2f}")
print(f"  Green line (Y'): length = {arrow_length_XY:.2f}")
print(f"  Blue line (Z'): length = {arrow_length_Z:.2f}")


print(f"\nLocal axes at O={center}:")
print(f"X' axis (red): {X_prime}")
print(f"Y' axis (green): {Y_prime}")
print(f"Z' axis (blue): {B_norm}")

SaveScreenshot('magnetic_analysis_local.png', view, ImageResolution=[1600, 900])
SaveData('slice_data_local.csv', slice1)

with open('local_system_info.txt', 'w') as f:
    f.write("Local coordinate system info:\n")
    f.write(f"Analysis center: {center}\n")
    f.write(f"Sphere radius: {radius}\n")
    f.write(f"Axis length X': {arrow_length_XY}\n")
    f.write(f"Axis length Y': {arrow_length_XY}\n")
    f.write(f"Axis length Z': {arrow_length_Z}\n\n")
    f.write("Local axes:\n")
    f.write(f"  X' = {X_prime}\n")
    f.write(f"  Y' = {Y_prime}\n")
    f.write(f"  Z' = {B_norm}\n\n")
    f.write("Mean magnetic field:\n")
    f.write(f"  B_mean = [{b1_mean:.6e}, {b2_mean:.6e}, {b3_mean:.6e}]\n")
    f.write(f"  |B_mean| = {np.linalg.norm(B_mean):.6e}\n\n")


