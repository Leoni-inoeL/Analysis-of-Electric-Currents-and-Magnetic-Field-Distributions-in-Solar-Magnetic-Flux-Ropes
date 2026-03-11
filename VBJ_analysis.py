from paraview.simple import *
import paraview


source = GetActiveSource()

if source is None:
    raise RuntimeError("No active source found. Please load the file first.")

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

# B_RFP_TRACER <- B_Gs
stream_tracer_B1 = StreamTracer(Input=calculator_B)
stream_tracer_B1.SeedType = 'Point Cloud'
stream_tracer_B1.SeedType.Center = [-1.5, 0.0, 0.3]
stream_tracer_B1.SeedType.Radius = 0.3
stream_tracer_B1.SeedType.NumberOfPoints = 25
stream_tracer_B1.MaximumStreamlineLength = 12.0
stream_tracer_B1.Vectors = ['POINTS', 'B']
RenameSource('B_RFP_TRACER', stream_tracer_B1)
stream_tracer_B1.UpdatePipeline()

# Tube3 <- B_RFP_TRACER <- B_Gs
tube_B1 = Tube(Input=stream_tracer_B1)
tube_B1.Scalars = ['POINTS', 'B']
tube_B1.Radius = 0.1
tube_B1.NumberofSides = 6
RenameSource('Tube3', tube_B1)
tube_B1.UpdatePipeline()

#  B_LFP_tracer <- B_Gs
stream_tracer_B2 = StreamTracer(Input=calculator_B)
stream_tracer_B2.SeedType = 'Point Cloud'
stream_tracer_B2.SeedType.Center = [-1.5, 0.0, 0.3]
stream_tracer_B2.SeedType.Radius = 0.3
stream_tracer_B2.SeedType.NumberOfPoints = 25
stream_tracer_B2.MaximumStreamlineLength = 24.0
stream_tracer_B2.Vectors = ['POINTS', 'B']
RenameSource('B_LFP_tracer', stream_tracer_B2)
stream_tracer_B2.UpdatePipeline()

# Jz_slice_z=0.1Mm <- J_A/km^2
slice_filter = Slice(Input=calculator_J)
slice_filter.SliceType = 'Plane'
slice_filter.SliceType.Origin = [0.0, 0.0, 0.01]  # 0.01 = 0.1Mm
slice_filter.SliceType.Normal = [0.0, 0.0, 1.0]
slice_filter.Triangulatetheslice = 0
RenameSource('Jz_slice_z=0.1Mm', slice_filter)
slice_filter.UpdatePipeline()

calculator_Jz_slice = Calculator(Input=slice_filter)
calculator_Jz_slice.ResultArrayName = 'Jz'
calculator_Jz_slice.Function = 'jHat*j'
RenameSource('Jz_on_slice', calculator_Jz_slice)
calculator_Jz_slice.UpdatePipeline()

# J_RFP_tracer <- J_A/km^2
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

# Tube1 <- J_RFP_tracer
tube_J1 = Tube(Input=stream_tracer_J1)
tube_J1.Scalars = ['POINTS', 'j']
tube_J1.Radius = 0.1
tube_J1.NumberofSides = 6
RenameSource('Tube1', tube_J1)
tube_J1.UpdatePipeline()

# 3. J_LFP_tracer <- J_A/km^2
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

# Tube4 <- J_LFP_tracer <- J_A/km^2
tube_J2 = Tube(Input=stream_tracer_J2)
tube_J2.Scalars = ['POINTS', 'j']
tube_J2.Radius = 0.1
tube_J2.NumberofSides = 6
RenameSource('Tube4', tube_J2)
tube_J2.UpdatePipeline()

# J_Below_z=4Mm <- J_A/km^2
stream_tracer_J3 = StreamTracer(Input=calculator_J)
stream_tracer_J3.SeedType = 'Point Cloud'
stream_tracer_J3.SeedType.Center = [0.0, 0.0, 0.4]  # 0.4 = 4Mm
stream_tracer_J3.SeedType.Radius = 0.1
stream_tracer_J3.SeedType.NumberOfPoints = 50
stream_tracer_J3.MaximumStreamlineLength = 64.0
stream_tracer_J3.Vectors = ['POINTS', 'j']
stream_tracer_J3.IntegrationDirection = 'BOTH'
stream_tracer_J3.IntegratorType = 'Runge-Kutta 4-5'
RenameSource('J_Below_z=4Mm', stream_tracer_J3)
stream_tracer_J3.UpdatePipeline()

# Tube5 <- J_Below_z=4Mm <- J_A/km^2
tube_J3 = Tube(Input=stream_tracer_J3)
tube_J3.Scalars = ['POINTS', 'j']
tube_J3.Radius = 0.1
tube_J3.NumberofSides = 6
RenameSource('Tube5', tube_J3)
tube_J3.UpdatePipeline()

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
    slice_display.MeshVisibility = 1
    try:
        slice_display.ShowPlane = 1
    except:
        pass

for tube_filter in [tube_J1, tube_J2, tube_J3, tube_B1]:
    display = GetDisplayProperties(tube_filter)
    if display:
        display.Representation = 'Surface'

Render()
