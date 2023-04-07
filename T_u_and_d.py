# state file generated using paraview version 5.8.1

# ----------------------------------------------------------------
# setup views used in the visualization
# ----------------------------------------------------------------

# trace generated using paraview version 5.8.1
#
# To ensure correct image size when batch processing, please search 
# for and uncomment the line `# renderView*.ViewSize = [*,*]`

#### import the simple module from the paraview
from paraview.simple import *
import json
from types import SimpleNamespace
import numpy as np
import os

# import run metadata
outfile = r'fileloc'

# import simulation parameters
with open(outfile + '/' + outfile.split('/')[-1]  + '.json', 'r') as openfile:
    dictionary = json.load(openfile)

ins = SimpleNamespace(**dictionary)

if (type(ins.Tg) is float) or (type(ins.Tg) is int):
    Tg = ins.Tg
elif type(ins.Tg) is type(str):
    Tg = eval(ins.Tg.replace('vft', 'ins.vft'))
elif type(ins.Tg) is type(None):
    Tg = ins.vftc + ins.vftb/(12-ins.vfta) - 273
    
boundary = ['left','right','top','bottom']
max_u = np.nanmax(np.append(np.array([eval('ins.' + bound + '_ux') for bound in boundary],dtype=float),
                            np.array([eval('ins.' + bound + '_uy') for bound in boundary],dtype=float)))
min_u = np.nanmin(np.append(np.array([eval('ins.' + bound + '_ux') for bound in boundary],dtype=float),
                            np.array([eval('ins.' + bound + '_uy') for bound in boundary],dtype=float)))

#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# Create a new 'Line Chart View'
lineChartView1 = CreateView('XYChartView')
lineChartView1.ViewSize = [795, 268]
lineChartView1.LegendPosition = [466, 209]
lineChartView1.LeftAxisUseCustomRange = 1
lineChartView1.LeftAxisRangeMaximum = ins.T0
lineChartView1.BottomAxisUseCustomRange = 1
lineChartView1.BottomAxisRangeMaximum = ins.L_x
lineChartView1.RightAxisRangeMaximum = 6.66
lineChartView1.TopAxisRangeMaximum = 6.66

# Create a new 'Line Chart View'
lineChartView2 = CreateView('XYChartView')
lineChartView2.ViewSize = [795, 268]
lineChartView2.LegendPosition = [484, 226]
lineChartView2.LeftAxisLogScale = 1
lineChartView2.LeftAxisUseCustomRange = 1
lineChartView2.LeftAxisRangeMinimum = 1e-15
lineChartView2.LeftAxisRangeMaximum = max_u
lineChartView2.BottomAxisRangeMaximum = ins.L_x
lineChartView2.RightAxisRangeMaximum = 6.66
lineChartView2.TopAxisRangeMaximum = 6.66

# get the material library
materialLibrary1 = GetMaterialLibrary()

# Create a new 'Render View'
renderView1 = CreateView('RenderView')
renderView1.ViewSize = [795, 268]
renderView1.InteractionMode = '2D'
renderView1.AxesGrid = 'GridAxes3DActor'
renderView1.CenterOfRotation = [ins.L_x/2, ins.L_y/2, 0.0]
renderView1.StereoType = 'Crystal Eyes'
renderView1.CameraPosition = [ins.L_x/2, ins.L_y/2, 10000.0]
renderView1.CameraFocalPoint = [ins.L_x/2, ins.L_y/2, 0.0]
renderView1.CameraFocalDisk = 1.0
renderView1.CameraParallelScale = 0.3535533905932738
renderView1.BackEnd = 'OSPRay raycaster'
renderView1.OSPRayMaterialLibrary = materialLibrary1

# Create a new 'Render View'
renderView2 = CreateView('RenderView')
renderView2.ViewSize = [795, 268]
renderView2.InteractionMode = '2D'
renderView2.AxesGrid = 'GridAxes3DActor'
renderView2.CenterOfRotation = [ins.L_x/2, ins.L_y/2, 0.0]
renderView2.StereoType = 'Crystal Eyes'
renderView2.CameraPosition = [ins.L_x/2, ins.L_y/2, 10000.0]
renderView2.CameraFocalPoint = [ins.L_x/2, ins.L_y/2, 0.0]
renderView2.CameraFocalDisk = 1.0
renderView2.CameraParallelScale = 0.3535533905932738
renderView2.BackEnd = 'OSPRay raycaster'
renderView2.OSPRayMaterialLibrary = materialLibrary1

SetActiveView(None)

# ----------------------------------------------------------------
# setup view layouts
# ----------------------------------------------------------------

# create new layout object 'Layout #1'
layout1 = CreateLayout(name='Layout #1')
layout1.SplitHorizontal(0, 0.500000)
layout1.SplitVertical(1, 0.500000)
layout1.AssignView(3, renderView1)
layout1.AssignView(4, renderView2)
layout1.SplitVertical(2, 0.500000)
layout1.AssignView(5, lineChartView1)
layout1.AssignView(6, lineChartView2)

# ----------------------------------------------------------------
# restore active view
SetActiveView(lineChartView1)
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# setup the data processing pipelines
# ----------------------------------------------------------------
mu_files = []
P_files  = []
u_files  = []
d_files  = []
T_files  = []

for file in os.listdir(outfile):
    if 'vtk' in file:
        if '_mu_' in file:
            mu_files.append(outfile + '/' + file)
        if '_P_' in file:
            P_files.append(outfile + '/' + file)
        if '_u_' in file:
            u_files.append(outfile + '/' + file)
        if '_d_' in file:
            d_files.append(outfile + '/' + file)
        if '_T_' in file:
            T_files.append(outfile + '/' + file)

# create a new 'Legacy VTK Reader'
mu = LegacyVTKReader(FileNames=mu_files,registrationName='mu')
P = LegacyVTKReader(FileNames=P_files,registrationName='P')
u = LegacyVTKReader(FileNames=u_files,registrationName='u')
d = LegacyVTKReader(FileNames=d_files,registrationName='d')
T = LegacyVTKReader(FileNames=T_files,registrationName='T')

# create a new 'Contour'
contour1 = Contour(Input=T,registrationName='Tg')
contour1.ContourBy = ['POINTS', 'dataset1']
contour1.Isosurfaces = [Tg]
contour1.PointMergeMethod = 'Uniform Binning'

# create a new 'Plot Over Line'
plotOverLine2 = PlotOverLine(Input=u,
    Source='High Resolution Line Source',registrationName='u profile')

# init the 'High Resolution Line Source' selected for 'Source'
plotOverLine2.Source.Point1 = [0.0, ins.L_y/2, 0.0]
plotOverLine2.Source.Point2 = [ins.L_x, ins.L_y/2, 0.0]

# create a new 'Plot Over Line'
plotOverLine1 = PlotOverLine(Input=T,
    Source='High Resolution Line Source',registrationName='T profile')

# init the 'High Resolution Line Source' selected for 'Source'
plotOverLine1.Source.Point1 = [0.0, ins.L_y/2, 0.0]
plotOverLine1.Source.Point2 = [ins.L_x, ins.L_x/2, 0.0]

# create a new 'Plot Over Line'
plotOverLine3 = PlotOverLine(Input=d,
    Source='High Resolution Line Source',registrationName='d profile')

# init the 'High Resolution Line Source' selected for 'Source'
plotOverLine3.Source.Point1 = [0.0, ins.L_y/2, 0.0]
plotOverLine3.Source.Point2 = [ins.L_x, ins.L_y/2, 0.0]

# ----------------------------------------------------------------
# setup the visualization in view 'lineChartView1'
# ----------------------------------------------------------------

# show data from plotOverLine1
plotOverLine1Display = Show(plotOverLine1, lineChartView1, 'XYChartRepresentation')

# trace defaults for the display properties.
plotOverLine1Display.CompositeDataSetIndex = [0]
plotOverLine1Display.UseIndexForXAxis = 0
plotOverLine1Display.XArrayName = 'arc_length'
plotOverLine1Display.SeriesVisibility = ['dataset1']
plotOverLine1Display.SeriesLabel = ['arc_length', 'arc_length', 'dataset1', 'Temperature', 'vtkValidPointMask', 'vtkValidPointMask', 'Points_X', 'Points_X', 'Points_Y', 'Points_Y', 'Points_Z', 'Points_Z', 'Points_Magnitude', 'Points_Magnitude']
plotOverLine1Display.SeriesColor = ['arc_length', '0', '0', '0', 'dataset1', '0.889998', '0.100008', '0.110002', 'vtkValidPointMask', '0.220005', '0.489998', '0.719997', 'Points_X', '0.300008', '0.689998', '0.289998', 'Points_Y', '0.6', '0.310002', '0.639994', 'Points_Z', '1', '0.500008', '0', 'Points_Magnitude', '0.650004', '0.340002', '0.160006']
plotOverLine1Display.SeriesPlotCorner = ['Points_Magnitude', '0', 'Points_X', '0', 'Points_Y', '0', 'Points_Z', '0', 'arc_length', '0', 'dataset1', '0', 'vtkValidPointMask', '0']
plotOverLine1Display.SeriesLabelPrefix = ''
plotOverLine1Display.SeriesLineStyle = ['Points_Magnitude', '1', 'Points_X', '1', 'Points_Y', '1', 'Points_Z', '1', 'arc_length', '1', 'dataset1', '1', 'vtkValidPointMask', '1']
plotOverLine1Display.SeriesLineThickness = ['Points_Magnitude', '2', 'Points_X', '2', 'Points_Y', '2', 'Points_Z', '2', 'arc_length', '2', 'dataset1', '2', 'vtkValidPointMask', '2']
plotOverLine1Display.SeriesMarkerStyle = ['Points_Magnitude', '0', 'Points_X', '0', 'Points_Y', '0', 'Points_Z', '0', 'arc_length', '0', 'dataset1', '0', 'vtkValidPointMask', '0']
plotOverLine1Display.SeriesMarkerSize = ['Points_Magnitude', '4', 'Points_X', '4', 'Points_Y', '4', 'Points_Z', '4', 'arc_length', '4', 'dataset1', '4', 'vtkValidPointMask', '4']

# show data from contour1
contour1Display = Show(contour1, lineChartView1, 'XYChartRepresentation')

# trace defaults for the display properties.
contour1Display.CompositeDataSetIndex = [0]
contour1Display.XArrayName = 'dataset1'
contour1Display.SeriesVisibility = ['dataset1']
contour1Display.SeriesLabel = ['dataset1', 'Tg', 'Points_X', 'Points_X', 'Points_Y', 'Points_Y', 'Points_Z', 'Points_Z', 'Points_Magnitude', 'Points_Magnitude']
contour1Display.SeriesColor = ['dataset1', '0', '0', '0', 'Points_X', '0.889998', '0.100008', '0.110002', 'Points_Y', '0.220005', '0.489998', '0.719997', 'Points_Z', '0.300008', '0.689998', '0.289998', 'Points_Magnitude', '0.6', '0.310002', '0.639994']
contour1Display.SeriesPlotCorner = ['Points_Magnitude', '0', 'Points_X', '0', 'Points_Y', '0', 'Points_Z', '0', 'dataset1', '0']
contour1Display.SeriesLabelPrefix = ''
contour1Display.SeriesLineStyle = ['Points_Magnitude', '1', 'Points_X', '1', 'Points_Y', '1', 'Points_Z', '1', 'dataset1', '1']
contour1Display.SeriesLineThickness = ['Points_Magnitude', '2', 'Points_X', '2', 'Points_Y', '2', 'Points_Z', '2', 'dataset1', '2']
contour1Display.SeriesMarkerStyle = ['Points_Magnitude', '0', 'Points_X', '0', 'Points_Y', '0', 'Points_Z', '0', 'dataset1', '0']
contour1Display.SeriesMarkerSize = ['Points_Magnitude', '4', 'Points_X', '4', 'Points_Y', '4', 'Points_Z', '4', 'dataset1', '4']

# ----------------------------------------------------------------
# setup the visualization in view 'lineChartView2'
# ----------------------------------------------------------------

# show data from plotOverLine2
plotOverLine2Display = Show(plotOverLine2, lineChartView2, 'XYChartRepresentation')

# trace defaults for the display properties.
plotOverLine2Display.CompositeDataSetIndex = [0]
plotOverLine2Display.UseIndexForXAxis = 0
plotOverLine2Display.XArrayName = 'arc_length'
plotOverLine2Display.SeriesVisibility = ['dataset1_Magnitude']
plotOverLine2Display.SeriesLabel = ['arc_length', 'arc_length', 'dataset1_X', 'dataset1_X', 'dataset1_Y', 'dataset1_Y', 'dataset1_Z', 'dataset1_Z', 'dataset1_Magnitude', 'Velocity Y', 'vtkValidPointMask', 'vtkValidPointMask', 'Points_X', 'Points_X', 'Points_Y', 'Points_Y', 'Points_Z', 'Points_Z', 'Points_Magnitude', 'Points_Magnitude']
plotOverLine2Display.SeriesColor = ['arc_length', '0', '0', '0', 'dataset1_X', '0.889998', '0.100008', '0.110002', 'dataset1_Y', '0.220005', '0.489998', '0.719997', 'dataset1_Z', '0.300008', '0.689998', '0.289998', 'dataset1_Magnitude', '0.6', '0.310002', '0.639994', 'vtkValidPointMask', '1', '0.500008', '0', 'Points_X', '0.650004', '0.340002', '0.160006', 'Points_Y', '0', '0', '0', 'Points_Z', '0.889998', '0.100008', '0.110002', 'Points_Magnitude', '0.220005', '0.489998', '0.719997']
plotOverLine2Display.SeriesPlotCorner = ['Points_Magnitude', '0', 'Points_X', '0', 'Points_Y', '0', 'Points_Z', '0', 'arc_length', '0', 'dataset1_Magnitude', '0', 'dataset1_X', '0', 'dataset1_Y', '0', 'dataset1_Z', '0', 'vtkValidPointMask', '0']
plotOverLine2Display.SeriesLabelPrefix = ''
plotOverLine2Display.SeriesLineStyle = ['Points_Magnitude', '1', 'Points_X', '1', 'Points_Y', '1', 'Points_Z', '1', 'arc_length', '1', 'dataset1_Magnitude', '1', 'dataset1_X', '1', 'dataset1_Y', '1', 'dataset1_Z', '1', 'vtkValidPointMask', '1']
plotOverLine2Display.SeriesLineThickness = ['Points_Magnitude', '2', 'Points_X', '2', 'Points_Y', '2', 'Points_Z', '2', 'arc_length', '2', 'dataset1_Magnitude', '2', 'dataset1_X', '2', 'dataset1_Y', '2', 'dataset1_Z', '2', 'vtkValidPointMask', '2']
plotOverLine2Display.SeriesMarkerStyle = ['Points_Magnitude', '0', 'Points_X', '0', 'Points_Y', '0', 'Points_Z', '0', 'arc_length', '0', 'dataset1_Magnitude', '0', 'dataset1_X', '0', 'dataset1_Y', '0', 'dataset1_Z', '0', 'vtkValidPointMask', '0']
plotOverLine2Display.SeriesMarkerSize = ['Points_Magnitude', '4', 'Points_X', '4', 'Points_Y', '4', 'Points_Z', '4', 'arc_length', '4', 'dataset1_Magnitude', '4', 'dataset1_X', '4', 'dataset1_Y', '4', 'dataset1_Z', '4', 'vtkValidPointMask', '4']

# show data from plotOverLine3
plotOverLine3Display = Show(plotOverLine3, lineChartView2, 'XYChartRepresentation')

# trace defaults for the display properties.
plotOverLine3Display.CompositeDataSetIndex = [0]
plotOverLine3Display.UseIndexForXAxis = 0
plotOverLine3Display.XArrayName = 'arc_length'
plotOverLine3Display.SeriesVisibility = ['dataset1_Magnitude']
plotOverLine3Display.SeriesLabel = ['arc_length', 'arc_length', 'dataset1_X', 'dataset1_X', 'dataset1_Y', 'dataset1_Y', 'dataset1_Z', 'dataset1_Z', 'dataset1_Magnitude', 'Displacement Y', 'vtkValidPointMask', 'vtkValidPointMask', 'Points_X', 'Points_X', 'Points_Y', 'Points_Y', 'Points_Z', 'Points_Z', 'Points_Magnitude', 'Points_Magnitude']
plotOverLine3Display.SeriesColor = ['arc_length', '0', '0', '0', 'dataset1_X', '0.889998', '0.100008', '0.110002', 'dataset1_Y', '0.220005', '0.489998', '0.719997', 'dataset1_Z', '0.300008', '0.689998', '0.289998', 'dataset1_Magnitude', '0', '0.666667', '0', 'vtkValidPointMask', '1', '0.500008', '0', 'Points_X', '0.650004', '0.340002', '0.160006', 'Points_Y', '0', '0', '0', 'Points_Z', '0.889998', '0.100008', '0.110002', 'Points_Magnitude', '0.220005', '0.489998', '0.719997']
plotOverLine3Display.SeriesPlotCorner = ['Points_Magnitude', '0', 'Points_X', '0', 'Points_Y', '0', 'Points_Z', '0', 'arc_length', '0', 'dataset1_Magnitude', '0', 'dataset1_X', '0', 'dataset1_Y', '0', 'dataset1_Z', '0', 'vtkValidPointMask', '0']
plotOverLine3Display.SeriesLabelPrefix = ''
plotOverLine3Display.SeriesLineStyle = ['Points_Magnitude', '1', 'Points_X', '1', 'Points_Y', '1', 'Points_Z', '1', 'arc_length', '1', 'dataset1_Magnitude', '1', 'dataset1_X', '1', 'dataset1_Y', '1', 'dataset1_Z', '1', 'vtkValidPointMask', '1']
plotOverLine3Display.SeriesLineThickness = ['Points_Magnitude', '2', 'Points_X', '2', 'Points_Y', '2', 'Points_Z', '2', 'arc_length', '2', 'dataset1_Magnitude', '2', 'dataset1_X', '2', 'dataset1_Y', '2', 'dataset1_Z', '2', 'vtkValidPointMask', '2']
plotOverLine3Display.SeriesMarkerStyle = ['Points_Magnitude', '0', 'Points_X', '0', 'Points_Y', '0', 'Points_Z', '0', 'arc_length', '0', 'dataset1_Magnitude', '0', 'dataset1_X', '0', 'dataset1_Y', '0', 'dataset1_Z', '0', 'vtkValidPointMask', '0']
plotOverLine3Display.SeriesMarkerSize = ['Points_Magnitude', '4', 'Points_X', '4', 'Points_Y', '4', 'Points_Z', '4', 'arc_length', '4', 'dataset1_Magnitude', '4', 'dataset1_X', '4', 'dataset1_Y', '4', 'dataset1_Z', '4', 'vtkValidPointMask', '4']

# ----------------------------------------------------------------
# setup the visualization in view 'renderView1'
# ----------------------------------------------------------------

# show data from melting_solid_T_0
T_Display = Show(T, renderView1, 'UnstructuredGridRepresentation')

# get color transfer function/color map for 'dataset1'
dataset1LUT = GetColorTransferFunction('dataset1')
dataset1LUT.AutomaticRescaleRangeMode = 'Never'
dataset1LUT.RGBPoints = [0, 0.231373, 0.298039, 0.752941, ins.T0/2, 0.865003, 0.865003, 0.865003, ins.T0, 0.705882, 0.0156863, 0.14902]
dataset1LUT.ScalarRangeInitialized = 1.0
dataset1LUT.VectorComponent = 1
dataset1LUT.VectorMode = 'Component'

# get opacity transfer function/opacity map for 'dataset1'
dataset1PWF = GetOpacityTransferFunction('dataset1')
dataset1PWF.Points = [0, 0.0, 0.5, 0.0, ins.T0, 1.0, 0.5, 0.0]
dataset1PWF.ScalarRangeInitialized = 1

# trace defaults for the display properties.
T_Display.Representation = 'Surface'
T_Display.ColorArrayName = ['POINTS', 'dataset1']
T_Display.LookupTable = dataset1LUT
T_Display.OSPRayScaleArray = 'dataset1'
T_Display.OSPRayScaleFunction = 'PiecewiseFunction'
T_Display.SelectOrientationVectors = 'None'
T_Display.ScaleFactor = 0.05
T_Display.SelectScaleArray = 'dataset1'
T_Display.GlyphType = 'Arrow'
T_Display.GlyphTableIndexArray = 'dataset1'
T_Display.GaussianRadius = 0.0025
T_Display.SetScaleArray = ['POINTS', 'dataset1']
T_Display.ScaleTransferFunction = 'PiecewiseFunction'
T_Display.OpacityArray = ['POINTS', 'dataset1']
T_Display.OpacityTransferFunction = 'PiecewiseFunction'
T_Display.DataAxesGrid = 'GridAxesRepresentation'
T_Display.PolarAxes = 'PolarAxesRepresentation'
T_Display.ScalarOpacityFunction = dataset1PWF
T_Display.ScalarOpacityUnitDistance = 0.07323817252277477

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
T_Display.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, ins.T0, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
T_Display.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, ins.T0, 1.0, 0.5, 0.0]

# show data from plotOverLine1
plotOverLine1Display_1 = Show(plotOverLine1, renderView1, 'GeometryRepresentation')

# trace defaults for the display properties.
plotOverLine1Display_1.Representation = 'Surface'
plotOverLine1Display_1.ColorArrayName = ['POINTS', 'dataset1']
plotOverLine1Display_1.LookupTable = dataset1LUT
plotOverLine1Display_1.OSPRayScaleArray = 'dataset1'
plotOverLine1Display_1.OSPRayScaleFunction = 'PiecewiseFunction'
plotOverLine1Display_1.SelectOrientationVectors = 'None'
plotOverLine1Display_1.ScaleFactor = 0.05
plotOverLine1Display_1.SelectScaleArray = 'dataset1'
plotOverLine1Display_1.GlyphType = 'Arrow'
plotOverLine1Display_1.GlyphTableIndexArray = 'dataset1'
plotOverLine1Display_1.GaussianRadius = 0.0025
plotOverLine1Display_1.SetScaleArray = ['POINTS', 'dataset1']
plotOverLine1Display_1.ScaleTransferFunction = 'PiecewiseFunction'
plotOverLine1Display_1.OpacityArray = ['POINTS', 'dataset1']
plotOverLine1Display_1.OpacityTransferFunction = 'PiecewiseFunction'
plotOverLine1Display_1.DataAxesGrid = 'GridAxesRepresentation'
plotOverLine1Display_1.PolarAxes = 'PolarAxesRepresentation'

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
plotOverLine1Display_1.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, ins.T0, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
plotOverLine1Display_1.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, ins.T0, 1.0, 0.5, 0.0]

# show data from contour1
contour1Display_1 = Show(contour1, renderView1, 'GeometryRepresentation')

# trace defaults for the display properties.
contour1Display_1.Representation = 'Surface'
contour1Display_1.ColorArrayName = ['POINTS', 'dataset1']
contour1Display_1.LookupTable = dataset1LUT
contour1Display_1.OSPRayScaleArray = 'dataset1'
contour1Display_1.OSPRayScaleFunction = 'PiecewiseFunction'
contour1Display_1.SelectOrientationVectors = 'None'
contour1Display_1.ScaleFactor = 0.05
contour1Display_1.SelectScaleArray = 'dataset1'
contour1Display_1.GlyphType = 'Arrow'
contour1Display_1.GlyphTableIndexArray = 'dataset1'
contour1Display_1.GaussianRadius = 0.0025
contour1Display_1.SetScaleArray = ['POINTS', 'dataset1']
contour1Display_1.ScaleTransferFunction = 'PiecewiseFunction'
contour1Display_1.OpacityArray = ['POINTS', 'dataset1']
contour1Display_1.OpacityTransferFunction = 'PiecewiseFunction'
contour1Display_1.DataAxesGrid = 'GridAxesRepresentation'
contour1Display_1.PolarAxes = 'PolarAxesRepresentation'

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
contour1Display_1.ScaleTransferFunction.Points = [ins.Tg, 0.0, 0.5, 0.0, ins.Tg+0.01, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
contour1Display_1.OpacityTransferFunction.Points = [ins.Tg, 0.0, 0.5, 0.0, ins.Tg+0.01, 1.0, 0.5, 0.0]

# setup the color legend parameters for each legend in this view

# get color legend/bar for dataset1LUT in view renderView1
dataset1LUTColorBar = GetScalarBar(dataset1LUT, renderView1)
dataset1LUTColorBar.Title = 'dataset1'
dataset1LUTColorBar.ComponentTitle = 'Y'

# set color bar visibility
dataset1LUTColorBar.Visibility = 1

# show color legend
T_Display.SetScalarBarVisibility(renderView1, True)

# show color legend
plotOverLine1Display_1.SetScalarBarVisibility(renderView1, True)

# show color legend
contour1Display_1.SetScalarBarVisibility(renderView1, True)

# ----------------------------------------------------------------
# setup the visualization in view 'renderView2'
# ----------------------------------------------------------------

# show data from melting_solid_u_0
u_Display = Show(u, renderView2, 'UnstructuredGridRepresentation')

# get separate color transfer function/color map for 'dataset1'
separate_u_Display_dataset1LUT = GetColorTransferFunction('dataset1', u_Display, separate=True)
separate_u_Display_dataset1LUT.RGBPoints = [min_u, 0.231373, 0.298039, 0.752941, (min_u + max_u)/2, 0.865003, 0.865003, 0.865003, max_u, 0.705882, 0.0156863, 0.14902]
separate_u_Display_dataset1LUT.ScalarRangeInitialized = 1.0

# get separate opacity transfer function/opacity map for 'dataset1'
separate_u_Display_dataset1PWF = GetOpacityTransferFunction('dataset1', u_Display, separate=True)
separate_u_Display_dataset1PWF.Points = [min_u, 0.0, 0.5, 0.0, max_u, 1.0, 0.5, 0.0]
separate_u_Display_dataset1PWF.ScalarRangeInitialized = 1

# trace defaults for the display properties.
u_Display.Representation = 'Surface'
u_Display.ColorArrayName = ['POINTS', 'dataset1']
u_Display.LookupTable = separate_u_Display_dataset1LUT
u_Display.OSPRayScaleArray = 'dataset1'
u_Display.OSPRayScaleFunction = 'PiecewiseFunction'
u_Display.SelectOrientationVectors = 'dataset1'
u_Display.ScaleFactor = 0.05
u_Display.SelectScaleArray = 'None'
u_Display.GlyphType = 'Arrow'
u_Display.GlyphTableIndexArray = 'None'
u_Display.GaussianRadius = 0.0025
u_Display.SetScaleArray = ['POINTS', 'dataset1']
u_Display.ScaleTransferFunction = 'PiecewiseFunction'
u_Display.OpacityArray = ['POINTS', 'dataset1']
u_Display.OpacityTransferFunction = 'PiecewiseFunction'
u_Display.DataAxesGrid = 'GridAxesRepresentation'
u_Display.PolarAxes = 'PolarAxesRepresentation'
u_Display.ScalarOpacityFunction = separate_u_Display_dataset1PWF
u_Display.ScalarOpacityUnitDistance = 0.07323817252277477

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
u_Display.ScaleTransferFunction.Points = [min_u, 0.0, 0.5, 0.0, max_u, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
u_Display.OpacityTransferFunction.Points = [min_u, 0.0, 0.5, 0.0, max_u, 1.0, 0.5, 0.0]

# set separate color map
u_Display.UseSeparateColorMap = True

# show data from plotOverLine2
plotOverLine2Display_1 = Show(plotOverLine2, renderView2, 'GeometryRepresentation')

# trace defaults for the display properties.
plotOverLine2Display_1.Representation = 'Surface'
plotOverLine2Display_1.ColorArrayName = ['POINTS', 'dataset1']
plotOverLine2Display_1.LookupTable = dataset1LUT
plotOverLine2Display_1.OSPRayScaleArray = 'arc_length'
plotOverLine2Display_1.OSPRayScaleFunction = 'PiecewiseFunction'
plotOverLine2Display_1.SelectOrientationVectors = 'dataset1'
plotOverLine2Display_1.ScaleFactor = 0.05
plotOverLine2Display_1.SelectScaleArray = 'None'
plotOverLine2Display_1.GlyphType = 'Arrow'
plotOverLine2Display_1.GlyphTableIndexArray = 'None'
plotOverLine2Display_1.GaussianRadius = 0.0025
plotOverLine2Display_1.SetScaleArray = ['POINTS', 'arc_length']
plotOverLine2Display_1.ScaleTransferFunction = 'PiecewiseFunction'
plotOverLine2Display_1.OpacityArray = ['POINTS', 'arc_length']
plotOverLine2Display_1.OpacityTransferFunction = 'PiecewiseFunction'
plotOverLine2Display_1.DataAxesGrid = 'GridAxesRepresentation'
plotOverLine2Display_1.PolarAxes = 'PolarAxesRepresentation'

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
plotOverLine2Display_1.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.5, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
plotOverLine2Display_1.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.5, 1.0, 0.5, 0.0]

# setup the color legend parameters for each legend in this view

# get color legend/bar for separate_u_Display_dataset1LUT in view renderView2
separate_u_Display_dataset1LUTColorBar = GetScalarBar(separate_u_Display_dataset1LUT, renderView2)
separate_u_Display_dataset1LUTColorBar.Title = 'dataset1'
separate_u_Display_dataset1LUTColorBar.ComponentTitle = 'Magnitude'

# set color bar visibility
separate_u_Display_dataset1LUTColorBar.Visibility = 1

# show color legend
u_Display.SetScalarBarVisibility(renderView2, True)

# ----------------------------------------------------------------
# setup color maps and opacity mapes used in the visualization
# note: the Get..() functions create a new object, if needed
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# finally, restore active source
SetActiveSource(plotOverLine3)
# ----------------------------------------------------------------