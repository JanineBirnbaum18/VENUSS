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
with open(outfile + '/' + outfile.split('/')[-1] +  '.json', 'r') as openfile:
    dictionary = json.load(openfile)

ins = SimpleNamespace(**dictionary)

if ins.temp:
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

# get the material library
materialLibrary1 = GetMaterialLibrary()

# Create a new 'Render View'
renderView1 = CreateView('RenderView')
renderView2 = CreateView('RenderView')
renderView3 = CreateView('RenderView')
renderView4 = CreateView('RenderView')

if ins.temp:
    renderViews = [renderView1, renderView2, renderView3, renderView4]
else:
    renderViews = [renderView1, renderView2]

for renderView in renderViews:
    renderView.ViewSize = [795, 268]
    renderView.InteractionMode = '2D'
    renderView.AxesGrid = 'GridAxes3DActor'
    renderView.CenterOfRotation = [ins.L_x/2, ins.L_y/2, 0.0]
    renderView.StereoType = 'Crystal Eyes'
    renderView.CameraPosition = [ins.L_x/2, ins.L_y/2, 10000.0]
    renderView.CameraFocalPoint = [ins.L_x/2, ins.L_y/2, 0.0]
    renderView.CameraFocalDisk = 1.0
    renderView.CameraParallelScale = 0.3535533905932738
    renderView.BackEnd = 'OSPRay raycaster'
    renderView.OSPRayMaterialLibrary = materialLibrary1

# ----------------------------------------------------------------
# setup view layouts
# ----------------------------------------------------------------

# create new layout object 'Layout #1'
if ins.temp:
    layout1 = CreateLayout(name='Layout #1')
    layout1.SplitHorizontal(0, 0.500000)
    layout1.SplitVertical(1, 0.500000)
    layout1.SplitVertical(2, 0.500000)
    layout1.AssignView(3, renderView1)
    layout1.AssignView(5, renderView2)
    layout1.AssignView(4, renderView3)
    layout1.AssignView(6, renderView4)
else:
    layout1 = CreateLayout(name='Layout #1')
    layout1.SplitVertical(1, 0.500000)
    layout1.AssignView(1, renderView1)
    layout1.AssignView(2, renderView2)

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
P = LegacyVTKReader(FileNames=P_files,registrationName='P')
u = LegacyVTKReader(FileNames=u_files,registrationName='u')
if ins.temp:
    mu = LegacyVTKReader(FileNames=mu_files,registrationName='mu')
    T = LegacyVTKReader(FileNames=T_files, registrationName='T')
    if ins.solidification:
        d = LegacyVTKReader(FileNames=d_files, registrationName='d')

# create a new 'Contour'
if ins.temp & ins.solidification:
    contour1 = Contour(Input=T,registrationName='Tg')
    contour1.ContourBy = ['POINTS', 'dataset1']
    contour1.Isosurfaces = [Tg]
    contour1.PointMergeMethod = 'Uniform Binning'

# ----------------------------------------------------------------
# setup the visualization in view 'renderView1'
# ----------------------------------------------------------------

def set_renderView(renderView,field,min_val,max_val,i=1,component='Magnitude',title=''):
    # show data
    component_dict = {'Magnitude':-1,'X':0,'Y':1,'Z':2}
    Display = Show(field, renderView, 'UnstructuredGridRepresentation')

    dataset1LUT = GetColorTransferFunction('dataset1',Display,separate=True)
    dataset1LUT.AutomaticRescaleRangeMode = 'Never'
    dataset1LUT.RGBPoints = [min_val, 0.231373, 0.298039, 0.752941,
                             (min_val + max_val)/2, 0.865003, 0.865003, 0.865003,
                             max_val, 0.705882, 0.0156863, 0.14902]
    dataset1LUT.ScalarRangeInitialized = 1.0
    dataset1LUT.VectorComponent = component_dict[component]
    dataset1LUT.VectorMode = 'Component'
    #Display.ScaleTransferFunction.Points = [min_val, 0.0, 0.5, 0.0, max_val, 1.0, 0.5, 0.0]

    # get opacity transfer function/opacity map for 'dataset1'
    dataset1PWF = GetOpacityTransferFunction('dataset1', Display, separate=True)
    dataset1PWF.Points = [min_val, 0.0, 0.5, 0.0, max_val, 1.0, 0.5, 0.0]
    dataset1PWF.ScalarRangeInitialized = 1
    #Display.OpacityTransferFunction.Points = [min_val, 0.0, 0.5, 0.0, max_val, 1.0, 0.5, 0.0]

    # trace defaults for the display properties.
    Display.Representation = 'Surface'
    Display.ColorArrayName = ['POINTS', 'dataset1']
    Display.LookupTable = dataset1LUT
    Display.OSPRayScaleArray = 'dataset1'
    Display.OSPRayScaleFunction = 'PiecewiseFunction'
    Display.SelectOrientationVectors = 'None'
    Display.ScaleFactor = 0.05
    Display.SelectScaleArray = 'None'
    Display.GlyphType = 'Arrow'
    Display.GlyphTableIndexArray = 'None'
    Display.GaussianRadius = 0.0025
    Display.SetScaleArray = ['POINTS', 'dataset1']
    Display.ScaleTransferFunction = 'PiecewiseFunction'
    Display.OpacityArray = ['POINTS', 'dataset1']
    Display.OpacityTransferFunction = 'PiecewiseFunction'
    Display.DataAxesGrid = 'GridAxesRepresentation'
    Display.PolarAxes = 'PolarAxesRepresentation'
    Display.UseSeparateColorMap = True
    Display.ScalarOpacityFunction = dataset1PWF
    Display.ScalarOpacityUnitDistance = 0.07323817252277477

    # get color legend/bar for dataset1LUT in view renderView1
    datasetLUTColorBar = GetScalarBar(dataset1LUT, renderView)
    datasetLUTColorBar.Title = title
    datasetLUTColorBar.ComponentTitle = component
    datasetLUTColorBar.Visibility = 1
    Display.SetScalarBarVisibility(renderView, True)

    return Display

def set_contour(renderView,contour,min_val,max_val):

    # show data from contour1
    contourDisplay = Show(contour, renderView, dataset1LUT, 'GeometryRepresentation')

    # trace defaults for the display properties.
    contourDisplay.Representation = 'Surface'
    contourDisplay.ColorArrayName = ['POINTS', 'dataset1']
    contourDisplay.LookupTable = dataset1LUT
    contourDisplay.OSPRayScaleArray = 'dataset1'
    contourDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
    contourDisplay.SelectOrientationVectors = 'None'
    contourDisplay.ScaleFactor = 0.05
    contourDisplay.SelectScaleArray = 'dataset1'
    contourDisplay.GlyphType = 'Arrow'
    contourDisplay.GlyphTableIndexArray = 'dataset1'
    contourDisplay.GaussianRadius = 0.0025
    contourDisplay.SetScaleArray = ['POINTS', 'dataset1']
    contourDisplay.ScaleTransferFunction = 'PiecewiseFunction'
    contourDisplay.OpacityArray = ['POINTS', 'dataset1']
    contourDisplay.OpacityTransferFunction = 'PiecewiseFunction'
    contourDisplay.DataAxesGrid = 'GridAxesRepresentation'
    contourDisplay.PolarAxes = 'PolarAxesRepresentation'

    # trace defaults for the display properties.
    contourDisplay.CompositeDataSetIndex = [0]
    contourDisplay.XArrayName = 'dataset1'
    contourDisplay.SeriesVisibility = ['dataset1']
    contourDisplay.SeriesLabel = ['dataset1', 'Tg', 'Points_X', 'Points_X', 'Points_Y', 'Points_Y', 'Points_Z',
                                  'Points_Z', 'Points_Magnitude', 'Points_Magnitude']
    contourDisplay.SeriesColor = ['dataset1', '0', '0', '0', 'Points_X', '0.889998', '0.100008', '0.110002', 'Points_Y',
                                  '0.220005', '0.489998', '0.719997', 'Points_Z', '0.300008', '0.689998', '0.289998',
                                  'Points_Magnitude', '0.6', '0.310002', '0.639994']
    contourDisplay.SeriesPlotCorner = ['Points_Magnitude', '0', 'Points_X', '0', 'Points_Y', '0', 'Points_Z', '0',
                                       'dataset1', '0']
    contourDisplay.SeriesLabelPrefix = ''
    contourDisplay.SeriesLineStyle = ['Points_Magnitude', '1', 'Points_X', '1', 'Points_Y', '1', 'Points_Z', '1',
                                      'dataset1', '1']
    contourDisplay.SeriesLineThickness = ['Points_Magnitude', '2', 'Points_X', '2', 'Points_Y', '2', 'Points_Z', '2',
                                          'dataset1', '2']
    contourDisplay.SeriesMarkerStyle = ['Points_Magnitude', '0', 'Points_X', '0', 'Points_Y', '0', 'Points_Z', '0',
                                        'dataset1', '0']
    contourDisplay.SeriesMarkerSize = ['Points_Magnitude', '4', 'Points_X', '4', 'Points_Y', '4', 'Points_Z', '4',
                                       'dataset1', '4']

    # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
    contourDisplay.ScaleTransferFunction.Points = [min_val, 0.0, 0.5, 0.0, max_val, 1.0, 0.5, 0.0]

    # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
    contourDisplay.OpacityTransferFunction.Points = [min_val, 0.0, 0.5, 0.0, max_val, 1.0, 0.5, 0.0]

    # show color legend
    contourDisplay.SetScalarBarVisibility(renderView, True)
    return contourDisplay

P_Display = set_renderView(renderView1,P,ins.p_atm,ins.p_atm + ins.L_y*10*ins.rho1,i=1,title='Pressure')
u_Display = set_renderView(renderView2,u,min_u,max_u,i=2,title='Velocity',component='Magnitude')
if ins.temp:
    T_Display = set_renderView(renderView3,T,0,ins.T0,i=3,title='Temperature')
    if ins.solidification:
        d_Display = set_renderView(renderView4,d,0,1e-12,i=4,title='Displacement',component='Magnitude')
    else:
        mu_Display = set_renderView(renderView4,mu,0,ins.max_eta,i=4,title='Viscosity')

# finally, restore active source
SetActiveSource(renderView1)