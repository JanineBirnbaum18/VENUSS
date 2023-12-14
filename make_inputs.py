import json
import os
import shutil
import numpy as np

dictionary = {
    # mesh geometry
    "ndim": 2,
    "symmetry": "axial", # problem symmetry either "planar" or "axial"
    "L_x": 1, # length of boundaries (m)
    "L_y": 1,
    "L_z": 1,
    "nx": 40, # number of elements
    "ny": 40,
    "etype": "cartesian", # element type cartesian, triangles grid, etc.

    # basis functions
    "u_k": 2, # polynomial order
    "p_k": 1,
    "t_k": 2,
    "t_enrich": True,
    "ls_k": 1,
    "mat_k": 1,

    # initial free surface level set
    "free_surface": True,
    "solve_air": True,
    "ls1p": "(X)*(X) + (Y-0.35)*(Y-0.35) - 0.3*0.3", # String using x,y,and z in m
    "ls1s": None,

    # initial temperature level set
    "ls2p": "0*X+1",
    "ls2s": None,

    # topography level set
    "topography": True,
    "solve_topography": True,
    "ls3p": "Y-0.35",
    "ls3s": None,

    # Pressure
    "compressible": True,
    "steady": False,
    "steady_init": True,
    "p_amb": 0, # ambient pressure (Pa)
    "p_basal": True,
    "p_bound": 'bottom_right',
    #"p_bound2": 'top_right',
    #"p_bound2_val": 0,
    "rho1": 2700, # density where ls1p<0 (lava) (kg/m3)
    "rho2": 1, # density where ls1p>0 (air) (kg/m3)
    "rho3": 3000, # density where ls3p<0 (ground) (kg/m3)
    "surface_tension": 0.08, # surface tension between fluids 1 and 2 (lava in air) (Pa/m)

    "beta1": 1e-12, # compressibility (1/Pa)
    "beta2": 1e-5,
    "beta3": 1e-12,

    # Temperature
    "temperature": True,
    "solidification": False,
    "T0": 1200, # lava temperature (deg C)
    "T_amb": 50, # initial ambient temperature (deg C)
    "T_init": None, # initial temperature field, None sets initial lava temperature to T0
    "basal_temp_i": 0, # initial basal temperature in deg C
    "kappa1": 1e-12, # thermal diffusivity (m2/s)
    "kappa2": 1e-7,
    "kappa3": 1e-12,
    "cp1": 1200, # heat capacity in J/KgK
    "cp2": 1200, # heat capacity in J/KgK
    "cp3": 1200, # heat capacity in J/KgK
    "emissivity": 0.93,
    "stefan_boltzmann": 5.67*1e-8, # (W/m2K4)
    "crust_cover": 0.75, # surface area fraction
    "heat_transfer_coeff": 1e-9, # in air (W/m2K)

    # Viscosity
    "eta_exp": "etar*exp(vfta + vftb/(T + 273 - vftc))", # (Pas)
    "vfta": np.log(200), #3.45,
    "vftb": 0, #633.2,
    "vftc": 0, #450.0+273,
    "etar": 1, # relative viscosity
    "Tg": 1000, # (deg C)
    "max_eta": 1e12, # (Pas)
    "eta2": 1, # (Pas)
    "eta3": 1e12,

    # Elastic properties
    "E": 100*1e9, # Young's modulus (Pa)
    "nu": 0.28, # Poisson's ratio

    # Body force, velocity boundary conditions
    "f_x": None, # body force (N/m3)
    "f_y": "-rho*9.81",
    "left_ux": 0, # Dirichlet velocity condition (m/s) as float,str,or None
    "left_uy": None,
    "left_dux": None, # Neumann stress condition (Pa/m)
    "left_duy": 0,
    "right_ux": None,
    "right_uy": None,
    "right_dux": 0,
    "right_duy": 0,
    "top_ux": 0,
    "top_uy": None,
    "top_dux": None,
    "top_duy": 0,
    "bottom_ux": 0,
    "bottom_uy": 0,
    "bottom_dux": None,
    "bottom_duy": None,

    "influx": False,
    "fix_ls": True,
    "fix_ls_bound": 'left',
    "influx_ux": "((1/2*(0.04**2 - (0.04-(Y))**2)*2700*9.81/53*(np.sin(13.25*np.pi/180)) - 0.09*np.exp(1000*(Y-0.04)))*(Y<=0.04))", # velocity in m/s
    "influx_dux": None,
    "influx_uy": 0,
    "influx_duy": None,
    "influx_left": -0.01, # position of flux region in m
    "influx_right": 0.01,
    "influx_top": 0.05,
    "influx_bottom": -0.05,

    "basal_velocity": 'no_slip', # 'no_slip' or 'no_normal'

    # Temperature boundary conditions
    "left_t": None, # Dirichlet temperature condition (deg C) as float,str, or None
    "left_dt": 0, # Neumann temperature condition (deg C/m)
    "right_t": None,
    "right_dt": 0,
    "top_t": 50,
    "top_dt": None,
    "bottom_t": None,
    "bottom_dt": 50,

    "influx_t": "1150*((Y>=0)*(Y<=0.0401)) + 50",
    "influx_dt": None,

    "surface_temp": None,
    "surface_flux": None, # (W/m2) or 'forced convection,radiation'
    "basal_temp": None,
    "basal_flux": None, # in W/m2 or 'conduction'

    # time discretization
    "tf": 0.04, # final time in s
    "dt": 0.01, # time step in s
    "restart": False,

    # Solve options
    "stab_p": 'SUPG', # None or SUPG for pressure stabilization
    "stab_psi": 'GLS', # level set stabilization (None, SUPG, GLS)
    "stab_d": 'SUPG', # displacement stabilization (None, SUPG)
    "stab_t": 'GLS', # temp stabilization (None, SUPG, GLS)
    "epsilon_psi": 0.001, # coefficient for viscous relaxation of free surface where interface velocity F = (1 - visc_coeff*curvature)
    "kappa_psi": 1e-6, # diffusion coefficient for curvature calculation
    "ndt0": 1, # number of substeps for BDF bootstrap

    "max_residual": 1e-11,
    "max_iter": 5,
    "n_outer_iter": 1,

    # output options
    "outfile": './Results/spreading_drop_axial',
    "noutput": 1, # number of timesteps between output
    "ndigits": 3,
    "vtk": True,
    "viz": 'basic_viz.py',
    "plots": False,
    "plot_mesh": False,

    # true solutions
    "true_p": False,
    "true_ux": False,
    "true_uy": False,
    "true_dx": False,
    "true_dy": False,
    "true_ls1": False,
    "true_t": False,
    "true_ls2": False}

try:
    os.listdir('./' + dictionary["outfile"])
except Exception as e:
    os.mkdir('./' + dictionary["outfile"])

with open(dictionary["outfile"] + '/' + dictionary["outfile"].split('/')[-1] + '.json', 'w') as outfile:
    json.dump(dictionary,outfile)

if not(type(dictionary["viz"]) is type(None)):
    shutil.copy('./' + dictionary["viz"],'./' + dictionary["outfile"] + '/' + dictionary["viz"])
    with open(r'./' + dictionary["viz"], 'r') as file:
        vizdata = file.read()
        vizdata = vizdata.replace('fileloc',str(os.getcwd()) + dictionary["outfile"].strip('.'))

    with open(r'./' + dictionary["outfile"] + '/' + dictionary["viz"], 'w') as file:
        file.write(vizdata)
    file.close()