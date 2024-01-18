import json
import os
import shutil
import numpy as np

dictionary = {
    # mesh geometry
    "ndim": 2,
    "symmetry": "planar", # problem symmetry either "planar" or "axial"
    "L_x": 1, # length of boundaries (m)
    "L_y": 1,
    "nx": 29, # number of elements
    "ny": 29,
    "etype": "cartesian", # element type cartesian, triangles grid, etc.

    # basis functions
    "u_k": 2, # polynomial order
    "p_k": 1,
    "t_k": 1,
    "t_enrich": True,
    "ls_k": 1,
    "mat_k": 0,

    # initial free surface level set
    "free_surface": True,
    "solve_air": True,
    "ls1p": "0.05*cos(2*2*3.1416*x) + (y-0.5)", # String using x,y,and z in m
    "ls1s": None,

    # initial temperature level set
    "ls2p": "0*X+1",
    "ls2s": None,

    # topography level set
    "topography": False,
    "solve_topography": True,
    "ls3p": "0*X+1",
    "ls3s": None,

    # Pressure
    "compressible": False,
    "steady": True,
    "steady_init": True,
    "p_amb": 0, # ambient pressure (Pa)
    "p_bound": 'top_left',
    "rho1": 2000, # density where ls1p<0 (lava) (kg/m3)
    "rho2": 1, # density where ls1p>0 (air) (kg/m3)
    "rho3": 1, # density where ls3p<0 (ground) (kg/m3)
    "surface_tension": 0.0, # surface tension between fluids 1 and 2 (lava in air) (Pa/m)

    "beta1": 1e-10, # compressibility (1/Pa)
    "beta2": 1e-10,
    "beta3": 1e-10,

    # Temperature
    "temperature": False,
    "solidification": False,
    "T0": 0, # lava temperature (deg C)
    "T_amb": 0, # initial ambient temperature (deg C)
    "T_init": None, # initial temperature field, None sets initial lava temperature to T0
    "basal_temp_i": 0, # initial basal temperature in deg C
    "kappa1": 1e-12, # thermal diffusivity (m2/s)
    "kappa2": 1e-12,
    "kappa3": 1e-12,
    "cp1": 1200, # heat capacity in J/KgK
    "cp2": 1200, # heat capacity in J/KgK
    "cp3": 1200, # heat capacity in J/KgK
    "emissivity": 0.93,
    "stefan_boltzmann": 5.67*1e-8, # (W/m2K4)
    "crust_cover": 0.75, # surface area fraction
    "heat_transfer_coeff": 1e-9, # in air (W/m2K)

    # Viscosity
    "eta_exp": "etar*10**(vfta + vftb/(T + 273 - vftc))", # (Pas)
    "vfta": np.log10(800), #3.45,
    "vftb": 0, #633.2,
    "vftc": 0, #450.0+273,
    "etar": 1, # relative viscosity
    "Tg": 0, # (deg C)
    "max_eta": 1, # (Pas)
    "eta2": 0.01, # (Pas)
    "eta3": 1,

    # Elastic properties
    "E": 100*1e9, # Young's modulus (Pa)
    "nu": 0.28, # Poisson's ratio

    # Body force, velocity boundary conditions
    "f_x": None, # body force (N/m3)
    "f_y": "-9.81*rho",
    "left_ux": 0, # Dirichlet velocity condition (m/s) as float,str,or None
    "left_uy": None,
    "left_dux": None, # Neumann stress condition (Pa/m)
    "left_duy": 0,
    "right_ux": 0,
    "right_uy": None,
    "right_dux": None,
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
    "fix_ls": False,
    "fix_ls_bound": 'left',
    "influx_ux": 0, # velocity in m/s
    "influx_dux": None,
    "influx_uy": 0,
    "influx_duy": None,
    "influx_left": -0.01, # position of flux region in m
    "influx_right": 0.01,
    "influx_top": 1.05,
    "influx_bottom": -0.05,

    "basal_velocity": 'no_slip', # 'no_slip' or 'no_normal'

    # Temperature boundary conditions
    "left_t": 0, # Dirichlet temperature condition (deg C) as float,str, or None
    "left_dt": None, # Neumann temperature condition (deg C/m)
    "right_t": 0,
    "right_dt": None,
    "top_t": 0,
    "top_dt": None,
    "bottom_t": 0,
    "bottom_dt": None,

    "influx_t": 0,
    "influx_dt": None,

    "surface_temp": None,
    "surface_flux": None, # (W/m2) or 'forced convection,radiation'
    "basal_temp": None,
    "basal_flux": None, # in W/m2 or 'conduction'

    # time discretization
    "tf": 2, # final time in s
    "dt": 0.01, # time step in s
    "restart": False,

    # Solve options
    "stab_p": 'SUPG', # None or SUPG for pressure stabilization
    "stab_psi": 'GLS', # level set stabilization (None, SUPG, GLS)
    "stab_d": 'SUPG', # displacement stabilization (None, SUPG)
    "stab_t": 'GLS', # temp stabilization (None, SUPG, GLS)
    "epsilon_psi": 0.00001, # coefficient for viscous relaxation of free surface where interface velocity F = (1 - visc_coeff*curvature)
    "kappa_psi": 1e-4, # diffusion coefficient for curvature calculation
    "ndt0": 1, # number of substeps for BDF bootstrap

    "max_residual": 1e-10,
    "max_iter": 10,
    "n_outer_iter": 2,

    # output options
    "outfile": './Results/relax_29x_01t',
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
    "true_ls1": "(0.05*np.cos(2*2*3.1416*X)*(np.exp(-ti/(2*2*2*3.1416*800/2000/9.81)/((0.5*2*2*3.1416 + np.sinh(0.5*2*2*3.1416)*np.cosh(0.5*2*2*3.1416))/np.sinh(0.5*2*2*3.1416)**2))) + (Y-0.5))",
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