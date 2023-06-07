import json
import os
import shutil
import numpy as np

dictionary = {
    # mesh geometry
    "ndim": 2,
    "L_x": 0.5, # length of boundaries (m)
    "L_y": 0.5,
    "L_z": 0.5,
    "nx": 30, # number of elements
    "ny": 30,
    "etype": "cartesian", # element type cartesian, triangles grid, etc.

    # basis functions
    "u_k": 2, # polynomial order
    "p_k": 1,
    "t_k": 2,
    "ls_k": 1,
    "nmat": 1,

    # initial free surface level set
    "free_surface": True,
    "solve_air": True,
    "ls1p": "(x - 0.25)*(x - 0.25) + (y - 0.25)*(y - 0.25) - 0.1*0.1", # String using x,y,and z in m
    "ls1s": None,

    # initial temperature level set
    "ls2p": "0*x+1",
    "ls2s": None,

    # topography level set
    "topography": True,
    "solve_topography": True,
    "ls3p": "y-0.25",
    "ls3s": None,

    # Pressure
    "compressible": False,
    "steady": True,
    "p_atm": 0, # atmospheric pressure (Pa)
    "rho1": 2500, # density where ls1p<0 (lava) (kg/m3)
    "rho2": 1, # density where ls1p>0 (air) (kg/m3)
    "rho3": 3000, # density where ls3p<0 (ground) (kg/m3)

    "beta1": 1e-10, # compressibility (1/Pa)
    "beta2": 1e-10,
    "beta3": 1e-10,

    "surface_tension": 0.36, # surface tension in N/m

    # Temperature
    "temp": False,
    "solidification": False,
    "T0": 100, # lava temperature (deg C)
    "T_init": None, # initial temperature field, None sets initial lava temperature to T0
    "T_atm": 100, # initial atmospheric temperature (deg C)
    "basal_temp_i": 600, # initial basal temperature in deg C
    "kappa1": 1e3, # thermal diffusivity (m2/s)
    "kappa2": 1e3,
    "kappa3": 1e-8,
    "cp": 840, # heat capacity in J/KgK
    "emissivity": 0.9,
    "stefan_boltzman": 5.67*1e-8, # (W/m2K4)
    "crust_cover": 0.75, # surface area fraction
    "heat_transfer_coeff": 1e-9, # in air (W/m2K)

    # Viscosity
    "eta_exp": "exp(vfta + vftb/(T + 273 - vftc))", # (Pas)
    "vfta": np.log(1e2),
    "vftb": 0.0,
    "vftc": 0.0,
    "Tg": 500, # (deg C)
    "max_eta": 1e7, # (Pas)
    "eta2": 1, # (Pas)
    "eta3": 1e7,

    # Elastic properties
    "E": 100*1e9, # Young's modulus (Pa)
    "nu": 0.28, # Poisson's ratio

    # Body force, velocity boundary conditions
    "f_x": None, # body force (N/m3)
    "f_y": "-rho*9.81",
    "left_ux": 0, # Dirichlet velocity condition (m/s) as float,str,or None
    "left_uy": None,
    "left_udx": 0, # Neumann velocity condition (1/s)
    "left_udy": None,
    "right_ux": 0,
    "right_uy": 0,
    "right_udx": 0,
    "right_udy": None,
    "top_ux": 0,
    "top_uy": 0,
    "top_udx": None,
    "top_udy": None,
    "bottom_ux": 0,
    "bottom_uy": 0,
    "bottom_udx": None,
    "bottom_udy": None,

    "influx": False,
    "influx_ux": 0, # velocity in m/s
    "influx_uy": 0,
    "influx_left": 0.225, # position of flux region in m
    "influx_right": 0.275,
    "influx_top": 0.09,
    "influx_bottom": 0.11,

    "basal_velocity": 'no_slip', # 'no_slip' or 'no_normal'

    # Temp boundary conditions
    "left_T": 100, # Dirichlet temperature condition (deg C) as float,str, or None
    "left_flux_T": None, # Neumann temperature condition (deg C/m)
    "right_T": 100,
    "right_flux_T": None,
    "top_T": 100,
    "top_flux_T": None,
    "bottom_T": 100,
    "bottom_flux_T": None,

    "influx_T": 1100,
    "influx_flux_T": None,

    "surface_temp": None,
    "surface_flux": None, # (W/m2) or 'forced convection,radiation'
    "basal_temp": None,
    "basal_flux": None, # in W/m2 or 'conduction'

    # time discretization
    "tf": 2, # final time in s
    "dt": 0.01, # time step in s
    "restart": False,

    # Solve options
    "stab_p": None, # None, SUPG for pressure stabilization
    "stab_t": 'GLS', # None, SUPG, or GLS for temp stabilization
    "visc_coeff": 0.01, # coefficient for viscous relaxation of free surface where interface velocity F = (1 - visc_coeff*curvature)
    "epsilon_psi": 1e-5,

    # output options
    "outfile": './Results/spreading_drop_iso',
    "noutput": 1, # number of timesteps between output
    "ndigits": 4,
    "vtk": True,
    "viz": 'basic_viz.py',
    "plots": True,
    "plot_mesh": False,

    # true solutions
    "true_p": False,
    "true_u": False,
    "true_d": False,
    "true_ls1": False,
    "true_t":False,
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