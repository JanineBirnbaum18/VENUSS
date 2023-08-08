import json
import os
import shutil
import numpy as np

h = 59
dt = 0.01

dictionary = {
    # mesh geometry
    "ndim": 2,
    "L_x": 1, # length of boundaries (m)
    "L_y": 1,
    "L_z": 1,
    "nx": h, # number of elements
    "ny": h,
    "etype": "cartesian", # element type cartesian, triangles grid, etc.

    # basis functions
    "u_k": 2, # polynomial order
    "p_k": 1,
    "t_k": 2,
    "t_enrich": True,
    "ls_k": 1,
    "mat_k": 1,

    # initial free surface level set
    "free_surface": False,
    "solve_air": True,
    "ls1p": "(x - 0.5)*(x - 0.5) + (y)*(y) - 0.2*0.2", # String using x,y,and z in m
    "ls1s": None,

    # initial temperature level set
    "ls2p": "0*x+1",
    "ls2s": None,

    # topography level set
    "topography": False,
    "solve_topography": True,
    "ls3p": "y-0.25",
    "ls3s": None,

    # Pressure
    "compressible": False,
    "steady": True,
    "steady_init": True,
    "p_atm": 0, # atmospheric pressure (Pa)
    "p_bound": 'top_left',
    "rho1": 2500, # density where ls1p<0 (lava) (kg/m3)
    "rho2": 1, # density where ls1p>0 (air) (kg/m3)
    "rho3": 3000, # density where ls3p<0 (ground) (kg/m3)

    "beta1": 1e-10, # compressibility (1/Pa)
    "beta2": 1e-10,
    "beta3": 1e-10,

    "surface_tension": 0.36, # surface tension in N/m

    # Temperature
    "temp": True,
    "solidification": True,
    "T0": 1000, # lava temperature (deg C)
    "T_init": None, # initial temperature field, None sets initial lava temperature to T0
    "T_atm": 0, # initial atmospheric temperature (deg C)
    "basal_temp_i": 50, # initial basal temperature in deg C
    "kappa1": 1, # thermal diffusivity (m2/s)
    "kappa2": 1e-7,
    "kappa3": 1e-8,
    "cp1": 1200, # heat capacity in J/KgK
    "cp2": 1200, # heat capacity in J/KgK
    "cp3": 1200, # heat capacity in J/KgK
    "emissivity": 0.93,
    "stefan_boltzman": 5.67*1e-8, # (W/m2K4)
    "crust_cover": 0.75, # surface area fraction
    "heat_transfer_coeff": 1e-9, # in air (W/m2K)

    # Viscosity
    "eta_exp": "etar*exp(vfta + vftb/(T + 273 - vftc))", # (Pas)
    "vfta": -4.55,#3.45,
    "vftb": 5000,#633.2,
    "vftc": 886.0,#450.0+273,
    "etar": 1, # relative viscosity
    "Tg": 786, # (deg C)
    "max_eta": 1e12, # (Pas)
    "eta2": 100, # (Pas)
    "eta3": 1e12,

    # Elastic properties
    "E": 100*1e9, # Young's modulus (Pa)
    "nu": 0.28, # Poisson's ratio

    # Body force, velocity boundary conditions
    "f_x": None, # body force (N/m3)
    "f_y": None,
    "left_ux": 0, # Dirichlet velocity condition (m/s) as float,str,or None
    "left_uy": 0,
    "left_dux": None, # Neumann velocity condition (1/s)
    "left_duy": None,
    "right_ux": 0,
    "right_uy": 0.001,
    "right_dux": None,
    "right_duy": None,
    "top_ux": 0,
    "top_uy": None,
    "top_dux": None,
    "top_duy": 0,
    "bottom_ux": 0,
    "bottom_uy": None,
    "bottom_dux": None,
    "bottom_duy": 0,

    "influx": False,
    "fix_ls": True,
    "influx_ux": '0', # velocity in m/s
    "influx_uy": '0.01',
    "influx_left": 0.475, # position of flux region in m
    "influx_right": 0.525,
    "influx_top": 0.26,
    "influx_bottom": 0.24,

    "basal_velocity": 'no_slip', # 'no_slip' or 'no_normal'

    # Temp boundary conditions
    "left_t": 'np.sin(ti*np.pi*2)*500 + 500', # Dirichlet temperature condition (deg C) as float,str, or None
    "left_dt": None, # Neumann temperature condition (deg C/m)
    "right_t": 1000,
    "right_dt": None,
    "top_t": None,
    "top_dt": 0,
    "bottom_t": None,
    "bottom_dt": 0,

    "influx_t": None,
    "influx_dt": None,

    "surface_temp": None,
    "surface_flux": None, # (W/m2) or 'forced convection,radiation'
    "basal_temp": None,
    "basal_flux": None, # in W/m2 or 'conduction'

    # time discretization
    "tf": 1, # final time in s
    "dt": dt, # time step in s
    "restart": False,

    # Solve options
    "stab_p": 'SUPG', # None, GLS, or SUPG for pressure stabilization
    "stab_psi": 'SUPG', # level set stabilization
    "stab_d": 'SUPG', # displacement stabilization
    "stab_t": 'GLS', # temp stabilization
    "epsilon_psi": 0.0001, # coefficient for viscous relaxation of free surface where interface velocity F = (1 - visc_coeff*curvature)
    "kappa_psi": 1e-6, # diffusion coefficient for curvature calculation
    "ndt0": 10, # number of substeps for BDF bootstrap

    "max_residual": 3e-12,
    "max_iter": 10,

    # output options
    "outfile": './Results/stefan_variable_sinusoid',
    "noutput": 1, # number of timesteps between output
    "ndigits": 5,
    "vtk": True,
    "viz": 'T_u_and_d.py',
    "plots": True,
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