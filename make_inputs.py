import json

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

    # initial free surface level set
    "free_surface": False,
    "ls1p": "0*x-1", # String using x,y,and z in m
    "ls1s": None,

    # initial temperature level set
    "ls2p": "0*x+1",
    "ls2s": None,

    # topography level set
    "topography": False,
    "solve_topography": False,
    "ls3p": "0*x+1",
    "ls3s": None,

    # Pressure
    "compressible": False,
    "steady": True,
    "p_atm": 0, # atmospheric pressure (Pa)
    "rho1": 2500, # density where ls1p<0 (lava) (kg/m3)
    "rho2": 1, # density where ls1p>0 (air) (kg/m3)
    "rho3": 3000, # density where ls3p<0 (ground) (kg/m3)

    "beta1": 1e-5, # compressibility (1/Pa)
    "beta2": 1e-5,
    "beta3": 1e-5,

    # Temperature
    "temp": True,
    "solidification": True,
    "T0": 1000, # lava temperature (deg C)
    "T_init": None, # initial temperature field, None sets initial lava temperature to T0
    "T_atm": 25, # initial atmospheric temperature (deg C)
    "basal_temp_i": 600, # initial basal temperature in deg C
    "kappa1": 1e-3, # thermal diffusivity (m2/s)
    "kappa2": 1e3,
    "kappa3": 1e-8,
    "cp": 840, # heat capacity in J/KgK
    "emissivity": 0.9,
    "stefan_boltzman": 5.67*1e-8, # (W/m2K4)
    "crust_cover": 0.75, # surface area fraction
    "heat_transfer_coeff": 1e-9, # in air (W/m2K)

    # Viscosity
    "eta_exp": "exp(vfta + vftb/(T + 273 - vftc))", # (Pas)
    "vfta": -4.55,
    "vftb": 5978.4,
    "vftc": 595.3,
    "Tg": 500, # (deg C)
    "max_eta": 1e7, # (Pas)
    "eta2": 1, # (Pas)
    "eta3": 1e7,

    # Elastic properties
    "E": 100*1e9, # Young's modulus (Pa)
    "nu": 0.28, # Poisson's ratio

    # Body force, velocity boundary conditions
    "f_x": None, # body force (N/m3)
    "f_y": None,
    "left_ux": 0, # Dirichlet velocity condition (m/s) as float,str,or None
    "left_uy": 0,
    "left_udx": None, # Neumann velocity condition (1/s)
    "left_udy": None,
    "right_ux": 0,
    "right_uy": 0.001,
    "right_udx": None,
    "right_udy": None,
    "top_ux": None,
    "top_uy": None,
    "top_udx": 0,
    "top_udy": 0,
    "bottom_ux": None,
    "bottom_uy": None,
    "bottom_udx": 0,
    "bottom_udy": 0,

    "influx": False,
    "influx_ux": 0, # velocity in m/s
    "influx_uy": 0,
    "influx_left": 0.225, # position of flux region in m
    "influx_right": 0.275,
    "influx_top": 0.09,
    "influx_bottom": 0.11,

    "basal_velocity": 'no_slip', # 'no_slip' or 'no_normal'

    # Temp boundary conditions
    "left_T": 0, # Dirichlet temperature condition (deg C) as float,str, or None
    "left_flux_T": None, # Neumann temperature condition (deg C/m)
    "right_T": 1000,
    "right_flux_T": None,
    "top_T": None,
    "top_flux_T": 0,
    "bottom_T": None,
    "bottom_flux_T": 0,

    "influx_T": 1100,
    "influx_flux_T": None,

    "surface_temp": None,
    "surface_flux": None, # (W/m2) or 'forced convection,radiation'
    "basal_temp": None,
    "basal_flux": None, # in W/m2 or 'conduction'

    # time discretization
    "tf": 0.1, # final time in s
    "dt": 0.05, # time step in s
    "restart": False,

    # Solve options
    "stab_p": 'None', # None, SUPG for pressure stabilization
    "stab_t": 'GLS', # None, SUPG, or GLS for temp stabilization
    "solve_air": False, # turn to false to neglect velocity solution in fluid 2

    # output options
    "outfile": 'cooling_solidification_moving',
    "noutput": 2, # number of timesteps between output
    "vtk": True,
    "plots": True,
    "plot_mesh": False}

with open(dictionary["outfile"] + '.json', 'w') as outfile:
    json.dump(dictionary,outfile)