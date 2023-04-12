import sys
import os
import json
from types import SimpleNamespacels_u
import h5py

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate as sciinterp
import skfmm
import getfem as gf
from getfem import *

outfile = './Results/spreading_drop_iso_lslinear_lsinterp'

# import simulation parameters
with open(outfile + '/' + outfile.split('/')[-1] + '.json', 'r') as openfile:
    dictionary = json.load(openfile)

ins = SimpleNamespace(**dictionary)

dx = ins.L_x/ins.nx
dy = ins.L_y/ins.ny
xs = np.arange(2*ins.nx)*dx/2+dx/4
ys = np.arange(2*ins.ny)*dy/2+dy/4
x_grid, y_grid = np.meshgrid(xs, ys)

if (type(ins.Tg) is float) or (type(ins.Tg) is int):
    Tg = ins.Tg
elif type(ins.Tg) is type(str):
    Tg = eval(ins.Tg.replace('vft', 'ins.vft'))
elif type(ins.Tg) is type(None):
    Tg = ins.vftc + ins.vftb/(12-ins.vfta) - 273

ndigits = int(np.floor(np.log10(ins.tf/ins.dt))) + len(str(ins.dt).split('.')[-1].lstrip('0')
                                                       ) + len(str(ins.dt).split('.')[0].rstrip('0'))

if '.' in str(ins.dt)[:ndigits+2]:
    ndigits += 1
    ndecimal = len(str(ins.dt)[:ndigits+2].split('.')[-1].rstrip('0'))
else:
    ndecimal = 0

if ins.restart:
    try:
        hf = h5py.File(ins.outfile + '/' + ins.outfile.split('/')[-1] + '.h5', 'r')
        U = hf.get('last_u')[:]
        Previous_u = hf.get('last2_u')[:]
        P = hf.get('last_p')[:]
        Previous_p = hf.get('last2_p')[:]
        D = hf.get('last_d')[:]
        Previous_d = hf.get('last2_d')[:]
        if ins.temp:
            T = hf.get('last_T')[:]
            Previous_T = hf.get('last2_T')[:]
        if ins.free_surface:
            last_Ls1 = hf.get('last_Ls1')[:]
        if ins.temp & ins.solidification:
            last_Ls2 = hf.get('last_Ls2')[:]
        if ins.topography:
            last_Ls3 = hf.get('last_Ls3')[:]
        last_ti = hf.get('last_ti')[:]
        hf.close()
    except Exception as e:
        ins.restart = False

#### Mesh generation #####
if ins.plots:
    fig, ax = plt.subplots()
# Mesh generation for structured quad elements
if ins.ndim == 1:
    mesh = gf.Mesh(ins.etype, np.linspace(0, ins.L_x, ins.nx + 1))
elif ins.ndim == 2:
    mesh = gf.Mesh(ins.etype, np.linspace(0, ins.L_x, ins.nx + 1), np.linspace(0, ins.L_y, ins.ny + 1))
elif ins.ndim == 3:
    mesh = gf.Mesh(ins.etype, np.linspace(0, ins.L_x, ins.nx + 1), np.linspace(0, ins.L_y, ins.ny + 1),
                   np.linspace(0, ins.L_z, ins.nz + 1))
else:
    print('no valid dimension')

# outward vector in [x,y], tolerance angle theta (rad)

if ins.ndim == 2:
    # noinspection PyUnboundLocalVariable
    fb_left = mesh.outer_faces_with_direction([-1., 0.], 0.01)  # boundary 1
    fb_right = mesh.outer_faces_with_direction([1., 0.], 0.01)  # boundary 2
    fb_top = mesh.outer_faces_with_direction([0., 1.], 0.01)  # boundary 3
    fb_bottom = mesh.outer_faces_with_direction([0., -1.], 0.01)  # boundary 4

    if ins.influx:
        fb_influx = mesh.faces_from_cvid(mesh.convexes_in_box((ins.influx_left, ins.influx_bottom),
                                                              (ins.influx_right, ins.influx_top)))

bounds = ['left', 'right', 'top', 'bottom']
bounds_type = []
for i, bound in enumerate(bounds):
    mesh.set_region(i + 1, eval('fb_' + bound))
    if ins.temp:
        bounds_type = np.append(bounds_type, type(eval('ins.' + bound + '_T')))
    else:
        bounds_type = np.append(bounds_type, type(eval('ins.' + bound + '_ux')))

if ins.influx:
    mesh.set_region(i + 2, fb_influx)

if ins.vtk:
    # export to vtk for visualization through python
    mesh.export_to_vtk(ins.outfile + '/' + ins.outfile.split('/')[-1] + '_mesh.vtk')

# Levelset definition:
# free surface
if ins.free_surface:
    ls1 = gf.LevelSet(mesh, ins.ls_k, ins.ls1p, ins.ls1s)
    mls = gf.MeshLevelSet(mesh)
    mls.add(ls1)

# temperature contour
if ins.temp & ins.solidification:
    if not ins.free_surface:
        mls = gf.MeshLevelSet(mesh)
    ls2 = gf.LevelSet(mesh, ins.ls_k, ins.ls2p, ins.ls2s)
    mls.add(ls2)

# topography
if ins.topography:
    if not (ins.free_surface | (ins.temp & ins.solidification)):
        mls = gf.MeshLevelSet(mesh)
    ls3 = gf.LevelSet(mesh, ins.ls_k, ins.ls3p, ins.ls3s)
    mls.add(ls3)

if ins.restart:
    if ins.free_surface:
        ls1.set_values(last_Ls1)
    if ins.temp & ins.solidification:
        ls2.set_values(last_Ls2)
    if ins.topography:
        ls3.set_values(last_Ls3)

try:
    mls.adapt()
except Exception as e:
    print('No level sets')

if ins.free_surface:
    ls1_init = ls1.values(0)

if ins.plots:
    # visualize mesh points in python
    if ins.free_surface | (ins.temp & ins.solidification) | ins.topography:
        [x_ls, y_ls] = mls.cut_mesh().pts()  # get nodes
    else:
        [x_ls, y_ls] = mesh.pts()  # get nodes
    ax.scatter(x_ls, y_ls, 50, 'r', zorder=2)  # plot nodes

    # plot all mesh edges (no overlap)
    if ins.free_surface | (ins.temp & ins.solidification) | ins.topography:
        for face in mls.cut_mesh().faces_from_cvid('merge').transpose():
            edge = mls.cut_mesh().pid_in_faces(CVFIDs=face)
            ax.plot(x_ls[edge], y_ls[edge], 'k', zorder=0)

    # plot edges on BCs
    lr = True
    lb = True
    for bound in bounds:
        for face in eval('fb_' + bound).transpose():
            edge = mesh.pid_in_faces(CVFIDs=face)

            if ins.temp:
                bound_type = type(eval('ins.' + bound + '_T'))
            else:
                bound_type = type(eval('ins.' + bound + '_ux'))
            if bound_type is not type(None):
                color = 'r'  # plot Dirichlet bounds on temperature in red
                if ins.temp:
                    if lr:
                        label = 'Temp BC'
                        lr = False
                    else:
                        label = '_nolegend_'
                else:
                    if lr:
                        label = 'Vel BC'
                        lr = False
                    else:
                        label = '_nolegend_'
            else:
                color = 'b'  # plot Dirichlet bounds on temperature in blue
                if lb:
                    label = 'Flux BC'
                    lb = False
                else:
                    label = '_nolegend_'

            [x0, y0] = mesh.pts()
            ax.plot(x0[edge], y0[edge], color, linewidth=3, zorder=1, label=label)

    # plt.tricontour(x_ls,y_ls,ls.values(0),levels=[0],colors='k')

    ax.axis('equal')
    ax.set_xlabel('x (m)', fontsize=14)
    ax.set_ylabel('y (m)', fontsize=14)
    ax.set_title('Mesh', fontsize=16)
    plt.legend(bbox_to_anchor=(1.02, 1))
    fig.savefig(ins.outfile + '/' + ins.outfile.split('/')[-1] + '_mesh.pdf')

# mim = gf.MeshIm(mesh, 4)
# integration methods on cut domain
# third positional argument where: 'all', 'inside' (p<0), 'outside' (p>0), 'boundary'
# when there is more than one level set, ordered a, b, c, ...
# with desc string a*b is th intersection of domains, a+b is union, b-a is minus, !a is complementary
# CAUTION: this integration method will be defined only on the element cut by the level-set.
# For the ‘ALL’, ‘INSIDE’ and ‘OUTSIDE’ options it is mandatory to use the method MeshIm.set_integ()
# to define the integration method on the remaining elements

if ins.free_surface | (ins.temp & ins.solidification) | ins.topography:
    mim_all = gf.MeshIm('levelset', mls, 'all', gf.Integ('IM_TRIANGLE(5)'))
    mim_all.set_integ(4)
    if ins.solve_topography:
        if ins.solve_air | (not ins.free_surface):
            mim = gf.MeshIm('levelset', mls, 'all', gf.Integ('IM_TRIANGLE(5)'))
        else:
            mim = gf.MeshIm('levelset', mls, 'inside(a+c)', gf.Integ('IM_TRIANGLE(5)'))
        mim.set_integ(4)
    else:
        if ins.solve_air | (not ins.free_surface):
            mim = gf.MeshIm('levelset', mls, 'outside(c)', gf.Integ('IM_TRIANGLE(5)'))
        else:
            mim = gf.MeshIm('levelset', mls, 'inside(a-c)', gf.Integ('IM_TRIANGLE(5)'))
        mim.set_integ(4)
    if ins.free_surface:
        mim_surf = gf.MeshIm('levelset', mls, 'boundary(a+c)', gf.Integ('IM_TRIANGLE(5)'))
        mim_surf.set_integ(4)

    if ins.topography:
        mim_base = gf.MeshIm('levelset', mls, 'boundary(c)', gf.Integ('IM_TRIANGLE(5)'))
        mim_base.set_integ(4)

    mim_integ = gf.MeshIm('levelset', mls, 'boundary(a-c)', gf.Integ('IM_TRIANGLE(5)'))
    mim_integ.set_integ(4)
else:
    mim_all = gf.MeshIm(mesh,4)
    mim = gf.MeshIm(mesh,4)
    mim_integ = gf.MeshIm(mesh,4)

if ins.plots:
    fig2, ax2 = plt.subplots()
    ax2.scatter(mim.im_nodes()[0, :], mim.im_nodes()[1, :], 10, 'b')
    ax2.scatter(mim_integ.im_nodes()[0, :], mim_integ.im_nodes()[1, :], 10, 'r')

# Define variable fields and approximations
# velocity
mfu = gf.MeshFem(mesh, ins.ndim)  # vector field
mfu.set_classical_fem(ins.u_k)  # continuous piecewise quadratic
# pressure
mfp = gf.MeshFem(mesh, 1)  # scalar field
mfp.set_classical_fem(ins.p_k)  # continuous piecewise linear
# mfp.set_classical_discontinuous_fem(1) # continuous piecewise linear
# mfp.set_classical_discontinuous_fem(0) # continuous piecewise linear

mfls = gf.MeshFem(mesh, 1)
mfls.set_classical_fem(ins.ls_k)

# temperature
if ins.temp:
    mft = gf.MeshFem(mesh, 1)  # scalar field
    mft.set_classical_fem(ins.t_k)  # continuous piecewise quadratic
    # mft.set_classical_discontinuous_fem(2)

# mfls_p = gf.MeshFem('levelset',mls,mfp)
# mfls_u = gf.MeshFem('levelset',mls,mfu)

# if solidification:
#    mim_crust = gf.MeshIm('levelset',mls,'inside(a+b+!c)', gf.Integ('IM_STRUCTURED_COMPOSITE(IM_TRIANGLE(6),3)'))

# above topography
if ins.free_surface | (ins.temp & ins.solidification) | ins.topography:
    ind_p = 'mfp.basic_dof_from_cv(np.append(mim_integ.convex_index(),mim.convex_index()))'
    ind_u = 'mfu.basic_dof_from_cv(np.append(mim_integ.convex_index(),mim.convex_index()))'
    if ins.temp:
        ind_t = 'mft.basic_dof_from_cv(np.append(mim_integ.convex_index(),mim.convex_index()))'
else:
    ind_p = 'mfp.basic_dof_from_cv(np.append(mim.convex_index(),mim.convex_index()))'
    ind_u = 'mfu.basic_dof_from_cv(np.append(mim.convex_index(),mim.convex_index()))'
    if ins.temp:
        ind_t = 'mft.basic_dof_from_cv(np.append(mim.convex_index(),mim.convex_index()))'

# get mesh coordinates
D_u = mfu.basic_dof_nodes()
ones_u = np.ones(D_u.shape[1])
x_u = D_u[0, :]
y_u = D_u[1, :]

# x_u2 = x_u.copy().reshape(int(len(x_u)/2),2)
# y_u2 = y_u.copy().reshape(int(len(y_u)/2),2)

D_p = mfp.basic_dof_nodes()
ones_p = np.ones(D_p.shape[1])
x_p = D_p[0, :]
y_p = D_p[1, :]

D_ls = mfls.basic_dof_nodes()
ones_ls = np.ones(D_ls.shape[1])
x_ls = D_ls[0, :]
y_ls = D_ls[1, :]
# n = int(np.sqrt(x_ls.shape[0]))
if ins.ndim == 2:
    edges_ls = np.where((x_ls < dx) | (x_ls > (ins.L_x - dx)) | (y_ls < dy) | (y_ls > (ins.L_y - dy)))

if ins.temp:
    D_t = mft.basic_dof_nodes()
    ones_t = np.ones(D_t.shape[1])
    x_t = D_t[0, :]
    y_t = D_t[1, :]

# d_temp = sciinterp.griddata(D_ls.transpose(), ls.values(0),
#                            np.array([x_grid.flatten(),y_grid.flatten()]).transpose(),
#                            method='linear',fill_value=0).reshape((ny+1,nx+1))
# ls_init = skfmm.distance(d_temp,dx=[dy,dx]).flatten()
# ls.set_values(ls_init.flatten())
# mls.adapt()

if ins.free_surface:
    Ls1 = ls1.values(0)
    if ins.ndim == 2:
        Ls1[edges_ls] = max(ins.L_x, ins.L_y)
    elif ins.ndim == 3:
        Ls1[edges_ls] = max(max(ins.L_x, ins.L_y), ins.L_z)
    Ls_ext = sciinterp.griddata(D_ls.transpose(), Ls1,
                                np.array([x_grid.flatten(), y_grid.flatten()]).transpose(), method='nearest')
    Ls1 = skfmm.distance(Ls_ext.reshape((2 * ins.ny, 2 * ins.nx)), dx=[dy / 2, dx / 2]).flatten()
    Ls1 = sciinterp.griddata(np.array([x_grid.flatten(), y_grid.flatten()]).transpose(), Ls1,
                             D_ls.transpose(), method='nearest')
    ls1.set_values(Ls1)
    mls.adapt()
    # mim.adapt()

    if ins.plots:
        ax.tricontour(x_ls, y_ls, ls1.values(0), levels=[0], colors='k')
        cbar = ax.tricontourf(x_ls, y_ls, ls1.values(0), levels=[-10, 0, 10])
        # plt.colorbar(cbar,ax=ax)

if ins.plots:
    ax2.set_xlim(ax.get_xlim())
    ax2.set_ylim(ax.get_ylim())

    ax2.set_xlabel('x (m)')
    ax2.set_ylabel('y (m)')

######## Problem setup ################

md_init = gf.Model('real')  # real vs complex system
md = gf.Model('real')  # real vs complex system

# Add Data
# Level set
if ins.free_surface:
    md.add_initialized_fem_data('ls1', mfls, ls1.values(0))

# drop dof outside region of interest
if ins.free_surface | ins.topography:
    mfp_cut = gf.MeshFem('partial', mfp, np.arange(mfp.nbdof()))
    mfu_cut = gf.MeshFem('partial', mfu, np.arange(mfu.nbdof()))
    if ins.temp:
        mft_cut = gf.MeshFem('partial', mft, np.arange(mft.nbdof()))

    mfp_cut.set_partial(eval(ind_p))
    mfu_cut.set_partial(eval(ind_u))
    if ins.temp:
        mft_cut.set_partial(eval(ind_t))

if ins.restart:
    md.add_initialized_fem_data('Previous_u', mfu, U)
    md.add_initialized_fem_data('Previous2_u', mfu, Previous_u)
    md.add_initialized_fem_data('Previous_p', mfp, P)
    md.add_initialized_fem_data('Previous2_p', mfp, Previous_p)
    md.add_initialized_fem_data('Previous_d', mfu, D)
    md.add_initialized_fem_data('Previous2_d', mfu, Previous_d)
else:
    u_init = ones_u * 0
    p_init = ones_p * ins.p_atm
    d_init = ones_u * 0
    md.add_initialized_fem_data('Previous_u', mfu, u_init)
    md.add_initialized_fem_data('Previous2_u', mfu, u_init)
    md.add_initialized_fem_data('p_init', mfp, p_init)
    md.add_initialized_fem_data('Previous_p', mfp, p_init)
    md.add_initialized_fem_data('Previous2_p', mfp, p_init)
    md.add_initialized_fem_data('Previous_d', mfu, d_init)
    md.add_initialized_fem_data('Previous2_d', mfu, d_init)

# Density
rho = ones_ls * ins.rho1
if ins.free_surface:
    rho[ls1.values(0) > 0] = ins.rho2
if ins.topography:
    rho[ls3.values(0) < 0] = ins.rho3
md.add_initialized_fem_data('rho', mfls, rho)
md.add_initialized_fem_data('rho_init', mfls, rho)

md_init.add_initialized_fem_data('rho', mfls, rho)

# Compressibility
beta = ones_ls * ins.beta1
if ins.free_surface:
    beta[ls1.values(0) > 0] = ins.beta2
if ins.topography:
    beta[ls3.values(0) < 0] = ins.beta3
md.add_initialized_fem_data('beta', mfls, beta)

if ins.temp:
    # Thermal diffusivity
    kappa = ones_ls * ins.kappa1
    if ins.free_surface:
        kappa[ls1.values(0) > 0] = ins.kappa2
    if ins.topography:
        kappa[ls3.values(0) < 0] = ins.kappa3
    md.add_initialized_fem_data('kappa', mfls, kappa)
    md.add_initialized_data('cp', [ins.cp])

# Temperature
if ins.temp:
    md.add_initialized_data('vfta', [ins.vfta])
    md.add_initialized_data('vftb', [ins.vftb])
    md.add_initialized_data('vftc', [ins.vftc])

    # Initial temp
    if ins.restart:
        md.add_initialized_fem_data('Previous_t', mft, T)
        md.add_initialized_fem_data('Previous2_t', mft, Previous_T)
        T_init = Previous_T

    else:
        if (type(ins.T_init) is float) or (type(ins.T_init) is int):
            T_init = mft.eval('1') * ins.T_init
        elif type(ins.T_init) is str:
            T_init = eval(ins.T_init)
        elif type(ins.T_init) is type(None):
             T_init = mft.eval('1') * ins.T0
        if ins.free_surface:
            T_init[ls1_t > 0] = ins.T_atm
        if ins.topography:
            T_init[ls3_t < 0] = ins.T_atm

        md.add_initialized_fem_data('Previous_t', mft, T_init)
        md.add_initialized_fem_data('Previous2_t', mft, T_init)

    if ins.temp & ins.solidification:
        T_ls = sciinterp.griddata(D_t.transpose(), T_init, D_ls.transpose(), method='nearest')
        ls2.set_values((T_ls - Tg) / Tg)
        mls.adapt()

# viscosity
if ins.temp:
    T_p = sciinterp.griddata(D_t.transpose(), T_init, D_p.transpose(), method='nearest')
    T_ls = sciinterp.griddata(D_t.transpose(), T_init, D_ls.transpose(), method='nearest')
    eta_exp = ins.eta_exp.replace('vfta', str(ins.vfta)).replace('vftb', str(ins.vftb)).replace(
        'vftc', str(ins.vftc))

    eta = eval(eta_exp.replace('exp', 'np.exp').replace('T', 'T_ls'))
    eta[T_ls <= (ins.vftc - 273)] = ins.max_eta

    if ins.free_surface:
        eta[ls1.values(0) > 0] = ins.eta2
    if ins.topography:
        eta[ls3.values(0) < 0] = ins.eta3
else:
    eta = eval(ins.eta_exp.replace('exp', 'np.exp').replace('T', str(ins.T0)).replace('vft','ins.vft')) * ones_ls
    if ins.free_surface:
        eta[ls1.values(0) > 0] = ins.eta2
    if ins.topography:
        eta[ls3.values(0) < 0] = ins.eta3

# time dependence
md.add_initialized_data('dt', [ins.dt])
md.add_initialized_data('BDF0', [1])
md.add_initialized_data('BDF1', [-1])
md.add_initialized_data('BDF2', [0])

md_init.add_initialized_data('dt', [ins.dt])

# Elasticity
lam = -2/3*eta
mu = eta
solid = 0 * ones_ls

if ins.temp & ins.solidification:
    lam_solid = ins.E * ins.nu / ((1 + ins.nu) * (1 - 2 * ins.nu))
    mu_solid = ins.E / (2 * (1 + ins.nu))
    if ins.free_surface:
        lam[(ls1.values(0) <= 0) & (T_ls < Tg)] = lam_solid
        mu[(ls1.values(0) <= 0) & (T_ls < Tg)] = mu_solid
        solid[(ls1.values(0) <= 0) & (T_ls < Tg)] = 1
    else:
        lam[(T_ls < Tg)] = lam_solid
        mu[(T_ls < Tg)] = mu_solid
        solid[(T_ls < Tg)] = 1

md.add_initialized_fem_data('lambda', mfls, lam)
md.add_initialized_fem_data('mu', mfls, mu)
md.add_initialized_fem_data('solid', mfls, solid)

md_init.add_initialized_fem_data('lambda', mfls, lam)
md_init.add_initialized_fem_data('mu', mfls, mu)
md_init.add_initialized_fem_data('solid', mfls, solid)

if ins.free_surface | (ins.temp & ins.solidification) | ins.topography:
    mls_cut = mls.cut_mesh()
    radii = np.zeros(mls_cut.nbpts())
    D_cut = mls_cut.pts()
    for pid in mls_cut.pid():
        radii[pid] = np.min(mls_cut.convex_radius(mls_cut.cvid_from_pid(pid, share=True)))
    if ins.temp:
        radii_t = sciinterp.griddata(D_cut.transpose(), radii, D_t.transpose())

else:
    radii = np.mean(mesh.convex_radius()) * ones_ls
    if ins.temp:
        radii_t = np.mean(mesh.convex_radius()) * ones_t

# Add governing equations
# Add variables

if ins.free_surface | ins.topography:
    md.add_fem_variable('u', mfu_cut)
    md.add_fem_variable('p', mfp_cut)
    if ins.temp:
        md.add_fem_variable('t', mft_cut)
else:
    md.add_fem_variable('u', mfu)
    md.add_fem_variable('p', mfp)
    if ins.temp:
        md.add_fem_variable('t', mft)

md_init.add_fem_variable('u', mfu)
md_init.add_fem_variable('p', mfp)

md_init.add_linear_incompressibility_brick(mim_all, 'u', 'p')
md_init.add_isotropic_linearized_elasticity_brick(mim_all, 'u', 'lambda*(dt*solid + (1-solid))', 'mu*(dt*solid + (1-solid))')

md.add_macro('h', 'element_size')

# compressible mass balance
if ins.compressible:
    md.add_macro('rhof', 'rho_init*exp(beta*(p-p_init))')
    md.add_macro('Previous_rho', 'rho_init*exp(beta*(Previous_p-p_init))')
    md.add_macro('Previous2_rho', 'rho_init*exp(beta*(Previous2_p-p_init))')

    S_SUPG_p = "(u.Grad_p)*(u.Grad_Test_p)"
    tau_SUPG_p = 'pow(1/(2/dt + Norm(u)/(2*h)),-1/2)'

    md.add_nonlinear_term(mim, 'BDF0/dt*rhof*Test_p + Trace(Grad(rhof*u))*Test_p')
    md.add_source_term_brick(mim, 'p', '-(BDF1*Previous_rho+BDF2*Previous2_rho)/dt')
    if 'SUPG' in ins.stab_p:
        md.add_nonlinear_term(mim, tau_SUPG_p + '*' + S_SUPG_p)
        md.add_nonlinear_term(mim, tau_SUPG_p + '*' + '(BDF1*Previous_rho+BDF2*Previous2_rho)/dt.(u.Grad_Test_p)')

else:
    md.add_linear_incompressibility_brick(mim, 'u', 'p')

# mometum balance
time_int_u = "rho*((BDF0*u+BDF1*Previous_u+BDF2*Previous2_u)/dt.Test_u)"
linear_elastic = "(lambda*(Div_u*Div_Test_u) + mu*((Grad_u + Grad_u'):Grad_Test_u))*(dt/BDF0*solid + (1-solid))"
residual_stress1 = "-(lambda*(Div_Previous_d*Div_Test_u) + mu*((Grad_Previous_d + Grad_Previous_d'):Grad_Test_u))*(BDF1/BDF0)*solid"
residual_stress2 = "-(lambda*(Div_Previous2_d*Div_Test_u) + mu*((Grad_Previous2_d + Grad_Previous2_d'):Grad_Test_u))*(BDF2/BDF0)*solid"

if ins.steady:
    md.add_nonlinear_term(mim, 'solid*' + time_int_u)
else:
    md.add_nonlinear_term(mim, time_int_u)
md.add_nonlinear_term(mim, linear_elastic)
if ins.solidification:
    md.add_nonlinear_term(mim, residual_stress1 + residual_stress2)

if ins.temp:
    time_int = "(BDF0*t+BDF1*Previous_t+BDF2*Previous2_t)/dt"
    advection = "(u.Grad_t)*Test_t"
    diffusion = "kappa*(Grad_t.Grad_Test_t)"
    S_SUPG_t = "(u.Grad_t)*(u.Grad_Test_t) + kappa*(Grad_t).Grad(u.Grad_Test_t)"
    S_GLS_t = "(kappa*(Grad(u.Grad_t).Grad_Test_t))"
    md.add_macro('Pe', "h*Norm(u)/(2*kappa)")
    md.add_macro('xi', "min(Pe/3,1)")

    md.add_nonlinear_term(mim, time_int + '*Test_t')
    md.add_nonlinear_term(mim, advection)
    md.add_nonlinear_term(mim, diffusion)

    tau_SUPG_t = '1/(2/dt + 4*kappa/(h*h) + 2*Norm(u)/h)*xi'
    tau_GLS_t = '1/(2/dt + 4*kappa/(h*h) + 2*Norm(u)/h)*xi'
    if 'SUPG' in ins.stab_t:
        # tau_supg = 'h/(2*Norm(u))*(cosh(Norm(u)*h/(2*kappa))/sinh(Norm(u)*h/(2*kappa)) - 2*kappa/(h*Norm(u)))'
        # tau_SUPG_t = 'h*h/(4*kappa)*min(1/3,1/Pe)'
        md.add_nonlinear_term(mim, tau_SUPG_t + '*(u.Grad_Test_t)*' + time_int)
        md.add_nonlinear_term(mim, tau_SUPG_t + '*' + S_SUPG_t)

    elif 'GLS' in ins.stab_t:
        md.add_nonlinear_term(mim, tau_GLS_t + '*(u.Grad_Test_t)*' + time_int)
        md.add_nonlinear_term(mim, tau_GLS_t + '*' + S_SUPG_t)
        md.add_nonlinear_term(mim, tau_GLS_t + '*' + S_GLS_t)

# add body force
if (type(ins.f_x) is type(None)) and (type(ins.f_y) is type(None)):
    f_xi = None
    f_yi = None
else:
    if (type(ins.f_x) is float) or (type(ins.f_x) is int):
        f_xi = ones_ls * ins.f_x
    elif type(ins.f_x) is str:
        f_xi = eval(ins.f_x.replace('y', 'y_ls').replace('x', 'x_ls'))
    elif type(ins.f_x) is type(None):
        f_xi = ones_ls * 0

    if (type(ins.f_y) is float) or (type(ins.f_y) is int):
        f_yi = ones_ls * ins.f_y
    elif type(ins.f_y) is str:
        f_yi = eval(ins.f_y.replace('y', 'y_ls').replace('x', 'x_ls'))
    elif type(ins.f_y) is type(None):
        f_yi = ones_ls * 0

    md.add_initialized_fem_data('body', mfls, [f_xi, f_yi])
    md.add_source_term_brick(mim, 'u', 'body')

    md_init.add_initialized_fem_data('body', mfls, [f_xi, f_yi])
    md_init.add_source_term_brick(mim_all, 'u', 'body')

# add boundary conditions
for i, bound in enumerate(bounds):

    # Dirichlet boundaries
    dirichlet = False
    if (type(eval('ins.' + bound + '_ux')) is float) or (type(eval('ins.' + bound + '_ux')) is int):
        data_ux = eval('ins.' + bound + '_ux') * ones_p
        dirichlet = True
    if type(eval('ins.' + bound + '_ux')) is str:
        data_ux = eval(eval('ins.' + bound + '_ux'))
        dirichlet = True
    if type(eval('ins.' + bound + '_uy')) is type(None):
        data_uy = 0 * ones_p

    if (type(eval('ins.' + bound + '_uy')) is float) or (type(eval('ins.' + bound + '_uy')) is int):
        data_uy = eval('ins.' + bound + '_uy') * ones_p
        dirichlet = True
    if type(eval('ins.' + bound + '_uy')) is str:
        data_uy = eval(eval('ins.' + bound + '_uy'))
        dirichlet = True
    if type(eval('ins.' + bound + '_ux')) is type(None):
        data_ux = 0 * ones_p

    if dirichlet:
        md.add_initialized_fem_data(bound + 'data', mfp, [data_ux, data_uy])
        md_init.add_initialized_fem_data(bound + 'data', mfp, [data_ux, data_uy])

        md.add_Dirichlet_condition_with_multipliers(mim, 'u', 2, i + 1,
                                                    dataname=bound + 'data')
        # md.add_Dirichlet_condition_with_simplification('u', i+1,
        #                                  dataname = bound+'data');
        md_init.add_Dirichlet_condition_with_multipliers(mim_all, 'u', 2, i + 1,
                                                         dataname=bound + 'data')

    else:
        # Neumann boundaries
        if (type(eval('ins.' + bound + '_udx')) is float) or (type(eval('ins.' + bound + '_udx')) is int):
            data_udx = eval('ins.' + bound + '_udx') * ones_p
        if type(eval('ins.' + bound + '_udx')) is str:
            data_udx = eval(eval('ins.' + bound + '_udx'))
        if type(eval('ins.' + bound + '_udy')) is type(None):
            data_udy = 0 * ones_p

        if (type(eval('ins.' + bound + '_udy')) is float) or (type(eval('ins.' + bound + '_udy')) is int):
            data_udy = eval('ins.' + bound + '_udy') * ones_p
        if type(eval('ins.' + bound + '_udy')) is str:
            data_udy = eval(eval('ins.' + bound + '_udy'))
        if type(eval('ins.' + bound + '_udx')) is type(None):
            data_udx = 0 * ones_p

        md.add_initialized_fem_data(bound + 'data', mfp, [data_udx, data_udy])
        md_init.add_initialized_fem_data(bound + 'data', mfp, [data_udx, data_udy])

        md.add_source_term_brick(mim, 'u', bound + 'data', region=i + 1)
        md_init.add_source_term_brick(mim_all, 'u', bound + 'data', region=i + 1)

# add pressure on boundary or free surface
md.add_initialized_fem_data('patm', mfp, [ins.p_atm * ones_p])
md_init.add_initialized_fem_data('patm', mfp, ins.p_atm * ones_p)
if ins.solve_air | (not ins.free_surface):
    md.add_Dirichlet_condition_with_multipliers(mim, 'p', 1, 3, dataname='patm')
    md_init.add_Dirichlet_condition_with_multipliers(mim_all, 'p', 1, 3, dataname='patm')

else:
    md.add_Dirichlet_condition_with_multipliers(mim_surf, 'p', 1, -1, dataname='patm')
    md_init.add_Dirichlet_condition_with_multipliers(mim_surf, 'p', 1, -1, dataname='patm')

if ins.topography:
    if 'no_slip' in ins.basal_velocity:
        md.add_initialized_fem_data('no_slip', mfp, [0 * ones_p, 0 * ones_p])
        md.add_Dirichlet_condition_with_multipliers(mim_base, 'u', 2, -1, dataname='no_slip')

        md_init.add_initialized_fem_data('no_slip', mfp, [0 * ones_p, 0 * ones_p])
        md_init.add_Dirichlet_condition_with_multipliers(mim_base, 'u', 2, -1, dataname='no_slip')

    if 'no_normal' in ins.basal_velocity:
        md.add_initialized_fem_data('no_normal', mfp, [0 * ones_p, 0 * ones_p])
        md.add_normal_source_term_brick(mim_base, 'u', -1, dataname='no_normal')

        md_init.add_initialized_fem_data('no_normal', mfp, [0 * ones_p, 0 * ones_p])
        md_init.add_normal_source_term_brick(mim_base, 'u', -1, dataname='no_normal')

if ins.temp:
    if ins.solve_air | (not ins.free_surface):
        for i, bound in enumerate(bounds):
            dirichlet = False
            # Dirichlet boundaries
            if (type(eval('ins.' + bound + '_T')) is float) or (type(eval('ins.' + bound + '_T')) is int):
                data_t = eval('ins.' + bound + '_T') * ones_t
                dirichlet = True
            if type(eval('ins.' + bound + '_T')) is str:
                data_t = eval(eval('ins.' + bound + '_T'))
                dirichlet = True

            if dirichlet:
                md.add_initialized_fem_data(bound + 'data_t', mft, [data_t])
                md.add_Dirichlet_condition_with_multipliers(mim, 't', 2, i + 1,
                                                            dataname=bound + 'data_t')

            # Neumann boundaries
            else:
                if (type(eval('ins.' + bound + '_flux_T')) is float) or (type(eval('ins.' + bound + '_flux_T')) is int):
                    data_t = -eval('ins.' + bound + '_flux_T') * ones_ls * kappa
                if type(eval('ins.' + bound + '_flux_T')) is str:
                    data_t = -eval(eval('ins.' + bound + '_flux_T')) * kappa
                md.add_initialized_fem_data(bound + 'data_t', mfls, [data_t, data_t])
                md.add_normal_source_term_brick(mim, 't', bound + 'data_t', i + 1)

    # add temperature or flux to free surface (default to temperature if provided)
    # fixed temperature
    elif (type(ins.surface_temp) is float) or (type(ins.surface_temp) is int):
        md.add_initialized_fem_data('surface_temp', mft, [ins.surface_temp * ones_t])
        md.add_Dirichlet_condition_with_multipliers(mim_surf, 't', 2, -1, dataname='surface_temp')
    # fixed temperature flux
    elif (type(ins.surface_flux) is float) or (type(ins.surface_flux) is int):
        md.add_initialized_fem_data('surface_flux', mft, [ins.surface_flux * ones_t])
        md.add_source_term_brick(mim_surf, 't', 'surface_flux/Previous_rho/cp', -1)
    # dynamic surface flux
    elif type(ins.surface_flux) is str:
        surface_flux_budget = ones_t * 0
        if 'radiation' in ins.surface_flux:
            surface_flux_budget += radii_t / ins.rho1 / ins.cp * ins.emissivity * ins.stefan_boltzman * ins.crust_cover * (
                        T_init ** 4 - ins.T_atm ** 4)
        if 'forced convection' in ins.surface_flux:
            surface_flux_budget += ins.heat_transfer_coeff * (T_init - ins.T_atm)
        md.add_initialized_fem_data('surface_flux', mft, [surface_flux_budget * ones_t])
        md.add_source_term_brick(mim_surf, 't', 'surface_flux', -1)

    if ins.topography:
        # add temperature or flux to flow base (default to temperature if provided)
        if (type(ins.basal_temp) is float) or (type(ins.basal_temp) is int):
            md.add_initialized_fem_data('basal_temp', mft, [ins.basal_temp * ones_t])
            md.add_Dirichlet_condition_with_multipliers(mim_base, 't', 2, -1, dataname='basal_temp')
        elif (type(ins.basal_flux) is float) or (type(ins.basal_flux) is int):
            md.add_initialized_fem_data('basal_flux', mft, [ins.basal_flux * ones_t])
            md.add_source_term_brick(mim_base, 't', 'basal_flux/Previous_rho/cp', -1)
        elif type(ins.basal_flux) is str:
            if 'conduction' in ins.basal_flux:
                md.add_initialized_fem_data('basal_flux', mft, [ins.kappa1 * (T_init - ins.basal_temp_i) / radii_t])
                md.add_source_term_brick(mim_base, 't', 'basal_flux', -1)

# influx
if ins.influx:
    md.add_initialized_fem_data('influxdata', mfp, [eval(ins.influx_ux) * ones_p, eval(ins.influx_uy) * ones_p])
    md.add_Dirichlet_condition_with_multipliers(mim, 'u', 2, 5,
                                                dataname='influxdata')
    if ins.temp:
        md.add_initialized_fem_data('influxdata_t', mft, [eval(ins.influx_T) * ones_t])
        md.add_Dirichlet_condition_with_multipliers(mim, 't', 2, 5,
                                                    dataname='influxdata_t')

#### Solve ####
print('Starting solve')
if not ins.restart:
    # initialize pressure and velocity with incompressible, steady stokes
    md_init.solve('max_res', 1E-10, 'max_iter', 100, 'noisy')
    p_init = md_init.variable('p')
    u_init = md_init.variable('u')

    md.set_variable('Previous_p', p_init)
    md.set_variable('Previous2_p', p_init)
    md.set_variable('Previous_u', u_init)
    md.set_variable('Previous2_u', u_init)

    print('time = 0')
    # bootstrap
    md.solve('max_res', 1E-10, 'max_iter', 100, 'noisy')

    if ins.free_surface:
        Ls1 = md.variable('ls1')
        # Previous_ls = md.variable('Previous_ls')

        # construct extension velocities
        # speed on the interface is U.Grad(Psi)
        Grad_Ls = compute_gradient(mfls, Ls1, mfls)
        Grad_mag = np.sqrt(Grad_Ls[0, :] ** 2 + Grad_Ls[1, :] ** 2)
        Grad_Ls[:, Grad_mag > 0] = Grad_Ls[:, Grad_mag > 0] / Grad_mag[Grad_mag > 0]

        u = md.variable('u')

        u_ls = sciinterp.griddata(mfu.basic_dof_nodes()[:, eval(ind_u)][:, ::2].transpose(), u[::2], D_ls.transpose(),
                                  method='nearest')
        v_ls = sciinterp.griddata(mfu.basic_dof_nodes()[:, eval(ind_u)][:, 1::2].transpose(), u[1::2], D_ls.transpose(),
                                  method='nearest')

        F_ext = sciinterp.griddata(D_ls.transpose(), u_ls * Grad_Ls[0, :] + v_ls * Grad_Ls[1, :],
                                   np.array([x_grid.flatten(), y_grid.flatten()]).transpose(), method='nearest')
        Ls_ext = sciinterp.griddata(D_ls.transpose(), Ls1,
                                    np.array([x_grid.flatten(), y_grid.flatten()]).transpose(), method='nearest')

        d, F_ext = skfmm.extension_velocities(Ls_ext.reshape((2 * ins.ny, 2 * ins.nx)),
                                              F_ext.reshape((2 * ins.ny, 2 * ins.nx)),
                                              dx=[dy / 2, dx / 2])
        F_ext = sciinterp.griddata(np.array([x_grid.flatten(), y_grid.flatten()]).transpose(), F_ext.flatten(),
                                   D_ls.transpose(),
                                   method='nearest').flatten()
        d = sciinterp.griddata(np.array([x_grid.flatten(), y_grid.flatten()]).transpose(), d.flatten(), D_ls.transpose(),
                               method='nearest').flatten()
        # md.set_variable('fext',F_ext.flatten())

        compute_convect(mfls, Ls1, mfls, F_ext * Grad_Ls, ins.dt, 100)
        md.set_variable('ls1', Ls1)

    Previous_u = u_init
    U = ones_u * 0
    U[eval(ind_u)] = md.variable('u')

    Previous_p = p_init
    P = ones_p * 0
    P[eval(ind_p)] = md.variable('p')

    Previous_d = d_init
    D = ones_u * 0
    solid_u = sciinterp.griddata(D_ls.transpose(), solid, D_u.transpose(), method='nearest')
    D[eval(ind_u)] = (Previous_d[eval(ind_u)] + ins.dt*md.variable('u'))*solid_u
    Previous_d = ones_u * D

    if ins.temp:
        Previous_T = T_init
        T = ones_t * ins.T_atm
        T[eval(ind_t)] = md.variable('t')

        T_ls = sciinterp.griddata(D_t.transpose(), T, D_ls.transpose(), method='nearest')
        if ins.solidification:
            ls2.set_values((T_ls - Tg) / Tg)

    if ins.vtk:
        u = md.variable('u')
        P = md.variable('p')

        mfu.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_u_' + '0'*ndigits + '.vtk', u_init)
        mfp.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_P_' + '0'*ndigits + '.vtk', p_init)
        if ins.free_surface:
            mfls.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_Ls1_' + '0'*ndigits + '.vtk', Ls1)
            mfls.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_rho_' + '0' * ndigits + '.vtk', rho)
        if ins.temp:
            T = md.variable('t')
            mft.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_T_' + '0'*ndigits + '.vtk', T_init)
            mfls.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_mu_' + '0'*ndigits + '.vtk', mu)
            if ins.solidification:
                mfu.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_d_' + '0'*ndigits + '.vtk', d_init)
                mft.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_Ls2_' + '0' * ndigits + '.vtk', (T_init-Tg)/Tg)
        if (ins.noutput==1):
            numstr = str(ins.dt * 10 ** ndecimal).split('.')[0].zfill(ndigits)

            mfu.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_u_' + numstr + '.vtk', U)
            mfp.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_P_' + numstr + '.vtk', P)
            if ins.free_surface:
                mfls.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_Ls1_' + '0' * ndigits + '.vtk', Ls1)
                mfls.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_rho_' + '0' * ndigits + '.vtk', rho)
            if ins.temp:
                T = md.variable('t')
                mft.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_T_' + numstr + '.vtk', T)
                mfls.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_mu_' + numstr + '.vtk', mu)
                if ins.solidification:
                    mfu.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_d_' + numstr + '.vtk', D)
                    mft.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_Ls2_' + '0' * ndigits + '.vtk',
                                      (T_init - Tg) / Tg)

# update BDF coefficients
BDF0 = 3 / 2
BDF1 = -2
BDF2 = 1 / 2

md.set_variable('BDF0', [BDF0])
md.set_variable('BDF1', [BDF1])
md.set_variable('BDF2', [BDF2])

if ins.restart:
    tstart = last_ti
else:
    tstart = ins.dt
    hf = h5py.File(ins.outfile + '/' + ins.outfile.split('/')[-1] + '.h5', 'w')
    hf.create_dataset('last_u',data=U)
    hf.create_dataset('last2_u',data=Previous_u)
    hf.create_dataset('last_p',data=P)
    hf.create_dataset('last2_p',data=Previous_p)
    hf.create_dataset('last_d',data=D)
    hf.create_dataset('last2_d',data=Previous_d)
    if ins.temp:
        hf.create_dataset('last_T',data=T)
        hf.create_dataset('last2_T',data=Previous_T)
    if ins.free_surface:
        hf.create_dataset('last_Ls1',data=Ls1)
    if ins.temp & ins.solidification:
        hf.create_dataset('last_Ls2',data=(T_ls-Tg)/Tg)
    if ins.topography:
        hf.create_dataset('last_Ls3',data=ls3.values(0))
    hf.create_dataset('last_ti',data=[0])
    hf.close()

# Main loop 
for i, ti in enumerate(np.arange(tstart, ins.tf, ins.dt)):
    if ins.restart:
        i+=1

    md.set_variable('Previous2_u', Previous_u)
    md.set_variable('Previous_u', U)

    md.set_variable('Previous2_p', Previous_p)
    md.set_variable('Previous_p', P)

    md.set_variable('Previous2_d', Previous_d)
    md.set_variable('Previous_d', D)

    if ins.free_surface:
        ls1.set_values(Ls1)
    if ins.free_surface | (ins.temp & ins.solidification) | ins.topography:
        mls.adapt()
        mim.adapt()
        mim_integ.adapt()
    if ins.free_surface:
        mim_surf.adapt()

        if ins.temp:
            mls_cut = mls.cut_mesh()
            radii = np.zeros(mls_cut.nbpts())
            D_cut = mls_cut.pts()
            for pid in mls_cut.pid():
                radii[pid] = np.min(mls_cut.convex_radius(mls_cut.cvid_from_pid(pid, share=True)))
            radii_t = sciinterp.griddata(D_cut.transpose(), radii, D_t.transpose())
            # md.set_variable('h',radii_t)
            # ls1_t = compute_interpolate_on(mfls,ls1.values(0),mft)

    if ins.free_surface | ins.topography:
        mfp_cut.set_partial(eval(ind_p))
        mfu_cut.set_partial(eval(ind_u))
        if ins.temp:
            mft_cut.set_partial(eval(ind_t))

    # density
    if ins.free_surface:
        rho[(ls1.values(0) <= 0)] = ins.rho1
        rho[(ls1.values(0) > 0)] = ins.rho2
    if ins.topography:
        rho[ls3.values(0) < 0] = ins.rho3
    md.set_variable('rho', rho)

    if ins.free_surface:
        beta[(ls1.values(0) <= 0)] = ins.beta1
        beta[(ls1.values(0) > 0)] = ins.beta2
    if ins.topography:
        beta[ls3.values(0) < 0] = ins.beta3
    md.set_variable('beta', beta)

    if ins.temp:
        # thermal diffusivity
        if ins.free_surface:
            kappa[(ls1.values(0) <= 0)] = ins.kappa1
            kappa[(ls1.values(0) > 0)] = ins.kappa2
        if ins.topography:
            kappa[ls3.values(0) < 0] = ins.kappa3
        md.set_variable('kappa', kappa)

        # shift_variables_for_time_integration
        md.set_variable('Previous2_t', Previous_T)
        md.set_variable('Previous_t', T)

        # update viscosity field
        T_p = sciinterp.griddata(D_t.transpose(), T, D_p.transpose(), method='nearest')
        T_ls = sciinterp.griddata(D_t.transpose(), T, D_p.transpose(), method='nearest')
        eta = eval(ins.eta_exp.replace('exp', 'np.exp').replace('T', 'T_ls').replace('vft','ins.vft'))
        eta[eta > ins.max_eta] = ins.max_eta
        eta[T_ls <= (ins.vftc - 273)] = ins.max_eta

        T_ls = sciinterp.griddata(D_t.transpose(), T, D_ls.transpose(), method='nearest')
        if ins.temp & ins.solidification:
            ls2.set_values((T_ls - Tg) / Tg)

    else:
        eta = eval(ins.eta_exp.replace('exp', 'np.exp').replace('T', str(ins.T0)).replace('vft','ins.vft')) * ones_ls

    if ins.free_surface:
        eta[(ls1.values(0) > 0)] = ins.eta2
    if ins.topography:
        eta[ls3.values(0) < 0] = ins.eta3
    mu = eta

    lam = -2/3*eta
    solid = 0 * ones_ls
    if ins.temp & ins.solidification:
        # elasticity
        if ins.free_surface:
            lam[(ls1.values(0) <= 0) & (ls2.values(0) > 0)] = lam_solid
            mu[(ls1.values(0) <= 0) & (ls2.values(0) > 0)] = mu_solid
            solid[(ls1.values(0) <= 0) & (ls2.values(0) > 0)] = 1
        else:
            lam[(T_ls < Tg)] = lam_solid
            mu[(T_ls < Tg)] = mu_solid
            solid[(T_ls < Tg)] = 1

        md.set_variable('solid', solid)

    md.set_variable('lambda', lam)
    md.set_variable('mu', mu)

    # update body force
    if (type(ins.f_x) is type(None)) and (type(ins.f_y) is type(None)):
        f_xi = None
        f_yi = None
    else:
        if (type(ins.f_x) is float) or (type(ins.f_x) is int):
            f_xi = ones_ls * ins.f_x
        elif type(ins.f_x) is str:
            f_xi = eval(ins.f_x.replace('y', 'y_ls').replace('x', 'x_ls'))
        elif type(ins.f_x) is type(None):
            f_xi = ones_ls * 0

        if (type(ins.f_y) is float) or (type(ins.f_y) is int):
            f_yi = ones_ls * ins.f_y
        elif type(ins.f_y) is str:
            f_yi = eval(ins.f_y.replace('y', 'y_ls').replace('x', 'x_ls'))
        elif type(ins.f_y) is type(None):
            f_yi = ones_ls * 0

        md.set_variable('body', [f_xi, f_yi])

            # update surface flux
    if ins.free_surface:
        if type(ins.surface_flux) is str:
            surface_flux_budget = ones_t * 0
            if 'radiation' in ins.surface_flux:
                surface_flux_budget += radii_t / ins.rho1 / ins.cp * ins.emissivity * ins.stefan_boltzman * ins.crust_cover * (
                            T ** 4 - ins.T_atm ** 4)
            if 'forced convection' in ins.surface_flux:
                surface_flux_budget += ins.heat_transfer_coeff * (T - ins.T_atm)
            md.set_variable('surface_flux', [surface_flux_budget * ones_t])

    if ins.topography:
        if type(ins.basal_flux) is str:
            if 'conduction' in ins.basal_flux:
                md.set_variable('basal_flux', [ins.kappa1 * (T - ins.basal_temp_i) / radii_t * ones_t])

    # Solve
    md.solve('max_res', 1E-10, 'max_iter', 100, 'noisy')

    if ins.free_surface:
        # construct extension velocities
        # speed on the interface is U.Grad(Psi)
        Grad_Ls = compute_gradient(mfls, Ls1, mfls)
        Grad_mag = np.sqrt(Grad_Ls[0, :] ** 2 + Grad_Ls[1, :] ** 2)
        Grad_Ls[:, Grad_mag > 0] = Grad_Ls[:, Grad_mag > 0] / Grad_mag[Grad_mag > 0]

        u = md.variable('u')

        u_ls = sciinterp.griddata(mfu.basic_dof_nodes()[:, eval(ind_u)][:, ::2].transpose(), u[::2], D_ls.transpose(),
                                  method='nearest', fill_value=0)
        v_ls = sciinterp.griddata(mfu.basic_dof_nodes()[:, eval(ind_u)][:, 1::2].transpose(), u[1::2], D_ls.transpose(),
                                  method='nearest', fill_value=0)

        F_ext = sciinterp.griddata(D_ls.transpose(), u_ls * Grad_Ls[0, :] + v_ls * Grad_Ls[1, :],
                                   np.array([x_grid.flatten(), y_grid.flatten()]).transpose(),
                                   method='nearest', fill_value=0)
        Ls_ext = sciinterp.griddata(D_ls.transpose(), Ls1,
                                    np.array([x_grid.flatten(), y_grid.flatten()]).transpose(),
                                    method='nearest', fill_value=1)

        d, F_ext = skfmm.extension_velocities(Ls_ext.reshape((2 * ins.ny, 2 * ins.nx)), F_ext.reshape((2 * ins.ny, 2 * ins.nx)),
                                              dx=[dy / 2, dx / 2])
        F_ext = sciinterp.griddata(np.array([x_grid.flatten(), y_grid.flatten()]).transpose(), F_ext.flatten(),
                                   D_ls.transpose(),
                                   method='nearest').flatten()
        d = sciinterp.griddata(np.array([x_grid.flatten(), y_grid.flatten()]).transpose(), d.flatten(),
                               D_ls.transpose(),
                               method='nearest').flatten()
        compute_convect(mfls, Ls1, mfls, F_ext * Grad_Ls, ins.dt, 100)
        md.set_variable('ls1', Ls1)

    Previous_u = U
    U = ones_u * 0
    U[eval(ind_u)] = md.variable('u')

    Previous_p = P
    P = ones_p * 0
    P[eval(ind_p)] = md.variable('p')

    Previous2_d = Previous_d
    Previous_d = D
    D = ones_u * 0
    solid_u = sciinterp.griddata(D_ls.transpose(), solid, D_u.transpose(), method='nearest')
    D[eval(ind_u)] = (- BDF2 / BDF0 * Previous2_d[eval(ind_u)] - BDF1 / BDF0 * Previous_d[
        eval(ind_u)] + ins.dt / BDF0 * md.variable('u')) * solid_u

    if ins.temp:
        Previous_T = T
        T = ones_t * ins.T_atm
        T[eval(ind_t)] = md.variable('t')

    if ((i + 1) % ins.noutput == 0) or (np.abs(ti-ins.tf)<ins.dt):
        print('Time = %g' % ti)
        numstr = str(ti * 10 ** ndecimal).split('.')[0].zfill(ndigits)
        # print('Average temperature %g' % np.mean(T))

        if ins.vtk:
            u = md.variable('u')
            P = md.variable('p')

            mfu.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_u_' + numstr + '.vtk', u)
            mfp.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_P_' + numstr + '.vtk', P)
            if ins.free_surface:
                mfls.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_Ls1_' + numstr + '.vtk', Ls1)
                mfls.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_rho_' + numstr + '.vtk', rho)
            if ins.temp:
                T = md.variable('t')
                mft.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_T_' + numstr + '.vtk', T)
                mfp.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_mu_' + numstr + '.vtk', mu)
                if ins.solidification:
                    mfu.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_d_' + numstr + '.vtk', D)
                    mft.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_Ls2_' + numstr + '.vtk', (T-Tg)/Tg)

        hf = h5py.File(ins.outfile + '/' + ins.outfile.split('/')[-1] + '.h5','a')
        hf['last_u'][:] = U
        hf['last2_u'][:] = Previous_u
        hf['last_p'][:] = P
        hf['last2_p'][:] = Previous_p
        hf['last_d'][:] = D
        hf['last2_d'][:] = Previous_d
        if ins.temp:
            hf['last_T'][:] = T
            hf['last2_T'][:] = Previous_T
        if ins.free_surface:
            hf['last_Ls1'][:] = Ls1
        if ins.temp & ins.solidification:
            hf['last_Ls2'][:] = (T_ls - Tg) / Tg
        if ins.topography:
            hf['last_Ls3'][:] = ls3.values(0)
        hf['last_ti'][:] = [ti]
        hf.close()

if ins.free_surface:
    ls1.set_values(Ls1)
if ins.free_surface | (ins.temp & ins.solidification) | ins.topography:
    mls.adapt()

if ins.temp & ins.solidification:
    T_ls = sciinterp.griddata(D_t.transpose(), T, D_ls.transpose(), method='nearest')
    ls2.set_values((T_ls - Tg) / Tg)

if ins.free_surface | ins.topography:
    mfp_cut.set_partial(eval(ind_p))
    mfu_cut.set_partial(eval(ind_u))
    if ins.temp:
        mft_cut.set_partial(eval(ind_t))

# Reshape for output
# Velocity
u = md.variable('u')
P = md.variable('p')
if ins.free_surface:
    Ls1 = md.variable('ls1')
if ins.temp:
    T = md.variable('t')
else:
    T = None
# %% md
# Visualize results
# %%
if ins.free_surface & ins.solve_air:
    ind_u_plot = eval(ind_u)
    ind_p_plot = eval(ind_p)
    if ins.temp:
        ind_t_plot = eval(ind_t)

else:
    ind_u_plot = eval(ind_u)
    ind_p_plot = eval(ind_p)
    if ins.temp:
        ind_t_plot = eval(ind_t)

if ins.plots:
    fig2, ax = plt.subplots(figsize=(16, 16), ncols=2, nrows=2)

    c1 = ax[0, 0].tripcolor(x_u[eval(ind_u)][ind_u_plot][::2], y_u[eval(ind_u)][ind_u_plot][::2], u[ind_u_plot][::2],
                            cmap='RdBu_r', shading='gouraud',
                            vmin=-np.max(np.abs(u[ind_u_plot][::2])), vmax=np.max(np.abs(u[ind_u_plot][::2])))
    c2 = ax[0, 1].tripcolor(x_u[eval(ind_u)][ind_u_plot][1::2], y_u[eval(ind_u)][ind_u_plot][1::2], u[ind_u_plot][1::2],
                            cmap='RdBu_r', shading='gouraud',
                            vmin=-np.max(np.abs(u[ind_u_plot][1::2])), vmax=np.max(np.abs(u[ind_u_plot][1::2])))
    c3 = ax[1, 0].tripcolor(x_p[eval(ind_p)][ind_p_plot], y_p[eval(ind_p)][ind_p_plot], P[ind_p_plot],
                            cmap='RdBu_r',
                            shading='gouraud')  # ,vmin=-1000,vmax=1000)#,vmin=0.999*np.mean(P),vmax=1.001*np.mean(P))
    if ins.temp:
        c4 = ax[1, 1].tripcolor(x_t[eval(ind_t)][ind_t_plot], y_t[eval(ind_t)][ind_t_plot], T[ind_t_plot],
                                cmap='RdBu_r', shading='gouraud', vmin=0, vmax=ins.T0)
        ax[1, 1].set_title('T', fontsize=18)
        plt.colorbar(c4, ax=ax[1, 1], shrink=0.5, label='Temperature ($^\circ$C)')

    elif ins.free_surface:
        c4 = ax[1, 1].tripcolor(x_ls, y_ls, Ls1, cmap='RdBu_r', shading='gouraud', vmin=-1, vmax=1)
        ax[1, 1].set_title('Level Set', fontsize=18)
        ax[1, 1].scatter(x_u[eval(ind_u)], y_u[eval(ind_u)], 5, 'k')
        plt.colorbar(c4, ax=ax[1, 1], shrink=0.5, label='Distance (m)')

    plt.colorbar(c1, ax=ax[0, 0], shrink=0.5, label='Horizontal velocity (m/s)')
    plt.colorbar(c2, ax=ax[0, 1], shrink=0.5, label='Vertical velocity (m/s)')
    plt.colorbar(c3, ax=ax[1, 0], shrink=0.5, label='$\Delta$ Pressure (Pa)')

    for i, axi in enumerate(ax.reshape(4, 1)):
        axi[0].set_xlabel('X', fontsize=14)
        axi[0].set_ylabel('Y', fontsize=14)
        axi[0].set_aspect('equal')
        if ins.free_surface:
            axi[0].tricontour(x_ls, y_ls, ls1.values(0), levels=[0], colors='k')

        if ins.temp & ins.solidification:
            axi[0].tricontour(x_ls, y_ls, ls2.values(0) - 1000 * (ls2.values(0) >= 0), levels=[0], colors='k',
                              linestyles='--')
    ax[0, 0].set_title('U', fontsize=18)
    ax[0, 1].set_title('V', fontsize=18)
    ax[1, 0].set_title('P', fontsize=18)

    fig2.savefig(outfile + '/' + ins.outfile.split('/')[-1] + '_solution.pdf')