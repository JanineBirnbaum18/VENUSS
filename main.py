import sys
import os
import json
from types import SimpleNamespace
import h5py

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate as sciinterp
import skfmm
import getfem as gf
from getfem import *

outfile = './Results/spreading_drop_iso'

# import simulation parameters
with open(outfile + '/' + outfile.split('/')[-1] + '.json', 'r') as openfile:
    dictionary = json.load(openfile)

ins = SimpleNamespace(**dictionary)
try:
    ins.ls_k
except Exception as e:
    ins.ls_k = 1

try:
    ins.nmat
except Exception as e:
    ins.nmat = 1

try:
    ins.visc_coeff
except Exception as e:
    ins.visc_coeff = 0.01

dx = ins.L_x/(ins.nx*(ins.ls_k))
dy = ins.L_y/(ins.ny*(ins.ls_k))
xs = np.linspace(0, ins.L_x, ins.nx*(ins.ls_k)+1)
ys = np.linspace(0, ins.L_y, ins.ny*(ins.ls_k)+1)
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
    fb_bottomleft = mesh.faces_from_cvid(mesh.convexes_in_box((-dx, -dy),
                                                          (dx, dy)))

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

mesh.set_region(5, eval('fb_bottomleft'))

if ins.influx:
    mesh.set_region(6, fb_influx)

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
mff = gf.MeshFem(mesh, 2)
mff.set_classical_fem(ins.ls_k)

#mflss = gf.MeshFem(mesh, 1)
#mflss.set_classical_fem(ins.ls_k+1)

mfmat = gf.MeshFem(mesh,1)
mfmat.set_classical_fem(ins.nmat)

#mflsgrad = gf.MeshFem(mesh,1)
#mflsgrad.set_classical_discontinuous_fem(ins.ls_k-1)

#mflssgrad = gf.MeshFem(mesh,1)
#mflssgrad.set_classical_discontinuous_fem(ins.ls_k)

#mflssgrad2 = gf.MeshFem(mesh,1)
#mflssgrad2.set_classical_discontinuous_fem(np.max([ins.ls_k-1,0]))

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

D_f = mff.basic_dof_nodes()
ones_f = np.ones(D_f.shape[1])

D_mat = mfmat.basic_dof_nodes()
ones_mat = np.ones(D_mat.shape[1])

#D_lsgrad = mflsgrad.basic_dof_nodes()
#ones_lsgrad = np.ones(D_lsgrad.shape[1])
#x_lsgrad = D_lsgrad[0, :]
#y_lsgrad = D_lsgrad[1, :]
#if ins.ndim == 2:
#    edges_lsgrad = np.where((x_lsgrad <= dx) | (x_lsgrad >= (ins.L_x - dx)) | (y_lsgrad <= dy) | (y_lsgrad >= (ins.L_y - dy)))

#D_lsgrad2 = mflssgrad2.basic_dof_nodes()
#ones_lsgrad2 = np.ones(D_lsgrad2.shape[1])

if ins.temp:
    D_t = mft.basic_dof_nodes()
    ones_t = np.ones(D_t.shape[1])
    x_t = D_t[0, :]
    y_t = D_t[1, :]

if ins.free_surface:
    Ls1 = ls1.values(0)

    # force to signed distance function
    #Ls_ext = sciinterp.griddata(D_ls.transpose(), Ls1,
    #                            np.array([x_grid.flatten(), y_grid.flatten()]).transpose(), method='nearest')
    #Ls1 = skfmm.distance(Ls_ext.reshape((x_grid.shape)), dx=[dy, dx]).flatten()
    #Ls1 = sciinterp.griddata(np.array([x_grid.flatten(), y_grid.flatten()]).transpose(), Ls1,
    #                         D_ls.transpose(), method='nearest')
    #ls1.set_values(Ls1)
    #mls.adapt()
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
    md.add_initialized_fem_data('Previous_psi', mfls, Ls1)
    md.add_initialized_fem_data('Previous2_psi', mfls, Previous_Ls)
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
    md.add_initialized_fem_data('Previous_psi', mfls, ls1.values(0))
    md.add_initialized_fem_data('Previous2_psi', mfls, ls1.values(0))

md.add_initialized_fem_data('fext',mff,0*ones_f)

# Density
if ins.free_surface:
    ls1_mat = compute_interpolate_on(mfls,ls1.values(0),mfmat)
if ins.topography:
    ls3_mat = compute_interpolate_on(mfls,ls3.values(0),mfmat)
rho = ones_mat * ins.rho1
if ins.free_surface:
    rho[ls1_mat > 0] = ins.rho2
if ins.topography:
    rho[ls3_mat < 0] = ins.rho3
md.add_initialized_fem_data('rho', mfmat, rho)
md.add_initialized_fem_data('rho_init', mfmat, rho)

md_init.add_initialized_fem_data('rho', mfmat, rho)

# Compressibility
beta = ones_mat * ins.beta1
if ins.free_surface:
    beta[ls1_mat > 0] = ins.beta2
if ins.topography:
    beta[ls3_mat < 0] = ins.beta3
md.add_initialized_fem_data('beta', mfmat, beta)

if ins.temp:
    # Thermal diffusivity
    kappa = ones_mat * ins.kappa1
    if ins.free_surface:
        kappa[ls1_mat > 0] = ins.kappa2
    if ins.topography:
        kappa[ls3_mat < 0] = ins.kappa3
    md.add_initialized_fem_data('kappa', mfmat, kappa)
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
            ls1_t = compute_interpolate_on(mfls,ls1.values(0),mft)
            T_init[ls1_t > 0] = ins.T_atm
        if ins.topography:
            ls3_t = compute_interpolate_on(mfls, ls3.values(0), mft)
            T_init[ls3_t < 0] = ins.T_atm

        md.add_initialized_fem_data('Previous_t', mft, T_init)
        md.add_initialized_fem_data('Previous2_t', mft, T_init)

        T_mat = compute_interpolate_on(mft, T_init, mfmat)
        eta_exp = ins.eta_exp.replace('vfta', str(ins.vfta)).replace('vftb', str(ins.vftb)).replace(
            'vftc', str(ins.vftc))
        eta = eval(eta_exp.replace('exp', 'np.exp').replace('T', 'T_mat'))
        eta[T_mat <= (ins.vftc - 273)] = ins.max_eta

    if ins.solidification:
        T_ls = compute_interpolate_on(mft, T_init, mfls)
        ls2.set_values((T_ls - Tg) / Tg)
        mls.adapt()

    if ins.free_surface:
        eta[ls1_mat] = ins.eta2
    if ins.topography:
        eta[ls3_mat] = ins.eta3
else:
    eta = eval(ins.eta_exp.replace('exp', 'np.exp').replace('T', str(ins.T0)).replace('vft','ins.vft')) * ones_mat
    if ins.free_surface:
        eta[ls1_mat > 0] = ins.eta2
    if ins.topography:
        eta[ls3_mat < 0] = ins.eta3

# time dependence
md.add_initialized_data('dt', [ins.dt])
BDF0 = 1
BDF1 = -1
BDF2 = 0
md.add_initialized_data('BDF0', [1])
md.add_initialized_data('BDF1', [-1])
md.add_initialized_data('BDF2', [0])

md_init.add_initialized_data('dt', [ins.dt])

# Elasticity
lam = -2/3*eta
mu = eta
solid = 0 * ones_mat

if ins.temp & ins.solidification:
    lam_solid = ins.E * ins.nu / ((1 + ins.nu) * (1 - 2 * ins.nu))
    mu_solid = ins.E / (2 * (1 + ins.nu))
    ls2_mat = compute_interpolate_on(mfls,ls2.values(0),mfmat)
    if ins.free_surface:
        lam[(ls1_mat <= 0) & (ls2_mat < 0)] = lam_solid
        mu[(ls1_mat <= 0) & (ls2_mat < 0)] = mu_solid
        solid[(ls1_mat <= 0) & (ls2_mat < 0)] = 1
    else:
        lam[(ls2_mat < 0)] = lam_solid
        mu[(ls2_mat < 0)] = mu_solid
        solid[(ls2_mat < 0)] = 1

md.add_initialized_fem_data('lambda', mfmat, lam)
md.add_initialized_fem_data('mu', mfmat, mu)
md.add_initialized_fem_data('solid', mfmat, solid)

md_init.add_initialized_fem_data('lambda', mfmat, lam)
md_init.add_initialized_fem_data('mu', mfmat, mu)
md_init.add_initialized_fem_data('solid', mfmat, solid)

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
    md.add_fem_variable('d',mfu_cut)
    md.add_fem_variable('p', mfp_cut)
    if ins.temp:
        md.add_fem_variable('t', mft_cut)
else:
    md.add_fem_variable('u', mfu)
    md.add_fem_variable('d', mfu)
    md.add_fem_variable('p', mfp)
    if ins.temp:
        md.add_fem_variable('t', mft)

md.add_fem_variable('psi',mfls)
md.add_fem_variable('psis',mfls)
md.add_initialized_data('psie',ins.epsilon_psi)

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
linear_elastic_adv = "(lambda*(Trace(Grad(u.Grad_Previous_d))*Div_Test_u) + mu*((Grad(u.Grad_Previous_d) + Grad(u.Grad_Previous_d)'):Grad_Test_u))*(dt/BDF0*solid)"
residual_stress1 = "-(lambda*(Div_Previous_d*Div_Test_u) + mu*((Grad_Previous_d + Grad_Previous_d'):Grad_Test_u))*(BDF1/BDF0)*solid"
residual_stress2 = "-(lambda*(Div_Previous2_d*Div_Test_u) + mu*((Grad_Previous2_d + Grad_Previous2_d'):Grad_Test_u))*(BDF2/BDF0)*solid"

if ins.steady:
    md.add_nonlinear_term(mim, 'solid*' + time_int_u)
else:
    md.add_nonlinear_term(mim, time_int_u)
md.add_nonlinear_term(mim, linear_elastic)
#md.add_nonlinear_term(mim, linear_elastic_adv)

if ins.free_surface:

    md.add_nonlinear_term(mim,"psis*Test_psis + psie*Grad_psis.Grad_Test_psis")
    md.add_source_term_brick(mim,'psis','Previous_psi')

    time_int_psi = "((BDF0*psi+BDF1*Previous_psi+BDF2*Previous2_psi)/dt)"
    advection_psi = "(fext:Grad_psi)*Test_psi"
    md.add_nonlinear_term(mim, time_int_psi + '*Test_psi')
    md.add_nonlinear_term(mim, advection_psi)
    #tau_SUPG_psi = "1/(2/dt + 2*Norm((fext*Grad_psi))/h)"
    #S_SUPG_psi = "((fext*Grad_psi).Grad_psi)*((fext*Grad_psi).Grad_Test_psi)"

    #md.add_nonlinear_term(mim, tau_SUPG_psi + '*((fext*Grad_psi).Grad_Test_psi)*' + time_int_psi)
    #md.add_nonlinear_term(mim, tau_SUPG_psi + '*' + S_SUPG_psi)
md.disable_variable('psis')
md.disable_variable('psi')


if ins.solidification:
    md.add_nonlinear_term(mim, residual_stress1 + residual_stress2)
    time_int_d = "((BDF0*d+BDF1*Previous_d+BDF2*Previous2_d)/dt)"
    advection_d = "(u.Grad_d)*Test_d"
    md.add_nonlinear_term(mim, time_int_d + '*Test_d')
    md.add_nonlinear_term(mim, advection_d)
    md.add_nonlinear_term(mim, "-u.Test_d")
    tau_SUPG_d = "1/(2/dt + 2*Norm(u)/h)"
    S_SUPG_d = "(u.Grad_d)*(u.Grad_Test_d)"

    md.add_nonlinear_term(mim, tau_SUPG_d + '*(u.Grad_Test_d)*' + time_int_d)
    md.add_nonlinear_term(mim, tau_SUPG_d + '*' + S_SUPG_d)
md.disable_variable('d')

if ins.temp:
    time_int_t = "(BDF0*t+BDF1*Previous_t+BDF2*Previous2_t)/dt"
    advection_t = "(u.Grad_t)*Test_t"
    diffusion = "kappa*(Grad_t.Grad_Test_t)"
    S_SUPG_t = "(u.Grad_t)*(u.Grad_Test_t) + kappa*(Grad_t).Grad(u.Grad_Test_t)"
    S_GLS_t = "(kappa*(Grad(u.Grad_t).Grad_Test_t))"
    md.add_macro('Pe', "h*Norm(u)/(2*kappa)")
    md.add_macro('xi', "min(Pe/3,1)")

    md.add_nonlinear_term(mim, time_int_t + '*Test_t')
    md.add_nonlinear_term(mim, advection_t)
    md.add_nonlinear_term(mim, diffusion)

    tau_SUPG_t = '1/(2/dt + 4*kappa/(h*h) + 2*Norm(u)/h)*xi'
    tau_GLS_t = '1/(2/dt + 4*kappa/(h*h) + 2*Norm(u)/h)*xi'
    if 'SUPG' in ins.stab_t:
        # tau_supg = 'h/(2*Norm(u))*(cosh(Norm(u)*h/(2*kappa))/sinh(Norm(u)*h/(2*kappa)) - 2*kappa/(h*Norm(u)))'
        # tau_SUPG_t = 'h*h/(4*kappa)*min(1/3,1/Pe)'
        md.add_nonlinear_term(mim, tau_SUPG_t + '*(u.Grad_Test_t)*' + time_int_t)
        md.add_nonlinear_term(mim, tau_SUPG_t + '*' + S_SUPG_t)

    elif 'GLS' in ins.stab_t:
        md.add_nonlinear_term(mim, tau_GLS_t + '*(u.Grad_Test_t)*' + time_int_t)
        md.add_nonlinear_term(mim, tau_GLS_t + '*' + S_SUPG_t)
        md.add_nonlinear_term(mim, tau_GLS_t + '*' + S_GLS_t)

# add body force
if (type(ins.f_x) is type(None)) and (type(ins.f_y) is type(None)):
    f_xi = None
    f_yi = None
else:
    if (type(ins.f_x) is float) or (type(ins.f_x) is int):
        f_xi = ones_mat * ins.f_x
    elif type(ins.f_x) is str:
        f_xi = eval(ins.f_x.replace('y', 'y_ls').replace('x', 'x_ls'))
    elif type(ins.f_x) is type(None):
        f_xi = ones_mat * 0

    if (type(ins.f_y) is float) or (type(ins.f_y) is int):
        f_yi = ones_mat * ins.f_y
    elif type(ins.f_y) is str:
        f_yi = eval(ins.f_y.replace('y', 'y_ls').replace('x', 'x_ls'))
    elif type(ins.f_y) is type(None):
        f_yi = ones_mat * 0

    md.add_initialized_fem_data('body', mfmat, [f_xi, f_yi])
    md.add_source_term_brick(mim, 'u', 'body')

    md_init.add_initialized_fem_data('body', mfmat, [f_xi, f_yi])
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
            data_udx = eval('ins.' + bound + '_udx') * ones_u
        if type(eval('ins.' + bound + '_udx')) is str:
            data_udx = eval(eval('ins.' + bound + '_udx'))
        if type(eval('ins.' + bound + '_udy')) is type(None):
            data_udy = 0 * ones_u

        if (type(eval('ins.' + bound + '_udy')) is float) or (type(eval('ins.' + bound + '_udy')) is int):
            data_udy = eval('ins.' + bound + '_udy') * ones_u
        if type(eval('ins.' + bound + '_udy')) is str:
            data_udy = eval(eval('ins.' + bound + '_udy'))
        if type(eval('ins.' + bound + '_udx')) is type(None):
            data_udx = 0 * ones_u

        md.add_initialized_fem_data(bound + 'data', mfu, [data_udx, data_udy])
        md_init.add_initialized_fem_data(bound + 'data', mfu, [data_udx, data_udy])

        md.add_normal_source_term_brick(mim, 'u', bound + 'data', i + 1)
        md_init.add_normal_source_term_brick(mim_all, 'u', bound + 'data', i + 1)

# add pressure on boundary or free surface
md.add_initialized_fem_data('patm', mfp, [ins.p_atm * ones_p])
md_init.add_initialized_fem_data('patm', mfp, [ins.p_atm * ones_p])
if ins.solve_air | (not ins.free_surface):
    md.add_Dirichlet_condition_with_multipliers(mim, 'p', 1, 5, dataname='patm')
    md_init.add_Dirichlet_condition_with_multipliers(mim_all, 'p', 1, 5, dataname='patm')

else:
    md.add_Dirichlet_condition_with_multipliers(mim_surf, 'p', 1, -1, dataname='patm')
    md_init.add_Dirichlet_condition_with_multipliers(mim_surf, 'p', 1, -1, dataname='patm')

#if ins.free_surface:
#    md.add_initialized_fem_data('psis_bound',mfls,[Ls1])
#    for i, bound in enumerate(bounds):
#        md.add_Dirichlet_condition_with_multipliers(mim,'psis',1,i+1,dataname='psis_bound')

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
    md.add_Dirichlet_condition_with_multipliers(mim, 'u', 2, 6,
                                                dataname='influxdata')
    if ins.temp:
        md.add_initialized_fem_data('influxdata_t', mft, [eval(ins.influx_T) * ones_t])
        md.add_Dirichlet_condition_with_multipliers(mim, 't', 2, 6,
                                                    dataname='influxdata_t')

#### Solve ####
print('Starting solve')
if not ins.restart:
    # initialize pressure and velocity with incompressible, steady stokes
    md_init.solve('max_res', 1E-8, 'max_iter', 100, 'noisy')
    p_init = md_init.variable('p')
    u_init = md_init.variable('u')

    md.set_variable('Previous_p', p_init)
    md.set_variable('Previous2_p', p_init)
    md.set_variable('Previous_u', u_init)
    md.set_variable('Previous2_u', u_init)

    mfu.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_uinit_' + '0' * ndigits + '.vtk', u_init)
    mfp.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_Pinit_' + '0' * ndigits + '.vtk', p_init)

    print('time = 0')
    # bootstrap
    md.solve('max_res', 1E-10, 'max_iter', 100, 'noisy')

    md.disable_variable('u')
    md.disable_variable('p')
    if ins.temp:
        md.disable_variable('t')

    if ins.free_surface:
        md.enable_variable('psis')
        md.solve('max_res', 1E-10, 'max_iter', 10, 'noisy')
        md.disable_variable('psis')
        #Previous_ls = md.variable('Previous_ls')

        # construct extension velocities
        # speed on the interface is U.Grad(Psi)
        #Grad_psi = compute_gradient(mfls, md.variable('Previous_psi'), mflsgrad)
        #Grad_psis = compute_gradient(mflss, md.variable('psis'), mflssgrad)
        #Grad2_psis = compute_gradient(mflssgrad, Grad_psis, mflssgrad2)

        #Grad_psis_lsgrad = compute_interpolate_on(mflssgrad, Grad_psis, mflsgrad)
        #Grad_mag = np.sqrt(Grad_psis_lsgrad[0, :] ** 2 + Grad_psis_lsgrad[1, :] ** 2)
        #Grad_mag[Grad_mag<1e-14] = 1e-14

        #Grad2_psis_lsgrad = compute_interpolate_on(mflssgrad2,Grad2_psis,mflsgrad)
        #curvature = (Grad2_psis_lsgrad[0,0,:] * Grad_psis_lsgrad[1,:]**2 - (Grad2_psis_lsgrad[1,0,:] + Grad2_psis_lsgrad[0,1,:]) * Grad_psis_lsgrad[0,:] * Grad_psis_lsgrad[1,:] + Grad2_psis_lsgrad[1,1,:] * Grad_psis_lsgrad[0,:]**2)/Grad_mag**3
        #curvature[edges_lsgrad] = 0

        Psi_grid = sciinterp.griddata(D_ls.transpose(), md.variable('psis'),
                                   np.array([x_grid.flatten(), y_grid.flatten()]).transpose(), method='linear').reshape(x_grid.shape)

        dx_Psi_grid = np.zeros_like(Psi_grid)
        dx_Psi_grid[:, 1:-1] = (Psi_grid[:, 2:] - Psi_grid[:, :-2]) / (2 * dx)
        dx_Psi_grid[:, 0] = (Psi_grid[:, 1] - Psi_grid[:, 0]) / dx
        dx_Psi_grid[:, -1] = (Psi_grid[:, -1] - Psi_grid[:, -2]) / dx
        
        dy_Psi_grid = np.zeros_like(Psi_grid)
        dy_Psi_grid[1:-1,:] = (Psi_grid[2:,:]  - Psi_grid[:-2,:]) / (2*dy)
        dy_Psi_grid[0, :]   = (Psi_grid[1, :]  - Psi_grid[0, :])  / dy
        dy_Psi_grid[-1, :]  = (Psi_grid[-1, :] - Psi_grid[-2, :]) / dy

        mag_grad_Psi_grid = np.sqrt(dx_Psi_grid**2 + dy_Psi_grid**2)
        mag_grad_Psi_grid[mag_grad_Psi_grid < 1e-14] = 1e-14

        dxx_Psi_grid = np.zeros_like(Psi_grid)
        dxx_Psi_grid[:, 1:-1] = (Psi_grid[:, 2:] - 2 * Psi_grid[:, 1:-1] + Psi_grid[:, :-2]) / (dx ** 2)
        dxx_Psi_grid[:, 0] = dxx_Psi_grid[:, 1]
        dxx_Psi_grid[:, -1] = dxx_Psi_grid[:, -2]

        dyy_Psi_grid = np.zeros_like(Psi_grid)
        dyy_Psi_grid[1:-1, :] = (Psi_grid[2:, :] -2*Psi_grid[1:-1, :] + Psi_grid[:-2, :]) / (dy ** 2)
        dyy_Psi_grid[0, :]    = dyy_Psi_grid[1, :]
        dyy_Psi_grid[-1, :]   = dyy_Psi_grid[-2, :]

        dxy_Psi_grid = np.zeros_like(Psi_grid)
        dxy_Psi_grid[1:-1,1:-1] = (Psi_grid[2:, 2:] - Psi_grid[2:, :-2] - Psi_grid[:-2, 2:] + Psi_grid[:-2,:-2]) / (4*dx*dy)
        dxy_Psi_grid[0, 1:-1] = (Psi_grid[1, 2:] - Psi_grid[1, :-2] - Psi_grid[0, 2:] + Psi_grid[0, :-2]) / (2*dx*dy)
        dxy_Psi_grid[-1, 1:-1] = (Psi_grid[-1, 2:] - Psi_grid[-1, :-2] - Psi_grid[-2, 2:] + Psi_grid[-2, :-2]) / (2*dx*dy)
        dxy_Psi_grid[1:-1, 0] = (Psi_grid[2:, 1] - Psi_grid[2:, 1] - Psi_grid[:-2, 0] + Psi_grid[:-2, 0]) / (2*dx*dy)
        dxy_Psi_grid[1:-1, -1] = (Psi_grid[2:, -1] - Psi_grid[2:, -1] - Psi_grid[:-2, -2] + Psi_grid[:-2, -2]) / (2*dx*dy)
        dxy_Psi_grid[0, 0] = (Psi_grid[1, 1] - Psi_grid[1, 0] - Psi_grid[0, 1] + Psi_grid[0, 0]) / (dx*dy)
        dxy_Psi_grid[0, -1] = (Psi_grid[1, -1] - Psi_grid[1, -2] - Psi_grid[0, -1] + Psi_grid[0, -2]) / (dx * dy)
        dxy_Psi_grid[-1, 0] = (Psi_grid[-1, 1] - Psi_grid[-1, 0] - Psi_grid[-2, 1] + Psi_grid[-2, 0]) / (dx * dy)
        dxy_Psi_grid[-1, -1] = (Psi_grid[-1, -1] - Psi_grid[-1, -2] - Psi_grid[-2, -1] + Psi_grid[-2, -2]) / (dx * dy)

        curvature = (dxx_Psi_grid * (dy_Psi_grid ** 2) - 2*(dx_Psi_grid * dy_Psi_grid * dxy_Psi_grid) + dyy_Psi_grid * (dx_Psi_grid)**2)/(mag_grad_Psi_grid**3)

        if ins.free_surface | ins.topography:
            u_ls = compute_interpolate_on(mfu_cut,md.variable('u'),mfls)

        else:
            u_ls = compute_interpolate_on(mfu,md.variable('u'),mfls)

        ux_grid = sciinterp.griddata(D_ls.transpose(), u_ls[0, :],
                                     np.array([x_grid.flatten(), y_grid.flatten()]).transpose(),
                                     method='linear')
        uy_grid = sciinterp.griddata(D_ls.transpose(), u_ls[1, :],
                                     np.array([x_grid.flatten(), y_grid.flatten()]).transpose(),
                                     method='linear')

        F_grid = (ux_grid * dx_Psi_grid.flatten() + uy_grid * dy_Psi_grid.flatten())

        #eta_ls = compute_interpolate_on(mfmat,eta,mflsgrad)
        #relax_max = 0.1 * (np.max(np.sqrt(f_xi ** 2 + f_yi ** 2)) - np.min(np.sqrt(f_xi ** 2 + f_yi ** 2)))/eta_ls * np.sqrt(dx ** 2 + dy ** 2)
        relax_max = np.sqrt(dx**2 + dy**2)/(2*ins.dt)
        #Fmax = np.min([0.5*np.abs(F), relax_max * ones_lsgrad], axis=0)
        #Fmin = np.max([-0.5*np.abs(F), -relax_max * ones_lsgrad], axis=0)

        relax_speed = ins.visc_coeff * curvature

        relax_speed[relax_speed > relax_max] = relax_max
        relax_speed[relax_speed < -relax_max] = -relax_max

        if ins.topography:
            ls3_grid = sciinterp.griddata(D_ls.transpose(), ls3.values(0),
                                   np.array([x_grid.flatten(), y_grid.flatten()]).transpose(), method='linear').reshape(x_grid.shape)
            relax_speed[ls3_grid<0] = 0

        mask_grid = (Psi_grid>0)
        d_ls1, F_ext_grid = skfmm.extension_velocities(sciinterp.griddata(D_ls.transpose(), md.variable('psis'),
                                   np.array([x_grid.flatten(), y_grid.flatten()]).transpose(), method='linear').reshape(x_grid.shape),
                                              F_grid.reshape(x_grid.shape) - relax_speed,
                                              dx=[dy, dx],ext_mask=mask_grid)
        F_ext = sciinterp.griddata(np.array([x_grid.flatten(), y_grid.flatten()]).transpose(), F_ext_grid.flatten(),
                                   D_ls.transpose(),
                                   method='nearest').flatten()
        dx_Psi = sciinterp.griddata(np.array([x_grid.flatten(), y_grid.flatten()]).transpose(), dx_Psi_grid.flatten(),
                                   D_ls.transpose(),
                                   method='nearest').flatten()
        dy_Psi = sciinterp.griddata(np.array([x_grid.flatten(), y_grid.flatten()]).transpose(), dy_Psi_grid.flatten(),
                                    D_ls.transpose(),
                                    method='nearest').flatten()
        mag_grad_Psi = sciinterp.griddata(np.array([x_grid.flatten(), y_grid.flatten()]).transpose(), mag_grad_Psi_grid.flatten(),
                                    D_ls.transpose(),
                                    method='nearest').flatten()

        fext = 0*ones_f
        fext[::2] = F_ext*dx_Psi/mag_grad_Psi
        fext[1::2] = F_ext*dy_Psi/mag_grad_Psi
        #d = sciinterp.griddata(np.array([x_grid.flatten(), y_grid.flatten()]).transpose(), d.flatten(), D_ls.transpose(),
        #                       method='nearest').flatten()

        md.set_variable('fext',fext)
        #md.set_variable('psis_bound', md.variable('Previous_psi') + compute_interpolate_on(mflsgrad,
        #                                    (F_ext*Grad_psi)[0,:]*Grad_psi[0,:] + (F_ext*Grad_psi)[1,:]*Grad_psi[1,:],
        #                                                                                   mfls)*ins.dt)

        md.enable_variable('psi')
        md.solve('max_res', 1E-10, 'max_iter', 10, 'noisy')
        Previous_Ls = Ls1
        Ls1 = md.variable('psi')

    Previous_u = u_init
    U = ones_u * 0
    U[eval(ind_u)] = md.variable('u')

    Previous_p = p_init
    P = ones_p * 0
    P[eval(ind_p)] = md.variable('p')

    if ins.temp:
        Previous_T = T_init
        T = ones_t * ins.T_atm
        T[eval(ind_t)] = md.variable('t')

        T_ls = compute_interpolate_on(mft,T,mfls)
        if ins.solidification:
            ls2.set_values((T_ls - Tg) / Tg)

            Previous_d = d_init
            md.enable_variable('d')

            md.solve('max_res', 1E-10, 'max_iter', 10, 'noisy')

            D = ones_d * 0
            D[eval(ind_u)] = md.variable('d')

            if np.max(T) <= Tg:
                d_ls2 = np.ones(x_grid.shape) * (-2 * np.max([ins.L_x, ins.L_y]))
            else:
                Ls2_ext = sciinterp.griddata(D_ls.transpose(), ls2.values(0),
                                             np.array([x_grid.flatten(), y_grid.flatten()]).transpose(),
                                             method='nearest')
                d_ls2 = skfmm.distance(Ls2_ext.reshape(x_grid.shape), dx=[dy, dx])

            if ins.free_surface:
                d = np.maximum(np.array(d_ls1), np.array(d_ls2))
            else:
                d = d_ls2

            d, D_ext_x = skfmm.extension_velocities(d, compute_interpolate_on(mfu, D, mfls)[0,:].reshape(x_grid.shape),
                                                      dx=[dy, dx])
            d, D_ext_y = skfmm.extension_velocities(d, compute_interpolate_on(mfu, D, mfls)[1, :].reshape(x_grid.shape),
                                                    dx=[dy, dx])
            D_ext_x = sciinterp.griddata(np.array([x_grid.flatten(), y_grid.flatten()]).transpose(), D_ext_x.flatten(),
                                       D_u.transpose(),
                                       method='nearest').flatten()
            D_ext_y = sciinterp.griddata(np.array([x_grid.flatten(), y_grid.flatten()]).transpose(), D_ext_y.flatten(),
                                         D_u.transpose(),
                                         method='nearest').flatten()
            D_ext = D_ext_x
            D_ext[1::2] = D_ext_y[1::2]
            Previous_Ls_u = compute_interpolate_on(mfls,Previous_Ls,mfu)
            Ls1_u = compute_interpolate_on(mfls, Ls1, mfu)
            D[(Previous_Ls_u>=0)&(Ls1_u<=0)] = D_ext[(Previous_Ls_u>=0)&(Ls1_u<=0)]

            md.disable_variable('d')
        else:
            Previous_d = d_init
            D = d_init
    else:
        Previous_d = d_init
        D = d_init

    md.enable_variable('u')
    md.enable_variable('p')
    if ins.temp:
        md.enable_variable('t')

    if ins.vtk:
        u = md.variable('u')
        P = md.variable('p')

        ti = 0;

        mfu.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_u_' + '0'*ndigits + '.vtk', u_init)
        if ins.true_u:
            mfu.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_utrue_' + '0' * ndigits + '.vtk', eval(ins.true_u))
        mfp.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_P_' + '0'*ndigits + '.vtk', p_init)
        if ins.true_p:
            mfp.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_Ptrue_' + '0' * ndigits + '.vtk', eval(ins.true_p))
        if ins.free_surface:
            mfls.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_Ls1_' + '0'*ndigits + '.vtk', Ls1)
            if ins.true_ls1:
                mfls.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_Ls1true_' + '0' * ndigits + '.vtk',
                                  eval(ins.true_ls1))
            mff.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_Fext_' + '0'*ndigits + '.vtk', fext)
            mfls.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_curvature_' + '0'*ndigits + '.vtk',
                               sciinterp.griddata(np.array([x_grid.flatten(), y_grid.flatten()]).transpose(), curvature.flatten(),
                                         D_ls.transpose(),
                                         method='nearest'))
            mfmat.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_rho_' + '0' * ndigits + '.vtk', rho)
        if ins.temp:
            T = md.variable('t')
            mft.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_T_' + '0'*ndigits + '.vtk', T_init)
            if ins.true_t:
                mft.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_Ttrue_' + '0' * ndigits + '.vtk',
                                  eval(ins.true_t))
            mfmat.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_mu_' + '0'*ndigits + '.vtk', mu)
            if ins.solidification:
                mfu.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_d_' + '0'*ndigits + '.vtk', d_init)
                if ins.true_d:
                    mfu.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_dtrue_' + '0' * ndigits + '.vtk',
                                      eval(ins.true_d))
                mft.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_Ls2_' + '0' * ndigits + '.vtk', (T_init-Tg)/Tg)
        if (ins.noutput==1):
            numstr = str(ins.dt * 10 ** ndecimal).split('.')[0].zfill(ndigits)

            mfu.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_u_' + numstr + '.vtk', U)
            if ins.true_u:
                mfu.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_utrue_' + numstr + '.vtk',
                                  eval(ins.true_u))
            mfp.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_P_' + numstr + '.vtk', P)
            if ins.true_p:
                mfp.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_Ptrue_' + numstr + '.vtk',
                                  eval(ins.true_p))
            if ins.free_surface:
                mfls.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_Ls1_' + numstr + '.vtk', md.variable('Previous_psi'))
                if ins.true_ls1:
                    mfls.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_Ls1true_' + numstr + '.vtk',
                                      eval(ins.true_ls1))
                mff.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_Fext_' + numstr + '.vtk', fext)
                mfls.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_curvature_' + numstr + '.vtk',
                                   sciinterp.griddata(np.array([x_grid.flatten(), y_grid.flatten()]).transpose(),
                                                      curvature.flatten(),
                                                      D_ls.transpose(),
                                                      method='nearest'))
                mfmat.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_rho_' + numstr + '.vtk', rho)
            if ins.temp:
                T = md.variable('t')
                mft.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_T_' + numstr + '.vtk', T)
                if ins.true_u:
                    mft.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_Ttrue_' + numstr + '.vtk',
                                      eval(ins.true_t))
                mfmat.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_mu_' + numstr + '.vtk', mu)
                if ins.solidification:
                    mfu.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_d_' + numstr + '.vtk', D)
                    if ins.true_d:
                        mfu.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_dtrue_' + numstr + '.vtk',
                                          eval(ins.true_d))
                    mft.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_Ls2_' + numstr + '.vtk',
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
            #radii_t = compute_interpolate_on(mls_cut,radii,mft)
            # md.set_variable('h',radii_t)
            # ls1_t = compute_interpolate_on(mfls,ls1.values(0),mft)

    if ins.free_surface | ins.topography:
        mfp_cut.set_partial(eval(ind_p))
        mfu_cut.set_partial(eval(ind_u))
        if ins.temp:
            mft_cut.set_partial(eval(ind_t))

    # density
    if ins.free_surface:
        ls1_mat = compute_interpolate_on(mfls,ls1.values(0),mfmat)
        rho[(ls1_mat <= 0)] = ins.rho1
        rho[(ls1_mat > 0)] = ins.rho2
    if ins.topography:
        ls3_mat = compute_interpolate_on(mfls, ls3.values(0), mfmat)
        rho[ls3_mat < 0] = ins.rho3
    md.set_variable('rho', rho)

    if ins.free_surface:
        beta[(ls1_mat <= 0)] = ins.beta1
        beta[(ls1_mat > 0)] = ins.beta2
    if ins.topography:
        beta[ls3_mat < 0] = ins.beta3
    md.set_variable('beta', beta)

    if ins.temp:
        # thermal diffusivity
        if ins.free_surface:
            kappa[(ls1_mat <= 0)] = ins.kappa1
            kappa[(ls1_mat > 0)] = ins.kappa2
        if ins.topography:
            kappa[ls3_mat < 0] = ins.kappa3
        md.set_variable('kappa', kappa)

        # shift_variables_for_time_integration
        md.set_variable('Previous2_t', Previous_T)
        md.set_variable('Previous_t', T)

        # update viscosity field
        T_mat = compute_interpolate_on(mft,T,mfmat)
        ls2_mat = (T_mat - Tg) / Tg
        eta = eval(ins.eta_exp.replace('exp', 'np.exp').replace('T', 'T_mat').replace('vft','ins.vft'))
        eta[eta > ins.max_eta] = ins.max_eta
        eta[T_ls <= (ins.vftc - 273)] = ins.max_eta
        if ins.temp & ins.solidification:
            T_ls = compute_interpolate_on(mft,T,mfls)
            ls2.set_values((T_ls - Tg) / Tg)

    else:
        eta = eval(ins.eta_exp.replace('exp', 'np.exp').replace('T', str(ins.T0)).replace('vft','ins.vft')) * ones_mat

    if ins.free_surface:
        eta[(ls1_mat > 0)] = ins.eta2
    if ins.topography:
        eta[ls3_mat < 0] = ins.eta3
    mu = eta

    lam = -2/3*eta
    solid = 0 * ones_mat
    if ins.temp & ins.solidification:
        # elasticity
        if ins.free_surface:
            lam[(ls1_mat <= 0) & (ls2_mat <= 0)] = lam_solid
            mu[(ls1_mat <= 0) & (ls2_mat <= 0)] = mu_solid
            solid[(ls1_mat <= 0) & (ls2_mat <= 0)] = 1
        else:
            lam[ls2_mat <= 0] = lam_solid
            mu[ls2_mat <= 0] = mu_solid
            solid[ls2_mat <= 0] = 1

        md.set_variable('solid', solid)

    md.set_variable('lambda', lam)
    md.set_variable('mu', mu)

    # update body force
    if (type(ins.f_x) is type(None)) and (type(ins.f_y) is type(None)):
        f_xi = None
        f_yi = None
    else:
        if (type(ins.f_x) is float) or (type(ins.f_x) is int):
            f_xi = ones_mat * ins.f_x
        elif type(ins.f_x) is str:
            f_xi = eval(ins.f_x.replace('y', 'y_ls').replace('x', 'x_ls'))
        elif type(ins.f_x) is type(None):
            f_xi = ones_mat * 0

        if (type(ins.f_y) is float) or (type(ins.f_y) is int):
            f_yi = ones_mat * ins.f_y
        elif type(ins.f_y) is str:
            f_yi = eval(ins.f_y.replace('y', 'y_ls').replace('x', 'x_ls'))
        elif type(ins.f_y) is type(None):
            f_yi = ones_mat * 0

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

    md.disable_variable('u')
    md.disable_variable('p')
    if ins.temp:
        md.disable_variable('t')

    if ins.free_surface:
        md.enable_variable('psis')
        md.solve('max_res', 1E-10, 'max_iter', 10, 'noisy')
        md.disable_variable('psis')

        # construct extension velocities
        # speed on the interface is U.Grad(Psi)
        md.set_variable('Previous2_psi',Previous_Ls)
        md.set_variable('Previous_psi', Ls1)
        #Grad_psi = compute_gradient(mfls, md.variable('psi'), mflsgrad)
        #Grad_psis = compute_gradient(mflss, md.variable('psis'), mflssgrad)
        #Grad2_psis = compute_gradient(mflssgrad, Grad_psis, mflssgrad2)

        #Grad_psis_lsgrad = compute_interpolate_on(mflssgrad, Grad_psis, mflsgrad)
        #Grad_mag = np.sqrt(Grad_psis_lsgrad[0, :] ** 2 + Grad_psis_lsgrad[1, :] ** 2)
        #Grad_mag[Grad_mag<1e-14] = 1e-14

        #Grad2_psis_lsgrad = compute_interpolate_on(mflssgrad2,Grad2_psis,mflsgrad)
        #curvature = (Grad2_psis_lsgrad[0, 0, :] * Grad_psis_lsgrad[1, :] ** 2 - (Grad2_psis_lsgrad[1, 0, :] + Grad2_psis_lsgrad[0, 1, :]) * Grad_psis_lsgrad[0, :] * Grad_psis_lsgrad[1, :] + Grad2_psis_lsgrad[1,1,:] * Grad_psis_lsgrad[0,:] ** 2) / Grad_mag ** 3
        #curvature[edges_lsgrad] = 0

        Psi_grid = sciinterp.griddata(D_ls.transpose(), md.variable('psis'),
                                      np.array([x_grid.flatten(), y_grid.flatten()]).transpose(),
                                      method='linear').reshape(x_grid.shape)

        dx_Psi_grid = np.zeros_like(Psi_grid)
        dx_Psi_grid[:, 1:-1] = (Psi_grid[:, 2:] - Psi_grid[:, :-2]) / (2 * dx)
        dx_Psi_grid[:, 0] = (Psi_grid[:, 1] - Psi_grid[:, 0]) / dx
        dx_Psi_grid[:, -1] = (Psi_grid[:, -1] - Psi_grid[:, -2]) / dx
        #dx_Psi_grid[:, 2:-2] = (-Psi_grid[:, 4:] + 8*Psi_grid[:, 3:-1] - 8*Psi_grid[:, 1:-3] + Psi_grid[:, :-4]) / (12 * dx)

        dy_Psi_grid = np.zeros_like(Psi_grid)
        dy_Psi_grid[1:-1, :] = (Psi_grid[2:, :] - Psi_grid[:-2, :]) / (2 * dy)
        dy_Psi_grid[0, :] = (Psi_grid[1, :] - Psi_grid[0, :]) / dy
        dy_Psi_grid[-1, :] = (Psi_grid[-1, :] - Psi_grid[-2, :]) / dy
        #dy_Psi_grid[2:-2, :] = (-Psi_grid[4:, :] + 8*Psi_grid[3:-1, :] - 8*Psi_grid[1:-3, :] + Psi_grid[:-4, :]) / (12 * dy)

        mag_grad_Psi_grid = np.sqrt(dx_Psi_grid ** 2 + dy_Psi_grid ** 2)
        mag_grad_Psi_grid[mag_grad_Psi_grid < 1e-14] = 1e-14

        dxx_Psi_grid = np.zeros_like(Psi_grid)
        dxx_Psi_grid[:, 1:-1] = (Psi_grid[:, 2:] - 2 * Psi_grid[:, 1:-1] + Psi_grid[:, :-2]) / (dx ** 2)
        dxx_Psi_grid[:, 0] = dxx_Psi_grid[:, 1]
        dxx_Psi_grid[:, -1] = dxx_Psi_grid[:, -2]
        #dxx_Psi_grid[:, 2:-2] = (-Psi_grid[:, 4:] + 16*Psi_grid[:, 3:-1] - 30*Psi_grid[:, 2:-2] + 16*Psi_grid[:, 1:-3] - Psi_grid[:, :-4]) / (12*dx ** 2)

        dyy_Psi_grid = np.zeros_like(Psi_grid)
        dyy_Psi_grid[1:-1, :] = (Psi_grid[2:, :] - 2 * Psi_grid[1:-1, :] + Psi_grid[:-2, :]) / (dy ** 2)
        dyy_Psi_grid[0, :] = dyy_Psi_grid[1, :]
        dyy_Psi_grid[-1, :] = dyy_Psi_grid[-2, :]
        #dyy_Psi_grid[2:-2, :] = (-Psi_grid[4:, :] + 16*Psi_grid[3:-1, :] - 30 * Psi_grid[2:-2, :] + 16*Psi_grid[1:-3, :] - Psi_grid[:-4, :]) / (12*dy ** 2)

        dxy_Psi_grid = np.zeros_like(Psi_grid)
        dxy_Psi_grid[1:-1, 1:-1] = (Psi_grid[2:, 2:] - Psi_grid[2:, :-2] - Psi_grid[:-2, 2:] + Psi_grid[:-2, :-2]) / (
                    4 * dx * dy)
        dxy_Psi_grid[0, 1:-1] = (Psi_grid[1, 2:] - Psi_grid[1, :-2] - Psi_grid[0, 2:] + Psi_grid[0, :-2]) / (
                    2 * dx * dy)
        dxy_Psi_grid[-1, 1:-1] = (Psi_grid[-1, 2:] - Psi_grid[-1, :-2] - Psi_grid[-2, 2:] + Psi_grid[-2, :-2]) / (
                    2 * dx * dy)
        dxy_Psi_grid[1:-1, 0] = (Psi_grid[2:, 1] - Psi_grid[2:, 1] - Psi_grid[:-2, 0] + Psi_grid[:-2, 0]) / (
                    2 * dx * dy)
        dxy_Psi_grid[1:-1, -1] = (Psi_grid[2:, -1] - Psi_grid[2:, -1] - Psi_grid[:-2, -2] + Psi_grid[:-2, -2]) / (
                    2 * dx * dy)
        dxy_Psi_grid[0, 0] = (Psi_grid[1, 1] - Psi_grid[1, 0] - Psi_grid[0, 1] + Psi_grid[0, 0]) / (dx * dy)
        dxy_Psi_grid[0, -1] = (Psi_grid[1, -1] - Psi_grid[1, -2] - Psi_grid[0, -1] + Psi_grid[0, -2]) / (dx * dy)
        dxy_Psi_grid[-1, 0] = (Psi_grid[-1, 1] - Psi_grid[-1, 0] - Psi_grid[-2, 1] + Psi_grid[-2, 0]) / (dx * dy)
        dxy_Psi_grid[-1, -1] = (Psi_grid[-1, -1] - Psi_grid[-1, -2] - Psi_grid[-2, -1] + Psi_grid[-2, -2]) / (dx * dy)

        curvature = (dxx_Psi_grid * (dy_Psi_grid ** 2) - 2 * (
                    dx_Psi_grid * dy_Psi_grid * dxy_Psi_grid) + dyy_Psi_grid * (dx_Psi_grid) ** 2) / (
                                     mag_grad_Psi_grid ** 3)

        if ins.free_surface | ins.topography:
            u_ls = compute_interpolate_on(mfu_cut, md.variable('u'), mfls)
        else:
            u_ls = compute_interpolate_on(mfu, md.variable('u'), mfls)

        ux_grid = sciinterp.griddata(D_ls.transpose(), u_ls[0, :],
                                     np.array([x_grid.flatten(), y_grid.flatten()]).transpose(),
                                     method='linear')
        uy_grid = sciinterp.griddata(D_ls.transpose(), u_ls[1, :],
                                     np.array([x_grid.flatten(), y_grid.flatten()]).transpose(),
                                     method='linear')

        F_grid = (ux_grid * dx_Psi_grid.flatten() + uy_grid * dy_Psi_grid.flatten())

        # eta_ls = compute_interpolate_on(mfmat,eta,mflsgrad)
        # relax_max = 0.1 * (np.max(np.sqrt(f_xi ** 2 + f_yi ** 2)) - np.min(np.sqrt(f_xi ** 2 + f_yi ** 2)))/eta_ls * np.sqrt(dx ** 2 + dy ** 2)
        relax_max = np.sqrt(dx ** 2 + dy ** 2) / (2 * ins.dt)
        # Fmax = np.min([0.5*np.abs(F), relax_max * ones_lsgrad], axis=0)
        # Fmin = np.max([-0.5*np.abs(F), -relax_max * ones_lsgrad], axis=0)

        relax_speed = ins.visc_coeff * curvature

        relax_speed[relax_speed > relax_max] = relax_max
        relax_speed[relax_speed < -relax_max] = -relax_max

        if ins.topography:
            ls3_grid = sciinterp.griddata(D_ls.transpose(), ls3.values(0),
                                   np.array([x_grid.flatten(), y_grid.flatten()]).transpose(), method='linear').reshape(x_grid.shape)
            relax_speed[ls3_grid<0] = 0

        mask_grid = (Psi_grid > 0)
        d_ls1, F_ext_grid = skfmm.extension_velocities(sciinterp.griddata(D_ls.transpose(), md.variable('psis'),
                                                                          np.array([x_grid.flatten(),
                                                                                    y_grid.flatten()]).transpose(),
                                                                          method='linear').reshape(x_grid.shape),
                                                       F_grid.reshape(x_grid.shape) - relax_speed,
                                                       dx=[dy, dx], ext_mask=mask_grid)

        F_ext = sciinterp.griddata(np.array([x_grid.flatten(), y_grid.flatten()]).transpose(), F_ext_grid.flatten(),
                                   D_ls.transpose(),
                                   method='nearest').flatten()
        dx_Psi = sciinterp.griddata(np.array([x_grid.flatten(), y_grid.flatten()]).transpose(), dx_Psi_grid.flatten(),
                                    D_ls.transpose(),
                                    method='nearest').flatten()
        dy_Psi = sciinterp.griddata(np.array([x_grid.flatten(), y_grid.flatten()]).transpose(), dy_Psi_grid.flatten(),
                                    D_ls.transpose(),
                                    method='nearest').flatten()
        mag_grad_Psi = sciinterp.griddata(np.array([x_grid.flatten(), y_grid.flatten()]).transpose(),
                                          mag_grad_Psi_grid.flatten(),
                                          D_ls.transpose(),
                                          method='nearest').flatten()

        fext = 0 * ones_f
        fext[::2] = F_ext * dx_Psi / mag_grad_Psi
        fext[1::2] = F_ext * dy_Psi / mag_grad_Psi
        # d = sciinterp.griddata(np.array([x_grid.flatten(), y_grid.flatten()]).transpose(), d.flatten(), D_ls.transpose(),
        #                       method='nearest').flatten()

        md.set_variable('fext', fext)
        md.enable_variable('psi')
        md.solve('max_res', 1E-10, 'max_iter', 10, 'noisy')
        Previous_Ls = Ls1
        Ls1 = md.variable('psi')
        md.disable_variable('psi')

    Previous_u = U
    U = ones_u * 0
    U[eval(ind_u)] = md.variable('u')

    Previous_p = P
    P = ones_p * 0
    P[eval(ind_p)] = md.variable('p')

    if ins.temp:
        Previous2_T = Previous_T
        Previous_T = T
        T = ones_t * ins.T_atm
        T[eval(ind_t)] = md.variable('t')

        T_ls = compute_interpolate_on(mft,T,mfls)
        if ins.solidification:
            ls2.set_values((T_ls - Tg) / Tg)

            md.enable_variable('d')

            md.solve('max_res', 1E-10, 'max_iter', 10, 'noisy')

            D = ones_d * 0
            D[eval(ind_u)] = md.variable('d')

            if np.max(T) <= Tg:
                d_ls2 = np.ones(x_grid.shape) * (-2 * np.max([ins.L_x, ins.L_y]))
            else:
                Ls2_ext = sciinterp.griddata(D_ls.transpose(), ls2.values(0),
                                             np.array([x_grid.flatten(), y_grid.flatten()]).transpose(),
                                             method='nearest')
                d_ls2 = skfmm.distance(Ls2_ext.reshape(x_grid.shape), dx=[dy, dx])

            if ins.free_surface:
                d = np.maximum(np.array(d_ls1), np.array(d_ls2))
            else:
                d = d_ls2

            d, D_ext_x = skfmm.extension_velocities(d, compute_interpolate_on(mfu, D, mfls)[0,:].reshape(x_grid.shape),
                                                      dx=[dy, dx])
            d, D_ext_y = skfmm.extension_velocities(d, compute_interpolate_on(mfu, D, mfls)[1, :].reshape(x_grid.shape),
                                                    dx=[dy, dx])
            D_ext_x = sciinterp.griddata(np.array([x_grid.flatten(), y_grid.flatten()]).transpose(), D_ext_x.flatten(),
                                       D_u.transpose(),
                                       method='nearest').flatten()
            D_ext_y = sciinterp.griddata(np.array([x_grid.flatten(), y_grid.flatten()]).transpose(), D_ext_y.flatten(),
                                         D_u.transpose(),
                                         method='nearest').flatten()
            D_ext = D_ext_x
            D_ext[1::2] = D_ext_y[1::2]
            Previous_Ls_u = compute_interpolate_on(mfls,Previous_Ls,mfu)
            Ls1_u = compute_interpolate_on(mfls, Ls1, mfu)
            D[(Previous_Ls_u>=0)&(Ls1_u<=0)] = D_ext[(Previous_Ls_u>=0)&(Ls1_u<=0)]

            md.disable_variable('d')

        else:
            Previous_d = d_init
            D = d_init
    else:
        Previous_d = d_init
        D = d_init

    md.enable_variable('u')
    md.enable_variable('p')
    if ins.temp:
        md.enable_variable('t')

    if ((i + 1) % ins.noutput == 0) or (np.abs(ti-ins.tf)<ins.dt):
        print('Time = %g' % ti)
        numstr = str(ti * 10 ** ndecimal).split('.')[0].zfill(ndigits)
        # print('Average temperature %g' % np.mean(T))

        if ins.vtk:
            u = md.variable('u')
            P = md.variable('p')

            mfu.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_u_' + numstr + '.vtk', u)
            if ins.true_u:
                mfu.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_utrue_' + numstr + '.vtk',
                                  eval(ins.true_u))
            mfp.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_P_' + numstr + '.vtk', P)
            if ins.true_p:
                mfp.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_Ptrue_' + numstr + '.vtk',
                                  eval(ins.true_p))
            if ins.free_surface:
                mfls.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_Ls1_' + numstr + '.vtk', md.variable('Previous_psi'))
                if ins.true_ls1:
                    mfls.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_Ls1true_' + numstr + '.vtk',
                                      eval(ins.true_ls1))
                mff.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_Fext_' + numstr + '.vtk', fext)
                mfls.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_curvature_' + numstr + '.vtk',
                                   sciinterp.griddata(np.array([x_grid.flatten(), y_grid.flatten()]).transpose(),
                                                      curvature.flatten(),
                                                      D_ls.transpose(),
                                                      method='nearest'))
                mfmat.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_rho_' + numstr + '.vtk', rho)
            if ins.temp:
                T = md.variable('t')
                mft.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_T_' + numstr + '.vtk', T)
                if ins.true_t:
                    mft.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_Ttrue_' + numstr + '.vtk',
                                      eval(ins.true_t))
                mfmat.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_mu_' + numstr + '.vtk', mu)
                if ins.solidification:
                    mfu.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_d_' + numstr + '.vtk', D)
                    if ins.true_d:
                        mfu.export_to_vtk(
                            outfile + '/' + ins.outfile.split('/')[-1] + '_dtrue_' + numstr + '.vtk',
                            eval(ins.true_d))
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
    T_ls = compute_interpolate_on(mft,T,mfls)
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
Ls1 = md.variable('psi')
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