import sys
import os
import json
from types import SimpleNamespace
import h5py

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate as sciinterp
import skfmm
import alphashape
import getfem as gf
from getfem import *
from curvature import *
from math import erf
from shapely.plotting import plot_polygon


#outfiles = []
#for dti in ['1','033','01','0033','001']:
#    outfiles = np.append(outfiles, 'cavity_60x_' + dti + 't_unsteady')

#outfiles = []
#for ei in ['_1e','_03e','','_003e','_001e']:
#    outfiles = np.append(outfiles, 'surface_relaxation_2_1Lx_' + '59x_01t' + ei )
#for Ki in ['-3','-4','-6', '-7']:
#    outfiles = np.append(outfiles, 'surface_relaxation_2_1Lx_' + '59x_01t_' + Ki + 'K')
#for nxi in ['119','120']:
#    for dti in ['1','033','01']:
#        outfiles = np.append(outfiles,'surface_relaxation_2_1Lx_' + nxi+'x_'+dti+'t')

#outfiles = []
#for hi in ['30','60','120']:
#    for dti in ['1','033','01','0033','001']:
#        outfiles = np.append(outfiles, 'stefan_iso_nosolid_' + hi + 'x_' + dti + 't')
#for hi in ['60']:
#    for dti in ['01']:
#        outfiles = np.append(outfiles, 'stefan_iso_nosolid_' + hi + 'x_' + dti + 't')

#outfiles = []
#for dxi in ['29','59','119']:
#    for dti in ['1','03','01']:
#        outfiles = np.append(outfiles, 'stefan_iso_steady_' + dxi + 'x_' + dti + 't')

#outfiles = ['stefan_solid_30x_001t']
outfiles = []
for dxi in ['59']:
    for dti in ['01']:
        outfiles = np.append(outfiles, 'spreading_drop_iso_' + dxi + 'x_' + dti + 't')

#outfiles = ['syracuse']
#outfiles = ['surface_relaxation_2_1Lx_29x_1t_topo']

for outfilei in outfiles:
    print(outfilei)
    outfile = './Results/' + outfilei

    # import simulation parameters
    with open(outfile + '/' + outfile.split('/')[-1] + '.json', 'r') as openfile:
        dictionary = json.load(openfile)

    ins = SimpleNamespace(**dictionary)

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

    if type(ins.ndigits) == int:
        ndigits = ins.ndigits
        ndecimal = len(str(ins.dt)[:ndigits+2].split('.')[-1].rstrip('0'))
    else:
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
            err_u_i = hf.get('err_u')[:]
            err_u = np.zeros(int(np.ceil(ins.tf / ins.dt)))
            err_u[:len(err_u_i)] = err_u_i

            P = hf.get('last_p')[:]
            Previous_p = hf.get('last2_p')[:]
            err_p_i = hf.get('err_p')[:]
            err_p = np.zeros(int(np.ceil(ins.tf / ins.dt)))
            err_p[:len(err_p_i)] = err_p_i

            D = hf.get('last_d')[:]
            Previous_d = hf.get('last2_d')[:]
            err_d_i = hf.get('err_d')[:]
            err_d = np.zeros(int(np.ceil(ins.tf / ins.dt)))
            err_d[:len(err_d_i)] = err_d_i

            if ins.temp:
                T = hf.get('last_T')[:]
                Previous_t = hf.get('last2_T')[:]
                err_t_i = hf.get('err_t')[:]
                err_t = np.zeros(int(np.ceil(ins.tf / ins.dt)))
                err_t[:len(err_t_i)] = err_t_i
            if ins.free_surface:
                last_Ls1 = hf.get('last_Ls1')[:]
                last2_Ls1 = hf.get('last2_Ls1')[:]
                err_ls1_i = hf.get('err_ls1')[:]
                err_ls1 = np.zeros(int(np.ceil(ins.tf / ins.dt)))
                err_ls1[:len(err_ls1_i)] = err_ls1_i
                expected_area = hf.get('expected_area')[:]
            if ins.temp & ins.solidification:
                last_Ls2 = hf.get('last_Ls2')[:]
            if ins.topography:
                last_Ls3 = hf.get('last_Ls3')[:]
            last_ti = hf.get('last_ti')[:]
            hf.close()
        except Exception as e:
            print(e)
            ins.restart = False
            hf.close()
            breakpoint()

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
        fb_bottomleft = mesh.outer_faces_in_box([-dx, -dy], [dx, dy])  # boundary 5
        fb_bottomright = mesh.outer_faces_in_box([ins.L_x-dx, -dy], [ins.L_x+dx, dy])  # boundary 5
        fb_topleft = mesh.outer_faces_in_box([-dx, ins.L_y-dy], [dx, ins.L_y + dy])  # boundary 6

        if ins.influx:
            fb_influx = mesh.faces_from_cvid(mesh.convexes_in_box((ins.influx_left, ins.influx_bottom),
                                                                  (ins.influx_right, ins.influx_top)))

    bounds = ['left', 'right', 'top', 'bottom']
    bounds_type = []
    for i, bound in enumerate(bounds):
        mesh.set_region(i + 1, eval('fb_' + bound))
        if ins.temp:
            bounds_type = np.append(bounds_type, type(eval('ins.' + bound + '_t')))
        else:
            bounds_type = np.append(bounds_type, type(eval('ins.' + bound + '_ux')))

    mesh.set_region(5,fb_bottomleft)
    #mesh.set_region(6,fb_topleft)

    if ins.influx:
        mesh.set_region(7, fb_influx)

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
        mls2 = gf.MeshLevelSet(mesh)
        ls2 = gf.LevelSet(mesh, ins.ls_k, ins.ls2p, ins.ls2s)
        mls2.add(ls2)
        mls2.adapt()

    # topography
    if ins.topography:
        if not ins.free_surface:
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

    if ins.free_surface | ins.topography:
        mls.adapt()

    if ins.free_surface:
        ls1_init = ls1.values(0)

    if ins.plots:
        # visualize mesh points in python
        if ins.free_surface | ins.topography:
            [x_ls, y_ls] = mls.cut_mesh().pts()  # get nodes
        else:
            [x_ls, y_ls] = mesh.pts()  # get nodes
        ax.scatter(x_ls, y_ls, 50, 'r', zorder=2)  # plot nodes

        # plot all mesh edges (no overlap)
        if ins.free_surface | ins.topography:
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
                    bound_type = type(eval('ins.' + bound + '_t'))
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
        plt.close(fig)

    if ins.free_surface | ins.topography:
        mim_all = gf.MeshIm('levelset', mls, 'all', gf.Integ('IM_TRIANGLE(5)'))
        mim_all.set_integ(4)
        if ins.topography:
            if ins.solve_topography:
                if ins.solve_air | (not ins.free_surface):
                    mim = gf.MeshIm('levelset', mls, 'all', gf.Integ('IM_TRIANGLE(5)'))
                else:
                    mim = gf.MeshIm('levelset', mls, 'inside(a+b)', gf.Integ('IM_TRIANGLE(5)'))
            else:
                if ins.solve_air | (not ins.free_surface):
                    mim = gf.MeshIm('levelset', mls, 'outside(b)', gf.Integ('IM_TRIANGLE(5)'))
                else:
                    mim = gf.MeshIm('levelset', mls, 'inside(a-b)', gf.Integ('IM_TRIANGLE(5)'))
        else:
            if ins.solve_air:
                mim = gf.MeshIm('levelset', mls, 'all', gf.Integ('IM_TRIANGLE(5)'))
            else:
                mim = gf.MeshIm('levelset', mls, 'inside(a)', gf.Integ('IM_TRIANGLE(5)'))
        mim.set_integ(4)

        if ins.free_surface:
            mim_surf = gf.MeshIm('levelset', mls, 'boundary(a+b)', gf.Integ('IM_TRIANGLE(5)'))
            mim_surf.set_integ(4)

        if ins.topography:
            mim_base = gf.MeshIm('levelset', mls, 'boundary(b)', gf.Integ('IM_TRIANGLE(5)'))
            mim_base.set_integ(4)

        mim_integ = gf.MeshIm('levelset', mls, 'boundary(a-b)', gf.Integ('IM_TRIANGLE(5)'))
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

    mfls = gf.MeshFem(mesh, 1)
    mfls.set_classical_fem(ins.ls_k)
    mff = gf.MeshFem(mesh, 2)
    mff.set_classical_fem(ins.ls_k)

    mfmat = gf.MeshFem(mesh,1)
    mfmat.set_classical_fem(ins.nmat)

    # pressure
    mfp0 = gf.MeshFem(mesh, 1)
    mfp0.set_classical_fem(ins.p_k)

    mfp_cutoff0 = gf.MeshFem(mesh, 1)
    mfp_cutoff0.set_classical_fem(0)
    if ins.free_surface:
        mf_cutoff = gf.GlobalFunction('cutoff',1,np.sqrt(dx**2 + dy**2),np.sqrt(dx**2 + dy**2)/100,np.sqrt(dx**2 + dy**2))
        mf_cutoff_ls1 = gf.MeshFem('global function',mesh,ls1,[mf_cutoff],2)
        DOFpts_p = mfp0.basic_dof_nodes()
        mfp_cutoff = gf.MeshFem('product', mf_cutoff_ls1, mfp_cutoff0)

        enriched_dofs = (np.nonzero(np.abs(compute_interpolate_on(mfls,ls1.values(0),DOFpts_p))<(np.sqrt(dx**2 + dy**2)))[0]+1)
        enriched_dofs = enriched_dofs[enriched_dofs>1]

        mfp_cutoff.set_enriched_dofs(enriched_dofs)
        mfp = gf.MeshFem('sum',mfp_cutoff,mfp0)
    else:
        mfp = gf.MeshFem(mesh, 1)  # scalar field
        mfp.set_classical_fem(ins.p_k)  # continuous piecewise linear

    # velocity
    mfu = gf.MeshFem(mesh, 2)  # vector field
    mfu.set_classical_fem(ins.u_k)  # continuous piecewise quadratic

    # temperature
    if ins.temp:
        if ins.t_enrich:
            if ins.free_surface:
                mft0 = gf.MeshFem(mesh, 1)
                mft0.set_classical_fem(ins.t_k)

                mft_cutoff0 = gf.MeshFem(mesh, 1)
                mft_cutoff0.set_classical_fem(0)

                DOFpts_t = mft0.basic_dof_nodes()
                mft_cutoff = gf.MeshFem('product', mf_cutoff_ls1, mft_cutoff0)

                enriched_dofs = (np.nonzero(
                    np.abs(compute_interpolate_on(mfls, ls1.values(0), DOFpts_t)) < (np.sqrt(dx ** 2 + dy ** 2)))[
                                     0] + 1)
                enriched_dofs = enriched_dofs[enriched_dofs > 1]

                mft_cutoff.set_enriched_dofs(enriched_dofs)
                mft = gf.MeshFem('sum', mft_cutoff, mft0)
            else:
                mft = gf.MeshFem(mesh, 1)  # scalar field
                mft.set_classical_fem(ins.t_k)  # continuous piecewise linear

        else:
            mft = gf.MeshFem(mesh, 1)  # scalar field
            mft.set_classical_fem(ins.t_k)  # continuous piecewise quadratic



    # above topography
    if ins.free_surface | ins.topography:
        ind_p = 'mfp.basic_dof_from_cv(np.append(mim_integ.convex_index(),mim.convex_index()))'
        ind_u = 'mfu.basic_dof_from_cv(np.append(mim_integ.convex_index(),mim.convex_index()))'
        if ins.temp:
            ind_t = 'mft.basic_dof_from_cv(np.append(mim_integ.convex_index(),mim.convex_index()))'
    else:
        ind_p = 'mfp.basic_dof_from_cv(mim_all.convex_index())'
        ind_u = 'mfu.basic_dof_from_cv(mim_all.convex_index())'
        if ins.temp:
            ind_t = 'mft.basic_dof_from_cv(mim_all.convex_index())'

    # get mesh coordinates
    D_u = mfu.basic_dof_nodes()
    ones_u = np.ones(D_u.shape[1])
    x_u = D_u[0, :]
    y_u = D_u[1, :]

    D_p = mfp.basic_dof_nodes()
    ones_p = np.ones(D_p.shape[1])
    x_p = D_p[0, :]
    y_p = D_p[1, :]

    D_p0 = mfp0.basic_dof_nodes()
    ones_p0 = np.ones(D_p0.shape[1])
    x_p0 = D_p0[0, :]
    y_p0 = D_p0[1, :]

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

    if ins.temp:
        D_t = mft.basic_dof_nodes()
        ones_t = np.ones(D_t.shape[1])
        x_t = D_t[0, :]
        y_t = D_t[1, :]

        if ins.plots:
            if ins.free_surface:
                ax.tricontour(x_ls, y_ls, ls1.values(0), levels=[0], colors='k')
                cbar = ax.tricontourf(x_ls, y_ls, ls1.values(0), levels=[-10, 0, 10])
                # plt.colorbar(cbar,ax=ax)

    if ins.plots:
        ax2.set_xlim(ax.get_xlim())
        ax2.set_ylim(ax.get_ylim())

        ax2.set_xlabel('x (m)')
        ax2.set_ylabel('y (m)')

    if ins.free_surface:
        pts,idx = mls.cut_mesh().pts_from_cvid(CVIDs = mls.cut_mesh().region(0)[0])
        ls1_interface = compute_interpolate_on(mfls,ls1.values(0),pts)
        if ins.topography:
            ls3_interface = compute_interpolate_on(mfls, ls3.values(0), pts)
            pts = pts[:,((np.abs(ls1_interface)<=0)&(ls3_interface>=-0))|((ls1_interface<=0)&(np.abs(ls3_interface)<=0))]
        else:
            pts = pts[:, ((np.abs(ls1_interface) <= np.sqrt(dx ** 2 + dy ** 2) / 100))]

        if np.min(np.abs(ls1.values(0)[edges_ls]))<np.sqrt(dx**2 + dy**2):
            area_init = 1
        else:
            area_init = alphashape.alphashape(pts.transpose(), 2 * np.sqrt(dx ** 2 + dy ** 2)).area
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


    ######## Problem setup ################

    md_init = gf.Model('real')  # real vs complex system
    md = gf.Model('real')  # real vs complex system

    # Add Data

    if ins.restart:
        md.add_initialized_fem_data('Previous_u', mfu, U)
        md.add_initialized_fem_data('Previous2_u', mfu, Previous_u)
        md.add_initialized_fem_data('Previous_p', mfp, P)
        md.add_initialized_fem_data('Previous2_p', mfp, Previous_p)
        md.add_initialized_fem_data('Previous_d', mfu, D)
        md.add_initialized_fem_data('Previous2_d', mfu, Previous_d)
        md.add_initialized_fem_data('Previous_psi', mfls, last2_Ls1)
        md.add_initialized_fem_data('Previous2_psi', mfls, last_Ls1)
    else:
        u_init = ones_u * 0
        p_init = ones_p0 * ins.p_atm
        d_init = ones_u * 0
        md.add_initialized_fem_data('Previous_u', mfu, u_init)
        md.add_initialized_fem_data('Previous2_u', mfu, u_init)
        md.add_initialized_fem_data('p_init', mfp, p_init)
        md.add_initialized_fem_data('Previous_p', mfp, p_init)
        md.add_initialized_fem_data('Previous2_p', mfp, p_init)
        md.add_initialized_fem_data('Previous_d', mfu, d_init)
        md.add_initialized_fem_data('Previous2_d', mfu, d_init)
        if ins.free_surface:
            psi_init = ls1.values(0)
            md.add_initialized_fem_data('Previous_psi', mfls, psi_init)
            md.add_initialized_fem_data('Previous2_psi', mfls, psi_init)

    fext = 0*ones_f
    md.add_initialized_fem_data('fext',mff,0*ones_f)
    md.add_initialized_fem_data('Previous_fext', mff, 0 * ones_f)

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
    rho_init = rho.copy()
    md.add_initialized_fem_data('rho', mfmat, rho)
    md.add_initialized_fem_data('Previous_rho', mfmat, rho)
    md.add_initialized_fem_data('rho_init', mfmat, rho)
    md_init.add_initialized_fem_data('rho', mfmat, rho)

    # Compressibility
    beta = ones_mat * ins.beta1
    if ins.free_surface:
        beta[ls1_mat > 0] = ins.beta2
    if ins.topography:
        beta[ls3_mat < 0] = ins.beta3
    beta_init = beta.copy()
    md.add_initialized_fem_data('beta', mfmat, beta)
    md.add_initialized_fem_data('Previous_beta', mfmat, beta)

    if ins.temp:
        # Thermal diffusivity
        kappa = ones_mat * ins.kappa1
        if ins.free_surface:
            kappa[ls1_mat > 0] = ins.kappa2
        if ins.topography:
            kappa[ls3_mat < 0] = ins.kappa3
        kappa_init = kappa.copy()
        md.add_initialized_fem_data('kappa', mfmat, kappa)
        md.add_initialized_fem_data('Previous_kappa', mfmat, kappa)
        md.add_initialized_data('cp', [ins.cp])

    # Temperature
    if ins.temp:

        # Initial temp
        if ins.restart:
            md.add_initialized_fem_data('Previous_t', mft, T)
            md.add_initialized_fem_data('Previous2_t', mft, Previous_t)
            t_init = Previous_t

        else:
            if (type(ins.T_init) is float) or (type(ins.T_init) is int):
                t_init = mft.eval('1') * ins.T_init
            elif type(ins.T_init) is str:
                t_init = eval(ins.T_init)
            elif type(ins.T_init) is type(None):
                 t_init = mft.eval('1') * ins.T0
            if ins.free_surface:
                ls1_t = compute_interpolate_on(mfls,ls1.values(0),mft)
                t_init[ls1_t > np.sqrt(dx**2 + dy**2)/100] = ins.T_atm
            if ins.topography:
                ls3_t = compute_interpolate_on(mfls, ls3.values(0), mft)
                t_init[ls3_t < 0] = ins.T_atm

            md.add_initialized_fem_data('Previous_t', mft, t_init)
            md.add_initialized_fem_data('Previous2_t', mft, t_init)

            T_mat = compute_interpolate_on(mft, t_init, mfmat)
            eta_exp = ins.eta_exp.replace('vfta', str(ins.vfta)).replace('vftb', str(ins.vftb)).replace(
                'vftc', str(ins.vftc)).replace('etar', str(ins.etar))
            eta = eval(eta_exp.replace('exp', 'np.exp').replace('T', 'T_mat'))
            eta[T_mat <= ins.vftb/(np.log(ins.max_eta)-ins.vfta) - 273 + ins.vftc] = ins.max_eta

        T_ls = compute_interpolate_on(mft, t_init, mfls)
        if ins.solidification:
            ls2.set_values((T_ls - Tg) / Tg)
            mls2.adapt()

        if ins.free_surface:
            eta[ls1_mat > 0] = ins.eta2
        if ins.topography:
            eta[ls3_mat < 0] = ins.eta3
    else:
        eta = eval(ins.eta_exp.replace('exp', 'np.exp').replace('T', str(ins.T0)).replace('vft','ins.vft')) * ones_mat
        if ins.free_surface:
            eta[ls1_mat > 0] = ins.eta2
        if ins.topography:
            eta[ls3_mat < 0] = ins.eta3

    # time dependence
    md.add_initialized_data('dt', [ins.dt/20])
    md.add_initialized_data('BDF0', [1])
    md.add_initialized_data('BDF1', [-1])
    md.add_initialized_data('BDF2', [0])
    md.add_initialized_data('BDFf', [1])

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

    md.add_initialized_fem_data('Previous_lambda', mfmat, lam)
    md.add_initialized_fem_data('Previous_mu', mfmat, mu)
    md.add_initialized_fem_data('Previous_solid', mfmat, solid)

    md_init.add_initialized_fem_data('lambda', mfmat, lam)
    md_init.add_initialized_fem_data('mu', mfmat, mu)
    md_init.add_initialized_fem_data('solid', mfmat, solid)

    if ins.free_surface | ins.topography:
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

    if ins.free_surface:
        md.add_fem_variable('psi',mfls)
        md.add_fem_variable('psis',mfls)
        md.add_initialized_data('psik',ins.kappa_psi)

        Psi_grid = sciinterp.griddata(D_ls.transpose(), ls1.values(0),
                                           np.array([x_grid.flatten(), y_grid.flatten()]).transpose(), method='linear').reshape(x_grid.shape)

        dx_Psi_grid,dy_Psi_grid,curvature,mag_grad_Psi_grid = compute_curvature(Psi_grid, dx, dy)
        md.add_initialized_fem_data('curvature',mfls,
                                    sciinterp.griddata(np.array([x_grid.flatten(), y_grid.flatten()]).transpose(), curvature.flatten(),
                                           D_ls.transpose(),
                                           method='nearest').flatten())
        md_init.add_initialized_fem_data('curvature',mfls,
                                    sciinterp.griddata(np.array([x_grid.flatten(), y_grid.flatten()]).transpose(), curvature.flatten(),
                                           D_ls.transpose(),
                                           method='nearest').flatten())
    md.add_initialized_data('surface_tension',ins.surface_tension)
    md_init.add_initialized_data('surface_tension',ins.surface_tension)

    # Define variables and weak forms

    md_init.add_fem_variable('u', mfu)
    md_init.add_fem_variable('p', mfp)

    md_init.add_linear_incompressibility_brick(mim_all, 'u', 'p')
    md_init.add_isotropic_linearized_elasticity_brick(mim_all, 'u', 'lambda*(dt*solid + (1-solid))', 'mu*(dt*solid + (1-solid))')
    #if ins.free_surface:
    #    md_init.add_nonlinear_term(mim_surf, "1e8*((Xfem_plus(p)-Xfem_minus(p))+surface_tension*curvature).(Xfem_plus(Test_p)-Xfem_minus(Test_p))")

    md.add_macro('h', 'element_size')

    # compressible mass balance
    if ins.compressible:
        md.add_macro('rhof', 'rho_init*exp(beta*(p-p_init))')

        time_int_p = '(BDF0*p+BDF1*Previous_p+BDF2*Previous2_p)/dt'

        md.add_nonlinear_term(mim, time_int_p + '*Test_p')
        md.add_nonlinear_term(mim, '(u.Grad_p)*Test_p + 1/beta*Div_u*Test_p')
        md.add_nonlinear_term(mim, 'BDFf*((Previous_u.Grad_Previous_p)*Test_p + 1/Previous_beta*Div_Previous_u*Test_p)')
        if 'SUPG' in ins.stab_p:
            S_SUPG_p = "(beta*(u.Grad_p)*(u.Grad_Test_p) + Div_u*(u.Grad_Test_p))"
            tau_SUPG_p = '1/(2/dt + 2*Norm(u)/h)'
            md.add_nonlinear_term(mim, tau_SUPG_p + '*' + time_int_p + '*(u.Grad_Test_p)')
            md.add_nonlinear_term(mim, tau_SUPG_p + '*' + S_SUPG_p)

            S_SUPG_p_init = "(Previous_beta*(Previous_u.Grad_Previous_p)*(Previous_u.Grad_Test_p) + Div_Previous_u*(Previous_u.Grad_Test_p))"
            tau_SUPG_p_init = 'BDFf*(1/(2/dt + 2*Norm(Previous_u)/h))'
            md.add_nonlinear_term(mim, tau_SUPG_p_init + '*' + time_int_p + '*(Previous_u.Grad_Test_p)')
            md.add_nonlinear_term(mim, tau_SUPG_p_init + '*' + S_SUPG_p_init)

        elif 'GLS' in ins.stab_p:
            S_GLS_p = "(u.Grad_p)*(Trace(Grad(u*Test_p)))"
            tau_GLS_p = '1/(2/dt + 2*Norm(u)/h)'
            md.add_nonlinear_term(mim, tau_GLS_p + '*' + time_int_p + '*(Trace(Grad(u*Test_p)))')
            md.add_nonlinear_term(mim, tau_GLS_p + '*' + S_GLS_p)

            S_GLS_p_init = "(Previous_u.Grad_Previous_p)*(Trace(Grad(Previous_u*Test_p)))"
            tau_GLS_p_init = '1/(2/dt + 2*Norm(Previous_u)/h)'
            md.add_nonlinear_term(mim, tau_GLS_p_init + '*' + time_int_p + '*(Trace(Grad(Previous_u*Test_p)))')
            md.add_nonlinear_term(mim, tau_GLS_p_init + '*' + S_GLS_p_init)


    else:
        md.add_linear_incompressibility_brick(mim, 'u', 'p')
    #if ins.free_surface:
    #    md.add_nonlinear_term(mim_surf, "1e8*((Xfem_plus(p)-Xfem_minus(p))+surface_tension*curvature).(Xfem_plus(Test_p)-Xfem_minus(Test_p))")

    # mometum balance
    if ins.compressible:
        time_int_u = "rhof*((BDF0*u+BDF1*Previous_u+BDF2*Previous2_u)/dt.Test_u)"
    else:
        time_int_u = "rho*((BDF0*u+BDF1*Previous_u+BDF2*Previous2_u)/dt.Test_u)"

    if ins.steady:
        md.add_nonlinear_term(mim, 'solid*' + time_int_u)
    else:
        md.add_nonlinear_term(mim, time_int_u)

    linear_elastic = "((lambda*(Div_u*Div_Test_u) + mu*((Grad_u + Grad_u'):Grad_Test_u)))"
    linear_elastic_init = "BDFf*(Previous_lambda*(Div_Previous_u*Div_Test_u) + Previous_mu*((Grad_Previous_u + Grad_Previous_u'):Grad_Test_u))"

    if ins.temp & ins.solidification:
        md.add_nonlinear_term(mim, linear_elastic + '*(dt/BDF0*solid + (1-solid))')
        md.add_nonlinear_term(mim, linear_elastic_init + '*(dt/BDF0*Previous_solid + (1-Previous_solid))')

        linear_elastic_adv = "(lambda*(Trace(Grad(u.Grad_Previous_d))*Div_Test_u) + mu*((Grad(u.Grad_Previous_d) + Grad(u.Grad_Previous_d)'):Grad_Test_u))*(dt/BDF0*solid)"
        residual_stress1 = "-(lambda*(Div_Previous_d*Div_Test_u) + mu*((Grad_Previous_d + Grad_Previous_d'):Grad_Test_u))*(BDF1/BDF0)*solid"
        residual_stress2 = "-(lambda*(Div_Previous2_d*Div_Test_u) + mu*((Grad_Previous2_d + Grad_Previous2_d'):Grad_Test_u))*(BDF2/BDF0)*solid"

        md.add_nonlinear_term(mim, linear_elastic_adv)
        md.add_nonlinear_term(mim, residual_stress1 + '+' + residual_stress2)

    else:
        md.add_nonlinear_term(mim, linear_elastic)
        md.add_nonlinear_term(mim, linear_elastic_init)

    if ins.free_surface:
        md.add_nonlinear_term(mim_all,"psis*Test_psis - psi*Test_psis + dt*psik*Grad_psis.Grad_Test_psis")

        time_int_psi = "((BDF0*psi+BDF1*Previous_psi+BDF2*Previous2_psi)/dt)"
        advection_psi = "(fext.Grad_psi)*Test_psi"
        advection_psi_init = "BDFf*(Previous_fext.Grad_Previous_psi)*Test_psi"
        md.add_nonlinear_term(mim_all, time_int_psi + '*Test_psi')
        md.add_nonlinear_term(mim_all, advection_psi)
        md.add_nonlinear_term(mim_all, advection_psi_init)

        if 'SUPG' in ins.stab_psi:
            tau_SUPG_psi = "1/(2/dt + 2*Norm(fext)/h)"
            S_SUPG_psi = "(fext.Grad_psi).(fext.Grad_Test_psi)"
            md.add_nonlinear_term(mim, tau_SUPG_psi + '*' + time_int_psi + '.(fext.Grad_Test_psi)')
            md.add_nonlinear_term(mim, tau_SUPG_psi + '*' + S_SUPG_psi)

            tau_SUPG_psi_init = "BDFf/(2/dt + 2*Norm(Previous_fext)/h)"
            S_SUPG_psi_init = "(Previous_fext.Grad_Previous_psi).(Previous_fext.Grad_Test_psi)"
            md.add_nonlinear_term(mim, tau_SUPG_psi_init + '*' + time_int_psi + '.(Previous_fext.Grad_Test_psi)')
            md.add_nonlinear_term(mim, tau_SUPG_psi_init + '*' + S_SUPG_psi_init)

        elif 'GLS' in ins.stab_psi:
            tau_GLS_psi = "1/(2/dt + 2*Norm(fext)/h)"
            S_GLS_psi = "((fext.Grad_psi)*(Trace(Grad(fext*Test_psi))))"
            md.add_nonlinear_term(mim, tau_GLS_psi + '*' + time_int_psi + '.(Trace(Grad(fext*Test_psi)))')
            md.add_nonlinear_term(mim, tau_GLS_psi + '*' + S_GLS_psi)

            tau_GLS_psi_init = "BDFf/(2/dt + 2*Norm(Previous_fext)/h)"
            S_GLS_psi_init = "((Previous_fext.Grad_Previous_psi)*(Trace(Grad(Previous_fext*Test_psi))))"
            md.add_nonlinear_term(mim, tau_GLS_psi_init + '*' + time_int_psi + '.(Trace(Grad(Previous_fext*Test_psi)))')
            md.add_nonlinear_term(mim, tau_GLS_psi_init + '*' + S_GLS_psi_init)

        md.disable_variable('psis')
        md.disable_variable('psi')

    if ins.temp:
        time_int_t = "(BDF0*t+BDF1*Previous_t+BDF2*Previous2_t)/dt"
        advection_t = "(u.Grad_t)*Test_t"
        advection_t_init = "BDFf*(Previous_u.Grad_Previous_t)*Test_t"
        diffusion = "kappa*(Grad_t.Grad_Test_t)"
        diffusion_init = "BDFf*Previous_kappa*(Grad_Previous_t.Grad_Test_t)"

        S_SUPG_t = "((u.Grad_t)*(u.Grad_Test_t) + kappa*(Grad_t).Grad(u.Grad_Test_t))"
        S_GLS_t = "((u.Grad_t)*(Trace(Grad(u*Test_t))) + kappa*(Grad_t).Grad(Trace(Grad(u*Test_t))) + (u.Grad_t)*(Trace(Grad(kappa*Grad_Test_t))))"

        S_SUPG_t_init = "((Previous_u.Grad_Previous_t)*(Previous_u.Grad_Test_t) + Previous_kappa*(Grad_Previous_t).Grad(Previous_u.Grad_Test_t))"
        S_GLS_t_init = "((Previous_u.Grad_Previous_t)*(Trace(Grad(Previous_u*Test_t))) + Previous_kappa*(Grad_Previous_t).Grad(Trace(Grad(Previous_u*Test_t)))+ (Previous_u.Grad_Previous_t)*(Trace(Grad(kappa*Grad_Test_t))))"

        md.add_macro('Pe', "h*Norm(u)/(2*kappa)")
        md.add_macro('xi', "min(Pe/3,1)")

        md.add_macro('Pe_init', "h*Norm(Previous_u)/(2*Previous_kappa)")
        md.add_macro('xi_init', "min(Pe_init/3,1)")

        md.add_nonlinear_term(mim, time_int_t + '*Test_t')
        md.add_nonlinear_term(mim, advection_t)
        md.add_nonlinear_term(mim, diffusion)
        md.add_nonlinear_term(mim, advection_t_init)
        md.add_nonlinear_term(mim, diffusion_init)

        tau_SUPG_t = '1/(2/dt + 4*kappa/(h*h) + 2*Norm(u)/h)*xi'
        tau_GLS_t = '1/(2/dt + 4*kappa/(h*h) + 2*Norm(u)/h)*xi'

        tau_SUPG_t_init = 'BDFf/(2/dt + 4*Previous_kappa/(h*h) + 2*Norm(Previous_u)/h)*xi_init'
        tau_GLS_t_init = 'BDFf/(2/dt + 4*Previous_kappa/(h*h) + 2*Norm(Previous_u)/h)*xi_init'
        if 'SUPG' in ins.stab_t:
            # tau_supg = 'h/(2*Norm(u))*(cosh(Norm(u)*h/(2*kappa))/sinh(Norm(u)*h/(2*kappa)) - 2*kappa/(h*Norm(u)))'
            # tau_SUPG_t = 'h*h/(4*kappa)*min(1/3,1/Pe)'
            md.add_nonlinear_term(mim, tau_SUPG_t + '*' + time_int_t + '*(u.Grad_Test_t)')
            md.add_nonlinear_term(mim, tau_SUPG_t + '*' + S_SUPG_t)

            md.add_nonlinear_term(mim, tau_SUPG_t_init + '*' + time_int_t + '*(Previous_u.Grad_Test_t)')
            md.add_nonlinear_term(mim, tau_SUPG_t_init + '*' + S_SUPG_t_init)

        elif 'GLS' in ins.stab_t:
            md.add_nonlinear_term(mim, tau_GLS_t + '*' + time_int_t + '*(Trace(Grad(u*Test_t)))')
            #md.add_nonlinear_term(mim, tau_GLS_t + '*(kappa*(Grad(' + time_int_t + ').Grad_Test_t))')
            md.add_nonlinear_term(mim, tau_GLS_t + '*' + S_GLS_t)

            md.add_nonlinear_term(mim, tau_GLS_t_init + '*' + time_int_t + '*(Trace(Grad(Previous_u*Test_t)))')
            #md.add_nonlinear_term(mim, tau_GLS_t_init + '*(Previous_kappa*(Grad(' + time_int_t + ').Grad_Test_t))')
            md.add_nonlinear_term(mim, tau_GLS_t_init + '*' + S_GLS_t_init)

    if ins.temp & ins.solidification:
        time_int_d = "((BDF0*d+BDF1*Previous_d+BDF2*Previous2_d)/dt)"
        advection_d = "(u.Grad_d).Test_d"
        md.add_nonlinear_term(mim, time_int_d + '.Test_d')
        #md.add_nonlinear_term(mim, advection_d)
        md.add_nonlinear_term(mim, "-u.Test_d - BDFf*(Previous_u.Test_d)")

        if 'SUPG' in ins.stab_d:
            tau_SUPG_d = "1/(2/dt + 2*Norm(u)/h)"
            S_SUPG_d = "(u.Grad_d):(u.Grad_Test_d)"
            md.add_nonlinear_term(mim, tau_SUPG_d + '*' + time_int_d + '.(u.Grad_Test_d)')
            md.add_nonlinear_term(mim, tau_SUPG_d + '*' + S_SUPG_d)

        elif 'GLS' in ins.stab_d:
            tau_GLS_d = "1/(2/dt + 2*Norm(u)/h)"
            S_GLS_d = "(u.Grad_d):(Trace(Grad(u.Test_d))"
            md.add_nonlinear_term(mim, tau_GLS_d + '*' + time_int_d + '.(Trace(Grad(u.Test_d))')
            md.add_nonlinear_term(mim, tau_GLS_d + '*' + S_GlS_d)

        #tau_SUPG_d_init = "BDFf/(2/dt + 2*Norm(Previous_u)/h)"
        #md.add_nonlinear_term(mim, tau_SUPG_d_init + '*' + time_int_d + '.(Previous_u.Grad_Test_d)')
    md.disable_variable('d')

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
    vars = ['u']

    for var in vars:
        for i, bound in enumerate(bounds):
            # Dirichlet boundaries
            dirichlet = False
            H = np.array([[0,0],[0,0]])
            if (type(eval('ins.' + bound + '_' + var + 'x')) is float) or (type(eval('ins.' + bound + '_' + var + 'x')) is int):
                data_x = eval('ins.' + bound + '_' + var + 'x') * ones_p0
                H[0,0] = 1
                dirichlet = True
            if type(eval('ins.' + bound + '_' + var + 'x')) is str:
                data_x = eval(eval('ins.' + bound + '_' + var + 'x').replace('x','x_p').replace('y','y_p'))
                dirichlet = True
                H[0,0] = 1
            if type(eval('ins.' + bound + '_' + var + 'y')) is type(None):
                data_y = 0 * ones_p0

            if (type(eval('ins.' + bound + '_' + var + 'y')) is float) or (type(eval('ins.' + bound + '_' + var + 'y')) is int):
                data_y = eval('ins.' + bound + '_' + var + 'y') * ones_p0
                dirichlet = True
                H[1,1] = 1
            if type(eval('ins.' + bound + '_' + var + 'y')) is str:
                data_uy = eval(eval('ins.' + bound + '_' + var + 'y').replace('x','x_p').replace('y','y_p'))
                dirichlet = True
                H[1,1] = 1
            if type(eval('ins.' + bound + '_' + var + 'x')) is type(None):
                data_x = 0 * ones_p0

            if dirichlet:
                md.add_initialized_fem_data(bound + 'data_' + var, mfp0, [data_x, data_y])
                md_init.add_initialized_fem_data(bound + 'data_' + var, mfp0, [data_x, data_y])
                md.add_initialized_data(bound + 'H_' + var,H)
                md_init.add_initialized_data(bound + 'H_' + var, H)

                md.add_generalized_Dirichlet_condition_with_multipliers(mim, var, 2, i + 1,
                                                            dataname=bound + 'data_' + var,Hname=bound + 'H_' + var)
                if 'u' in var:
                    md_init.add_generalized_Dirichlet_condition_with_multipliers(mim_all, var, 2, i + 1,
                                                                        dataname=bound + 'data_' + var, Hname=bound + 'H_' + var)


            else:
                # Neumann boundaries
                if (type(eval('ins.' + bound + '_d' + var + 'x')) is float) or (type(eval('ins.' + bound + '_d' + var + 'x')) is int):
                    data_dx = eval('ins.' + bound + '_d' + var + 'x') * ones_u
                if type(eval('ins.' + bound + '_d' + var + 'x')) is str:
                    data_dx = eval(eval('ins.' + bound + '_d' + var + 'x'))
                if type(eval('ins.' + bound + '_d' + var + 'y')) is type(None):
                    data_dy = 0 * ones_u

                if (type(eval('ins.' + bound + '_d' + var + 'y')) is float) or (type(eval('ins.' + bound + '_d' + var + 'y')) is int):
                    data_dy = eval('ins.' + bound + '_d' + var + 'y') * ones_u
                if type(eval('ins.' + bound + '_d' + var + 'y')) is str:
                    data_dy = eval(eval('ins.' + bound + '_d' + var + 'y'))
                if type(eval('ins.' + bound + '_d' + var + 'x')) is type(None):
                    data_dx = 0 * ones_u

                md.add_initialized_fem_data(bound + 'data_' + var, mfu, [data_dx, data_dy])
                md_init.add_initialized_fem_data(bound + 'data_' + var, mfu, [data_dx, data_dy])

                md.add_normal_source_term_brick(mim, var, bound + 'data_' + var, i + 1)
                if 'u' in var:
                    md_init.add_normal_source_term_brick(mim_all, var, bound + 'data_' + var, i + 1)

    # add pressure on boundary or free surface

    md.add_initialized_fem_data('patm', mfp0, [ins.p_atm*ones_p0])
    md_init.add_initialized_fem_data('patm', mfp0, [ins.p_atm*ones_p0])

    p_basal = ins.p_atm * ones_ls
    if ins.topography:
        p_basal -= ls3.values(0)*ins.rho3*9.81
    elif ins.free_surface:
        p_basal -= ls1.values(0)*ins.rho1*9.81

    md.add_initialized_fem_data('pbasal', mfls, [p_basal])
    md_init.add_initialized_fem_data('pbasal', mfls, [p_basal])

    #md_init.add_Dirichlet_condition_with_multipliers(mim, 'p', 1, 5, dataname='pbasal')
    #md.add_Dirichlet_condition_with_multipliers(mim, 'p', 1, 5, dataname='pbasal')

    md_init.add_Dirichlet_condition_with_multipliers(mim, 'p', 1, 5, dataname='pbasal')

    if (not ins.free_surface) | ins.solve_air:
        if ins.steady & (not ins.compressible):
            md.add_Dirichlet_condition_with_multipliers(mim, 'p', 1, 5, dataname='pbasal')
    else:
        md.add_Dirichlet_condition_with_multipliers(mim_surf, 'p', 1, -1, dataname='patm')

    if ins.topography:
        if 'no_slip' in ins.basal_velocity:
            md.add_initialized_fem_data('no_slip', mfp0, [0 * ones_p0, 0 * ones_p0])
            md.add_Dirichlet_condition_with_multipliers(mim_base, 'u', 2, -1, dataname='no_slip')

            md_init.add_initialized_fem_data('no_slip', mfp0, [0 * ones_p0, 0 * ones_p0])
            md_init.add_Dirichlet_condition_with_multipliers(mim_base, 'u', 2, -1, dataname='no_slip')

        if 'no_normal' in ins.basal_velocity:
            md.add_initialized_fem_data('no_normal', mfp0, [0 * ones_p0, 0 * ones_p0])
            md.add_normal_source_term_brick(mim_base, 'u', -1, dataname='no_normal')

            md_init.add_initialized_fem_data('no_normal', mfp0, [0 * ones_p0, 0 * ones_p0])
            md_init.add_normal_source_term_brick(mim_base, 'u', -1, dataname='no_normal')


    if ins.temp:
        if ins.solve_air | (not ins.free_surface):
            for i, bound in enumerate(bounds):
                dirichlet = False
                # Dirichlet boundaries
                if (type(eval('ins.' + bound + '_t')) is float) or (type(eval('ins.' + bound + '_t')) is int):
                    data_t = eval('ins.' + bound + '_t') * ones_t
                    dirichlet = True
                if type(eval('ins.' + bound + '_t')) is str:
                    data_t = eval(eval('ins.' + bound + '_t'))
                    dirichlet = True

                if dirichlet:
                    md.add_initialized_fem_data(bound + 'data_t', mft, [data_t])
                    md.add_Dirichlet_condition_with_multipliers(mim, 't', 2, i + 1,
                                                                dataname=bound + 'data_t')

                # Neumann boundaries
                else:
                    if (type(eval('ins.' + bound + '_dt')) is float) or (type(eval('ins.' + bound + '_dt')) is int):
                        data_t = -eval('ins.' + bound + '_dt') * ones_ls * kappa
                    if type(eval('ins.' + bound + '_dt')) is str:
                        data_t = -eval(eval('ins.' + bound + '_dt')) * kappa
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
                            t_init ** 4 - ins.T_atm ** 4)
            if 'forced convection' in ins.surface_flux:
                surface_flux_budget += ins.heat_transfer_coeff * (t_init - ins.T_atm)
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
                    md.add_initialized_fem_data('basal_flux', mft, [ins.kappa1 * (t_init - ins.basal_temp_i) / radii_t])
                    md.add_source_term_brick(mim_base, 't', 'basal_flux', -1)

    # influx
    if ins.influx:
        md.add_initialized_fem_data('influxdata', mfp, [eval(ins.influx_ux) * ones_p0, eval(ins.influx_uy) * ones_p0])
        md.add_Dirichlet_condition_with_multipliers(mim, 'u', 2, 7,
                                                    dataname='influxdata')
        if ins.temp:
            md.add_initialized_fem_data('influxdata_t', mft, [eval(ins.influx_t) * ones_t])
            md.add_Dirichlet_condition_with_multipliers(mim, 't', 2, 7,
                                                        dataname='influxdata_t')

    #### Solve ####
    print('Starting solve')
    if not ins.restart:
        # initialize pressure and velocity with incompressible, steady stokes

        if ins.steady | ins.steady_init:
            md_init.solve('max_res', 1E-10, 'max_iter', 10, 'noisy')

            p_init = md_init.variable('p')
            u_init = md_init.variable('u')

        md.set_variable('Previous_p', p_init)
        md.set_variable('Previous2_p', p_init)
        md.set_variable('Previous_u', u_init)
        md.set_variable('Previous2_u', u_init)

        err_u = np.zeros(int(np.ceil(ins.tf / ins.dt) + 1))
        err_p = np.zeros(int(np.ceil(ins.tf / ins.dt) + 1))
        err_ls1 = np.zeros(int(np.ceil(ins.tf / ins.dt) + 1))
        err_d = np.zeros(int(np.ceil(ins.tf / ins.dt) + 1))
        err_t = np.zeros(int(np.ceil(ins.tf / ins.dt) + 1))

        numstr = '0' * ndigits

        if ins.vtk:
            mfu.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_u_' + numstr + '.vtk', u_init)
            if ins.true_ux:
                true_u = eval(ins.true_ux.replace('x', 'x_u').replace('y', 'y_u'))
                true_u[1::2] = eval(ins.true_uy.replace('x', 'x_u').replace('y', 'y_u'))[1::2]
                mfu.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_utrue_' + numstr + '.vtk',
                                  true_u)
                err_u[0] = compute_L2_dist(mfu, u_init, mim, mfu, true_u) / compute_L2_norm(mfu,
                                                                                                      true_u, mim)
            mfp.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_P_' + numstr + '.vtk', p_init)
            if ins.true_p:
                mfp.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_Ptrue_' + numstr + '.vtk',
                                  eval(ins.true_p.replace('x', 'x_p').replace('y', 'y_p')))
                err_p[0] = compute_L2_dist(mfp, p_init, mim, mfp,
                                                          eval(ins.true_p.replace('x', 'x_p').replace('y',
                                                                                                      'y_p'))) / compute_L2_norm(
                    mfp,
                    eval(ins.true_p.replace('x', 'x_p').replace('y', 'y_p')), mim)
            if ins.free_surface:
                mfls.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_Ls1_' + numstr + '.vtk',
                                   ls1.values(0))
                if ins.true_ls1:
                    mfls.export_to_vtk(
                        outfile + '/' + ins.outfile.split('/')[-1] + '_Ls1true_' + numstr + '.vtk',
                        eval(ins.true_ls1.replace('X', 'x_ls').replace('Y', 'y_ls').replace('ti','0')))
                    pts = mls.cut_mesh().pts()
                    ls1_interface = compute_interpolate_on(mfls, ls1.values(0), pts)
                    pts = compute_points(ls1_interface, pts, ins.topography, dx, dy)
                    err_ls1[0] = np.sqrt(np.sum(
                        (eval('-' + ins.true_ls1.replace('X', 'pts[0,:]').replace('Y', '0').replace('ti','0')) - pts[1, :]) ** 2)) / \
                                                pts.shape[1]
                mfmat.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_rho_' + numstr + '.vtk', rho)
            if ins.temp:
                T = md.variable('t')
                mft.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_T_' + numstr + '.vtk', t_init)
                if ins.true_t:
                    mft.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_Ttrue_' + numstr + '.vtk',
                                      eval(ins.true_t.replace('ti','0')))
                    err_t[0] = compute_L2_dist(mft, t_init, mim, mft,
                                                              eval(ins.true_t.replace('ti','0'))) / compute_L2_norm(mft, eval(
                        ins.true_t.replace('ti','0')), mim)
                mfmat.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_mu_' + numstr + '.vtk', mu)
                if ins.solidification:
                    mfu.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_d_' + numstr + '.vtk', d_init)
                    if ins.true_dx:
                        true_d = eval(ins.true_dx.replace('x', 'x_u').replace('y', 'y_u'))
                        true_d[1::2] = eval(ins.true_dy.replace('x', 'x_u').replace('y', 'y_u'))[1::2]
                        mfu.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_dtrue_' + numstr + '.vtk',
                                          true_d)
                        err_d[0] = compute_L2_dist(mfu, d_init, mim, mfu, true_d) / compute_L2_norm(mfu,
                                                                                                              true_d,
                                                                                                              mim)
                    mfmat.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_solid_' + numstr + '.vtk', solid)
                    mft.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_Ls2_' + numstr + '.vtk',
                                      (t_init - Tg) / Tg)


        if ins.free_surface:
            md.set_variable('psi',ls1.values(0))

        hf = h5py.File(ins.outfile + '/' + ins.outfile.split('/')[-1] + '.h5', 'w')
        hf.create_dataset('last_u', data=u_init)
        hf.create_dataset('last2_u', data=u_init)
        hf.create_dataset('err_u', data=err_u)
        hf.create_dataset('last_p', data=p_init)
        hf.create_dataset('last2_p', data=p_init)
        hf.create_dataset('err_p', data=err_p)
        hf.create_dataset('last_d', data=d_init)
        hf.create_dataset('last2_d', data=d_init)
        hf.create_dataset('err_d', data=err_d)
        if ins.temp:
            hf.create_dataset('last_T', data=t_init)
            hf.create_dataset('last2_T', data=t_init)
            hf.create_dataset('err_t', data=err_t)
        if ins.free_surface:
            hf.create_dataset('last_Ls1', data=ls1.values(0))
            hf.create_dataset('last2_Ls1', data=ls1.values(0))
            hf.create_dataset('err_ls1', data=err_ls1)
            hf.create_dataset('expected_area', data=[0])
        if ins.temp & ins.solidification:
            hf.create_dataset('last_Ls2', data=(T_ls - Tg) / Tg)
        if ins.topography:
            hf.create_dataset('last_Ls3', data=ls3.values(0))
        hf.create_dataset('last_ti', data=[0])
        hf.close()

    if ins.restart:
        tstart = last_ti
        t = np.arange(tstart, ins.tf+ins.dt, ins.dt)
    else:
        tstart = ins.dt
        t = np.append(np.arange(ins.dt/10,ins.dt+ins.dt/10,ins.dt/10), np.arange(tstart, ins.tf + ins.dt, ins.dt))

        Previous_u = u_init
        U = u_init
        Previous_p = p_init
        P = p_init
        Previous_d = d_init
        D = d_init
        if ins.temp:
            Previous_t = t_init
            T = t_init

        if ins.free_surface:
            pts, idx = mls.cut_mesh().pts_from_cvid(CVIDs=mls.cut_mesh().region(0)[0])
            ls1_interface = compute_interpolate_on(mfls, ls1.values(0), pts)
            if ins.topography:
                ls3_interface = compute_interpolate_on(mfls, ls3.values(0), pts)
                pts = pts[:, ((np.abs(ls1_interface) <= np.sqrt(dx ** 2 + dy ** 2) / 100) & (
                        ls3_interface >= -np.sqrt(dx ** 2 + dy ** 2) / 100)) | (
                                     (ls1_interface <= np.sqrt(dx ** 2 + dy ** 2) / 100) & (
                                     np.abs(ls3_interface) <= np.sqrt(dx ** 2 + dy ** 2) / 100))]
            else:
                pts = pts[:, ((np.abs(ls1_interface) <= np.sqrt(dx ** 2 + dy ** 2) / 100))]
            expected_area = alphashape.alphashape(pts.transpose(), 2 * np.sqrt(dx ** 2 + dy ** 2)).area
        else:
            expected_area = 1

    # Main loop
    for ti in t:
        if np.abs(ti - 2*ins.dt)<(ins.dt/100):
            md.set_variable('Previous2_u', u_init)
            md.set_variable('Previous_u', U)

            md.set_variable('Previous2_p', p_init)
            md.set_variable('Previous_p', P)

            md.set_variable('Previous2_d', d_init)
            md.set_variable('Previous_d', D)

            if ins.temp:
                md.set_variable('Previous2_t', t_init)
                md.set_variable('Previous_t', T)

            if ins.free_surface:
                md.set_variable('Previous2_psi', psi_init)
                md.set_variable('Previous_psi', ls1.values(0))

            md.set_variable('dt', [ins.dt])

        else:
            md.set_variable('Previous2_u', Previous_u)
            md.set_variable('Previous_u', U)

            md.set_variable('Previous2_p', Previous_p)
            md.set_variable('Previous_p', P)

            md.set_variable('Previous2_d', Previous_d)
            md.set_variable('Previous_d', D)

            if ins.temp:
                md.set_variable('Previous2_t', Previous_t)
                md.set_variable('Previous_t', T)

            if ins.free_surface:
                md.set_variable('Previous2_psi', md.variable('Previous_psi'))
                md.set_variable('Previous_psi', ls1.values(0))
                md.set_variable('Previous_fext',fext)

            md.set_variable('Previous_rho',rho)
            md.set_variable('Previous_beta', beta)
            if ins.temp:
                md.set_variable('Previous_kappa', kappa)

            md.set_variable('Previous_mu', mu)
            md.set_variable('Previous_lambda', lam)
            md.set_variable('Previous_solid', solid)

        md.enable_variable('u')
        md.enable_variable('p')
        if ins.temp:
            md.enable_variable('t')

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


            # update viscosity field
            T_mat = compute_interpolate_on(mft,T,mfmat)
            ls2_mat = (T_mat - Tg) / Tg
            eta = eval(ins.eta_exp.replace('exp', 'np.exp').replace('T', 'T_mat').replace('vft','ins.vft').replace('etar','ins.etar'))
            eta[eta > ins.max_eta] = ins.max_eta
            eta[T_mat <= ins.vftb/(np.log(ins.max_eta)-ins.vfta) - 273 + ins.vftc] = ins.max_eta
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
        if ins.temp:
            if ins.free_surface | ins.topography:
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

        # update BDF coefficients
        md.set_variable('BDF0', [3 / 2])
        md.set_variable('BDF1', [-2])
        md.set_variable('BDF2', [1 / 2])
        md.set_variable('BDFf', [0])
        if ti < 2*ins.dt:
            md.set_variable('dt',ins.dt/10)

        # Solve
        md.solve('max_res', 1E-10, 'max_iter', 10, 'noisy')

        Previous_u = U
        U = ones_u * 0
        U[eval(ind_u)] = md.variable('u')

        Previous_p = P
        P =  ones_p * ins.p_atm
        P[eval(ind_p)] = md.variable('p')

        if ins.temp:
            Previous_t = T
            T = ones_t * ins.T_atm
            T[eval(ind_t)] = md.variable('t')

        md.disable_variable('u')
        md.disable_variable('p')
        if ins.temp:
            md.disable_variable('t')

        if ins.temp:
            if ins.solidification:

                md.enable_variable('d')
                md.solve('max_res', 1E-10, 'max_iter', 10, 'noisy')

                Previous_d_temp = D
                D = ones_u * 0
                D[eval(ind_u)] = md.variable('d')
                D[sciinterp.griddata(D_mat.transpose(), solid, D_u.transpose())<(1-np.sqrt(dx**2 + dy**2)/4)] = 0


                if (np.min(ls2.values(0)) < 0) & (np.max(ls2.values(0)) > 0):
                    Ls2_ext = sciinterp.griddata(D_ls.transpose(), ls2.values(0),
                                                 np.array([x_grid.flatten(), y_grid.flatten()]).transpose(),
                                                 method='nearest')
                    d_ls2 = skfmm.distance(Ls2_ext.reshape(x_grid.shape), dx=[dy, dx])
                else:
                    d_ls2 = np.ones(x_grid.shape) * (-2 * np.max([ins.L_x, ins.L_y]))

                if ins.free_surface:
                    d = np.maximum(np.array(d_ls1), np.array(d_ls2))
                else:
                    d = d_ls2

                if (np.min(ls2.values(0)) < 0) & (np.max(ls2.values(0)) > 0):
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

                else:
                    D_ext_x = 0*ones_u

                md.disable_variable('d')

            else:
                Previous_d = d_init
                D = d_init

        if ((round(ti/ins.dt) % ins.noutput == 0) & (ti>=ins.dt)) or (np.abs(ti-ins.tf)<ins.dt):
            print('Time = %g' % ti)
            numstr = str(round(ti * 10 ** ndecimal)).split('.')[0].zfill(ndigits)
            print(numstr)
            # print('Average temperature %g' % np.mean(T))
            if ins.vtk:
                mfu.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_u_' + numstr + '.vtk', U)
                if ins.true_ux:
                    true_u = eval(ins.true_ux.replace('x', 'x_u').replace('y', 'y_u'))
                    true_u[1::2] = eval(ins.true_uy.replace('x', 'x_u').replace('y', 'y_u'))[1::2]
                    mfu.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_utrue_' + numstr + '.vtk',
                                      true_u)
                    err_u[int(ti / ins.dt)] = compute_L2_dist(mfu, U, mim, mfu, true_u) / compute_L2_norm(mfu,
                        true_u, mim)
                mfp.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_P_' + numstr + '.vtk', P)
                if ins.true_p:
                    mfp.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_Ptrue_' + numstr + '.vtk',
                                       eval(ins.true_p.replace('x','x_p').replace('y','y_p')))
                    err_p[int(ti / ins.dt)] = compute_L2_dist(mfp, P, mim, mfp,
                                                              eval(ins.true_p.replace('x','x_p').replace('y','y_p'))) / compute_L2_norm(mfp,
                                                              eval(ins.true_p.replace('x','x_p').replace('y','y_p')),mim)
                if ins.free_surface:
                    mfls.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_Ls1_' + numstr + '.vtk',
                                       md.variable('Previous_psi'))
                    if ins.true_ls1:
                        mfls.export_to_vtk(
                            outfile + '/' + ins.outfile.split('/')[-1] + '_Ls1true_' + numstr + '.vtk',
                            eval(ins.true_ls1.replace('X', 'x_ls').replace('Y', 'y_ls')))
                        pts = mls.cut_mesh().pts()
                        ls1_interface = compute_interpolate_on(mfls, ls1.values(0), pts)
                        pts = compute_points(ls1_interface, pts, ins.topography, dx, dy)
                        err_ls1[int(ti / ins.dt)] = np.sqrt(np.sum(
                            (eval('-'+ins.true_ls1.replace('X', 'pts[0,:]').replace('Y', '0')) - pts[1, :]) ** 2)) / \
                                                    pts.shape[1]
                        #print(pts[1,:])
                        #print(eval('-'+ins.true_ls1.replace('X', 'pts[0,:]').replace('Y', '0')))
                        # err_ls1[int(ti/ins.dt)] = compute_L2_dist(mfls, md.variable('Previous_psi'), mim_all, mfls,
                        #                                          eval(ins.true_ls1))/compute_L2_norm(mfls,eval(ins.true_ls1),mim_all)
                    mfmat.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_rho_' + numstr + '.vtk', rho)
                if ins.temp:
                    T = md.variable('t')
                    mft.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_T_' + numstr + '.vtk', T)
                    if ins.true_t:
                        mft.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_Ttrue_' + numstr + '.vtk',
                                          eval(ins.true_t))
                        err_t[int(ti / ins.dt)] = compute_L2_dist(mft, T, mim, mft,
                                                                  eval(ins.true_t)) / compute_L2_norm(mft, eval(
                            ins.true_t), mim)
                    mfmat.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_mu_' + numstr + '.vtk', mu)
                    if ins.solidification:
                        mfu.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_d_' + numstr + '.vtk', D)
                        if ins.true_dx:
                            true_d = eval(ins.true_dx.replace('x', 'x_u').replace('y', 'y_u'))
                            true_d[1::2] = eval(ins.true_dy.replace('x', 'x_u').replace('y', 'y_u'))[1::2]
                            mfu.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_dtrue_' + numstr + '.vtk',
                                              true_d)
                            err_d[int(ti / ins.dt)] = compute_L2_dist(mfu, D, mim, mfu, true_d) / compute_L2_norm(mfu,
                                                                                                                  true_d,
                                                                                                                  mim)
                        mfmat.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_solid_' + numstr + '.vtk',solid)
                        mft.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_Ls2_' + numstr + '.vtk',
                                          (T - Tg) / Tg)

            hf = h5py.File(ins.outfile + '/' + ins.outfile.split('/')[-1] + '.h5', 'a')
            hf['last_u'][:] = U
            hf['last2_u'][:] = Previous_u
            hf['err_u'][:] = err_u
            hf['last_p'][:] = P
            hf['last2_p'][:] = Previous_p
            hf['err_p'][:] = err_p
            hf['last_d'][:] = D
            hf['last2_d'][:] = Previous_d
            hf['err_d'][:] = err_d
            if ins.temp:
                hf['last_T'][:] = T
                hf['last2_T'][:] = Previous_t
                hf['err_t'][:] = err_t
            if ins.free_surface:
                hf['last_Ls1'][:] = ls1.values(0)
                hf['last2_Ls1'][:] = md.variable('Previous_psi')
                hf['err_ls1'][:] = err_ls1
                hf['expected_area'][:] = [expected_area]
            if ins.temp & ins.solidification:
                hf['last_Ls2'][:] = (T_ls - Tg) / Tg
            if ins.topography:
                hf['last_Ls3'][:] = ls3.values(0)
            hf['last_ti'][:] = [ti]
            hf.close()

        if ins.free_surface:
            md.enable_variable('psis')
            md.solve('max_res', 1E-10, 'max_iter', 10, 'noisy')
            md.disable_variable('psis')

            # construct extension velocities
            # speed on the interface is U.Grad(Psi)

            Psi_grid = sciinterp.griddata(D_ls.transpose(), md.variable('psis'),
                                          np.array([x_grid.flatten(), y_grid.flatten()]).transpose(),
                                          method='nearest').reshape(x_grid.shape)

            dx_Psi_grid, dy_Psi_grid, curvature, mag_grad_Psi_grid = compute_curvature(Psi_grid, dx, dy)
            if ins.topography:
                Ls3_grid = sciinterp.griddata(D_ls.transpose(), ls3.values(0),
                                       np.array([x_grid.flatten(), y_grid.flatten()]).transpose(), method='linear').reshape(x_grid.shape)
            else:
                    Ls3_grid = np.ones_like(x_grid) + np.sqrt(dx ** 2 + dy ** 2)
            eta_grid = sciinterp.griddata(D_mat.transpose(), eta,
                                          np.array([x_grid.flatten(), y_grid.flatten()]).transpose(),
                                          method='nearest').reshape(x_grid.shape)
            mean_curvature = compute_mean_curvature(Psi_grid, Ls3_grid, curvature, eta_grid, dx, dy)

            ux_grid = sciinterp.griddata(D_u[:,::2].transpose(), U[::2],
                                         np.array([x_grid.flatten(), y_grid.flatten()]).transpose(),
                                         method='nearest')
            uy_grid = sciinterp.griddata(D_u[:,1::2].transpose(), U[1::2],
                                         np.array([x_grid.flatten(), y_grid.flatten()]).transpose(),
                                         method='nearest')

            #F_grid = ux_grid * (dx_Psi_grid/mag_grad_Psi_grid).flatten() + uy_grid * (dy_Psi_grid/mag_grad_Psi_grid).flatten()

            relax_speed = ins.epsilon_psi/eta_grid * (curvature - mean_curvature)

            relax_max = np.sqrt(dx ** 2 + dy ** 2) / (2 * ins.dt)
            relax_speed[relax_speed > relax_max] = relax_max
            relax_speed[relax_speed < -relax_max] = -relax_max

            relax_speed[Ls3_grid<np.sqrt(dx**2+dy**2)/2] = 0

            mask_grid = (Psi_grid > 0)
            d_ls1, Fx_ext_grid = skfmm.extension_velocities(sciinterp.griddata(D_ls.transpose(), md.variable('psi'),
                                                                              np.array([x_grid.flatten(),
                                                                                        y_grid.flatten()]).transpose(),
                                                                              method='nearest').reshape(x_grid.shape),
                                                           ux_grid.reshape(x_grid.shape) - relax_speed*dx_Psi_grid/mag_grad_Psi_grid,
                                                           dx=[dy, dx], ext_mask=mask_grid)
            d_ls1, Fy_ext_grid = skfmm.extension_velocities(sciinterp.griddata(D_ls.transpose(), md.variable('psi'),
                                                                               np.array([x_grid.flatten(),
                                                                                         y_grid.flatten()]).transpose(),
                                                                               method='nearest').reshape(x_grid.shape),
                                                            uy_grid.reshape(
                                                                x_grid.shape) - relax_speed * dy_Psi_grid / mag_grad_Psi_grid,
                                                            dx=[dy, dx], ext_mask=mask_grid)

            Fx_ext = sciinterp.griddata(np.array([x_grid.flatten(), y_grid.flatten()]).transpose(), Fx_ext_grid.flatten(),
                                       D_ls.transpose(),
                                       method='nearest').flatten()
            Fy_ext = sciinterp.griddata(np.array([x_grid.flatten(), y_grid.flatten()]).transpose(),
                                        Fy_ext_grid.flatten(),
                                        D_ls.transpose(),
                                        method='nearest').flatten()
            #dx_Psi = sciinterp.griddata(np.array([x_grid.flatten(), y_grid.flatten()]).transpose(), dx_Psi_grid.flatten(),
            #                            D_ls.transpose(),
            #                            method='nearest').flatten()
            #dy_Psi = sciinterp.griddata(np.array([x_grid.flatten(), y_grid.flatten()]).transpose(), dy_Psi_grid.flatten(),
            #                            D_ls.transpose(),
            #                            method='nearest').flatten()
            #mag_grad_Psi = sciinterp.griddata(np.array([x_grid.flatten(), y_grid.flatten()]).transpose(),
            #                                  mag_grad_Psi_grid.flatten(),
            #                                  D_ls.transpose(),
            #                                  method='nearest').flatten()

            fext = 0 * ones_f
            fext[::2] = Fx_ext
            fext[1::2] = Fy_ext

            md.set_variable('fext', fext)
            if ti<=ins.dt/10:
                fext_init = fext.copy()
                md.set_variable('Previous_fext',fext)

            if ((int(ti/ins.dt) % ins.noutput == 0) & (ti>=ins.dt)) or (np.abs(ti - ins.tf) < ins.dt):
                numstr = str(ti * 10 ** ndecimal).split('.')[0].zfill(ndigits)
                # print('Average temperature %g' % np.mean(T))
                if ins.vtk:
                    if ins.free_surface:
                        mff.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_Fext_' + numstr + '.vtk', fext)
                        mfls.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_psis_' + numstr + '.vtk', md.variable('psis'))
                        mfls.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_curvature_' + numstr + '.vtk',
                                           sciinterp.griddata(
                                               np.array([x_grid.flatten(), y_grid.flatten()]).transpose(),
                                               curvature.flatten(),
                                               D_ls.transpose(),
                                               method='nearest'))

            md.enable_variable('psi')
            md.solve('max_res', 1E-10, 'max_iter', 10, 'noisy')
            md.disable_variable('psi')

            # mass conservation correction
            ls1.set_values(md.variable('psi'))
            itr = 0
            area = 0
            if ins.influx:
                d_area = ins.dt * ((ins.influx_right - ins.influx_left) * eval(ins.influx_uy) + (
                            ins.influx_top - ins.influx_bottom) * eval(ins.influx_ux))
                expected_area += d_area
            while (np.abs(area - expected_area) / expected_area > 1e-4) & (itr < 10) & (np.min(np.abs(ls1.values(0)[edges_ls]))>=np.sqrt(dx**2 + dy**2)/2):
                if itr > 0:
                    if ins.topography:
                        ls1.set_values(ls1.values(0) + (area - expected_area) / pts.shape[1] / np.sqrt(dx ** 2 + dy ** 2)*(ls3.values(0)>np.sqrt(dx**2 + dy**2)/2)/2)
                    else:
                        ls1.set_values(ls1.values(0) + (area - expected_area) / pts.shape[1] / np.sqrt(dx ** 2 + dy ** 2) / 2)
                itr += 1
                mls.adapt()
                pts, idx = mls.cut_mesh().pts_from_cvid(CVIDs=mls.cut_mesh().region(0)[0])
                ls1_interface = compute_interpolate_on(mfls, ls1.values(0), pts)
                if ins.topography:
                    ls3_interface = compute_interpolate_on(mfls, ls3.values(0), pts)
                    pts = pts[:, ((np.abs(ls1_interface) <= np.sqrt(dx ** 2 + dy ** 2) / 100) & (
                        ls3_interface >= -np.sqrt(dx ** 2 + dy ** 2) / 100)) | (
                                     (ls1_interface <= np.sqrt(dx ** 2 + dy ** 2) / 100) & (
                                     np.abs(ls3_interface) <= np.sqrt(dx ** 2 + dy ** 2) / 100))]
                else:
                    pts = pts[:, ((np.abs(ls1_interface) <= np.sqrt(dx ** 2 + dy ** 2) / 100))]
                area = alphashape.alphashape(pts.transpose(), 2 * np.sqrt(dx ** 2 + dy ** 2)).area
                print((area - expected_area) / expected_area)

            mls.adapt()
            mim.adapt()
            mim_surf.adapt()
            mim_all.adapt()
            mim_integ.adapt()
            mfp_cut.set_partial(eval(ind_p))
            mfu_cut.set_partial(eval(ind_u))
            if ins.temp:
                mft_cut.set_partial(eval(ind_t))
                if ins.t_enrich & ins.free_surface:
                    enriched_dofs = np.nonzero(
                        np.abs(compute_interpolate_on(mfls, ls1.values(0), DOFpts_t)) < (np.sqrt(dx ** 2 + dy ** 2)))[
                                        0] + 1
                    enriched_dofs = enriched_dofs[enriched_dofs > 1]

                    mft_cutoff.set_enriched_dofs(enriched_dofs)


            enriched_dofs = np.nonzero(
                np.abs(compute_interpolate_on(mfls, ls1.values(0), DOFpts_p)) < (np.sqrt(dx ** 2 + dy ** 2)))[0] + 1
            enriched_dofs = enriched_dofs[enriched_dofs > 1]

            mfp_cutoff.set_enriched_dofs(enriched_dofs)

            if ins.temp & ins.solidification:
                Previous_Ls_u = compute_interpolate_on(mfls, md.variable('Previous_psi'), mfu)
                Ls1_u = compute_interpolate_on(mfls, ls1.values(0), mfu)
                D[(Previous_Ls_u >= 0) & (Ls1_u <= 0)] = D_ext[(Previous_Ls_u >= 0) & (Ls1_u <= 0)]


    # Python visualization
    if ins.free_surface | ins.topography:
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
    U = md.variable('u')
    P = md.variable('p')
    if ins.temp:
        T = md.variable('t')
    else:
        T = None
    # %% md
    # Visualize results
    # %%
    if ins.plots:
        fig2, ax = plt.subplots(figsize=(16, 16), ncols=2, nrows=2)

        c1 = ax[0, 0].tripcolor(x_u[eval(ind_u)][::2], y_u[eval(ind_u)][::2], U[::2],
                                cmap='RdBu_r', shading='gouraud',
                                vmin=-np.max(np.abs(U[::2])), vmax=np.max(np.abs(U[::2])))
        c2 = ax[0, 1].tripcolor(x_u[eval(ind_u)][1::2], y_u[eval(ind_u)][1::2], U[1::2],
                                cmap='RdBu_r', shading='gouraud',
                                vmin=-np.max(np.abs(U[1::2])), vmax=np.max(np.abs(U[1::2])))
        c3 = ax[1, 0].tripcolor(x_p[eval(ind_p)], y_p[eval(ind_p)], P,
                                cmap='RdBu_r',
                                shading='gouraud')  # ,vmin=-1000,vmax=1000)#,vmin=0.999*np.mean(P),vmax=1.001*np.mean(P))
        if ins.temp:
            c4 = ax[1, 1].tripcolor(x_t[eval(ind_t)], y_t[eval(ind_t)], T,
                                    cmap='RdBu_r', shading='gouraud', vmin=0, vmax=ins.T0)
            ax[1, 1].set_title('T', fontsize=18)
            plt.colorbar(c4, ax=ax[1, 1], shrink=0.5, label='Temperature ($^\circ$C)')

        elif ins.free_surface:
            c4 = ax[1, 1].tripcolor(x_ls, y_ls, ls1.values(0), cmap='RdBu_r', shading='gouraud', vmin=-1, vmax=1)
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
                axi[0].tricontour(x_ls, y_ls, ls2.values(0), levels=[0], colors='k',
                                  linestyles='--')
        ax[0, 0].set_title('U', fontsize=18)
        ax[0, 1].set_title('V', fontsize=18)
        ax[1, 0].set_title('P', fontsize=18)

        fig2.savefig(outfile + '/' + ins.outfile.split('/')[-1] + '_solution.pdf')
        plt.close(fig2)