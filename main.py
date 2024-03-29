import sys
import os
import json
from types import SimpleNamespace
import h5py

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate as sciinterp
from skimage.morphology import remove_small_objects
import skfmm
import alphashape
import getfem as gf
from getfem import *
from curvature import *
from math import erf
from shapely.plotting import plot_polygon

outfiles = [outfile for outfile in sys.argv[1:]]
print(outfiles)

for outfilei in outfiles:
    print(outfilei)
    outfile = './Results/' + outfilei

    # import simulation parameters
    with open(outfile + '/' + outfile.split('/')[-1] + '.json', 'r') as openfile:
        dictionary = json.load(openfile)

    ins = SimpleNamespace(**dictionary)

    dx = ins.L_x/(ins.nx*(ins.ls_k))
    dy = ins.L_y/(ins.ny*(ins.ls_k))
    xs = np.linspace(0, ins.L_x, ins.nx*(ins.u_k)+1)
    ys = np.linspace(0, ins.L_y, ins.ny*(ins.u_k)+1)
    x_grid, y_grid = np.meshgrid(xs, ys)

    if (type(ins.Tg) is float) or (type(ins.Tg) is int):
        Tg = ins.Tg
    elif type(ins.Tg) is type(str):
        Tg = eval(ins.Tg.replace('vft', 'ins.vft').replace('etar','ins.etar'))
    elif type(ins.Tg) is type(None):
        Tg = ins.vftc + ins.vftb/(np.log(10**12)-ins.vfta) - 273

    if type(ins.ndigits) == int:
        ndigits = ins.ndigits
        ndecimal = len(str(ins.dt)[:ndigits+1].split('.')[-1].rstrip('0'))
    else:
        ndigits = int(np.floor(np.log10(ins.tf/ins.dt))) + len(str(ins.dt).split('.')[-1].lstrip('0')
                                                           ) + len(str(ins.dt).split('.')[0].rstrip('0'))
        if '.' in str(ins.dt)[:ndigits+1]:
            ndigits += 1
            ndecimal = len(str(ins.dt)[:ndigits+1].split('.')[-1].rstrip('0'))
        else:
            ndecimal = 0

    for ls in ['ls1p', 'ls1s', 'ls2p', 'ls2s', 'ls3p', 'ls3s']:
        if type(ins.__dict__[ls]) is str:
            ins.__dict__[ls] = ins.__dict__[ls].replace('X','x').replace('Y','y')

    if ins.compressible:
        ins.steady = False

    ti = 0
    if ins.restart:
        try:
            hf = h5py.File(ins.outfile + '/' + ins.outfile.split('/')[-1] + '.h5', 'r+')
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

            if ins.temperature:
                T = hf.get('last_t')[:]
                Previous_t = hf.get('last2_T')[:]
                err_t_i = hf.get('err_t')[:]
                err_t = np.zeros(int(np.ceil(ins.tf / ins.dt)))
                err_t[:len(err_t_i)] = err_t_i
            if ins.free_surface:
                last_Ls1 = hf.get('last_ls1')[:]
                last2_Ls1 = hf.get('last2_ls1')[:]
                err_ls1_i = hf.get('err_ls1')[:]
                err_ls1 = np.zeros(int(np.ceil(ins.tf / ins.dt)))
                err_ls1[:len(err_ls1_i)] = err_ls1_i
                expected_area = hf.get('expected_area')[:]
            last_ti = hf.get('last_ti')[:]
            ti = last_ti
            print(last_ti)
            hf.close()
        except Exception as e:
            print(e)
            ins.restart = False
            hf.close()

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
        bounds = ['left', 'right', 'top', 'bottom']

        fb_left = mesh.outer_faces_with_direction([-1., 0.], 0.01)  # boundary 1
        fb_right = mesh.outer_faces_with_direction([1., 0.], 0.01)  # boundary 2
        fb_top = mesh.outer_faces_with_direction([0., 1.], 0.01)  # boundary 3
        fb_bottom = mesh.outer_faces_with_direction([0., -1.], 0.01)  # boundary 4

        if ins.influx:
            fb_influx = mesh.outer_faces_in_box([ins.influx_left, ins.influx_bottom],
                                                [ins.influx_right, ins.influx_top])

            fb_left = fb_left[:,~(np.in1d(fb_left[0], fb_influx[0]) & np.in1d(fb_left[1], fb_influx[1]))]
            fb_right = fb_right[:,~(np.in1d(fb_right[0], fb_influx[0]) & np.in1d(fb_right[1], fb_influx[1]))]
            fb_top = fb_top[:,~(np.in1d(fb_top[0], fb_influx[0]) & np.in1d(fb_top[1], fb_influx[1]))]
            fb_bottom = fb_bottom[:,~(np.in1d(fb_bottom[0], fb_influx[0]) & np.in1d(fb_bottom[1], fb_influx[1]))]

            bounds = bounds + ['influx']

    bounds_type = []
    for i, bound in enumerate(bounds):
        mesh.set_region(i + 1, eval('fb_' + bound))
        if ins.temperature:
            bounds_type = np.append(bounds_type, type(eval('ins.' + bound + '_t')))
        else:
            bounds_type = np.append(bounds_type, type(eval('ins.' + bound + '_ux')))

    if ins.vtk:
        # export to vtk for visualization through python
        mesh.export_to_vtk(ins.outfile + '/' + ins.outfile.split('/')[-1] + '_mesh.vtk')

    # Levelset definition:
    # free surface
    if ins.free_surface:
        ls1 = gf.LevelSet(mesh, ins.ls_k, ins.ls1p, ins.ls1s)
        ls1_previous = gf.LevelSet(mesh, ins.ls_k, ins.ls1p, ins.ls1s)
        ls1_previous2 = gf.LevelSet(mesh, ins.ls_k, ins.ls1p, ins.ls1s)
        ls1_init = gf.LevelSet(mesh, ins.ls_k, ins.ls1p, ins.ls1s)
        mls = gf.MeshLevelSet(mesh)
        mls.add(ls1)

        mlsxfem = gf.MeshLevelSet(mesh)
        mlsxfem_previous = gf.MeshLevelSet(mesh)
        mlsxfem_previous2 = gf.MeshLevelSet(mesh)
        mlsxfem_init = gf.MeshLevelSet(mesh)
        #if ins.topography:
        #    ls1xfem = gf.LevelSet(mesh, ins.ls_k, ins.ls1p, '-(' + ins.ls3p + ')')
        #    ls1xfem_previous = gf.LevelSet(mesh, ins.ls_k, ins.ls1p, '-(' + ins.ls3p + ')')
        #    ls1xfem_previous2 = gf.LevelSet(mesh, ins.ls_k, ins.ls1p, '-(' + ins.ls3p + ')')
        #    ls1xfem_init = gf.LevelSet(mesh, ins.ls_k, ins.ls1p, '-(' + ins.ls3p + ')')
        #    mlsxfem.add(ls1xfem)
        #    mlsxfem_previous.add(ls1xfem)
        #    mlsxfem_previous2.add(ls1xfem)
        #    mlsxfem_init.add(ls1xfem)
        #else:
        mlsxfem.add(ls1)
        mlsxfem_previous.add(ls1_previous)
        mlsxfem_previous2.add(ls1_previous2)
        mlsxfem_init.add(ls1_init)

    # temperature contour
    if ins.temperature & ins.solidification:
        mls2 = gf.MeshLevelSet(mesh)
        ls2 = gf.LevelSet(mesh, ins.t_k, ins.ls2p, ins.ls2s)
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
            ls1_previous.set_values(last2_Ls1)
            if ins.topography:
                ls1xfem.set_values(last_Ls1,'-(' + ins.ls3p + ')')

    if ins.free_surface | ins.topography:
        mls.adapt()
        mlsxfem.adapt()
        mlsxfem_previous.adapt()
        mlsxfem_previous2.adapt()
        mlsxfem_init.adapt()

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

                if ins.temperature:
                    bound_type = type(eval('ins.' + bound + '_t'))
                else:
                    bound_type = type(eval('ins.' + bound + '_ux'))
                if bound_type is not type(None):
                    color = 'r'  # plot Dirichlet bounds on temperature in red
                    if ins.temperature:
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
            mim_fluidbound = gf.MeshIm('levelset', mlsxfem, 'boundary(a)', gf.Integ('IM_TRIANGLE(5)'))

        if ins.topography:
            mim_base = gf.MeshIm('levelset', mlsxfem, 'boundary(b)', gf.Integ('IM_TRIANGLE(5)'))
            mim_topo = gf.MeshIm('levelset', mls, 'boundary(b-a)', gf.Integ('IM_TRIANGLE(5)'))

        mim_integ = gf.MeshIm('levelset', mls, 'boundary(a-b)', gf.Integ('IM_TRIANGLE(5)'))

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

    # pressure
    mfp0 = gf.MeshFem(mesh, 1)
    mfp0.set_classical_fem(ins.p_k)

    mfmat0 = gf.MeshFem(mesh, 1)
    mfmat0.set_classical_fem(ins.mat_k)

    mfmat_cutoff0 = gf.MeshFem(mesh, 1)
    mfmat_cutoff0.set_classical_fem(1)

    if ins.free_surface:
        mfp_cutoff0 = gf.MeshFem(mesh, 1)
        mfp_cutoff0.set_classical_fem(1)
        mf_cutoff = gf.GlobalFunction('cutoff',1,np.sqrt(dx**2 + dy**2),np.sqrt(dx**2 + dy**2)/100,np.sqrt(dx**2 + dy**2))
        mf_cutoff_ls1 = gf.MeshFem('global function',mesh,ls1,[mf_cutoff],2)
        mf_cutoff_ls1_previous = gf.MeshFem('global function', mesh, ls1_previous, [mf_cutoff], 2)
        mf_cutoff_ls1_previous2 = gf.MeshFem('global function', mesh, ls1_previous2, [mf_cutoff], 2)
        mf_cutoff_ls1_init = gf.MeshFem('global function', mesh, ls1_init, [mf_cutoff], 2)
        if ins.topography:
            mf_cutoff_ls3 = gf.MeshFem('global function', mesh, ls3, [mf_cutoff], 2)
            mfp_cutoff3 = gf.MeshFem('product', mf_cutoff_ls3, mfp_cutoff0)
            mfp_cutoff = gf.MeshFem('product', mf_cutoff_ls1, mfp_cutoff3)
            mfp_cutoff_previous = gf.MeshFem('product', mf_cutoff_ls1_previous, mfp_cutoff3)
            mfp_cutoff_previous2 = gf.MeshFem('product', mf_cutoff_ls1_previous2, mfp_cutoff3)
            mfp_cutoff_init = gf.MeshFem('product', mf_cutoff_ls1_init, mfp_cutoff3)
        else:
            mfp_cutoff = gf.MeshFem('product', mf_cutoff_ls1, mfp_cutoff0)
            mfp_cutoff_previous = gf.MeshFem('product', mf_cutoff_ls1_previous, mfp_cutoff0)
            mfp_cutoff_previous2 = gf.MeshFem('product', mf_cutoff_ls1_previous2, mfp_cutoff0)
            mfp_cutoff_init = gf.MeshFem('product', mf_cutoff_ls1_init, mfp_cutoff0)

        mfp = gf.MeshFem('levelset', mlsxfem, mfp0)
        mfp_previous = gf.MeshFem('levelset', mlsxfem_previous, mfp0)
        mfp_previous2 = gf.MeshFem('levelset', mlsxfem_previous2, mfp0)
        mfp_init = gf.MeshFem('levelset', mlsxfem_init, mfp0)


        #mfp = gf.MeshFem('sum', mfp_cutoff, mfp0)
        #mfp_previous = gf.MeshFem('sum', mfp_cutoff_previous, mfp0)
        #mfp_previous2 = gf.MeshFem('sum', mfp_cutoff_previous2, mfp0)
        #mfp_init = gf.MeshFem('sum', mfp_cutoff_init, mfp0)

        DOFpts_p = mfp.basic_dof_nodes()
        enriched_dofs1 = (np.nonzero(
            np.abs(compute_interpolate_on(mfls, ls1.values(0), DOFpts_p)) < (np.sqrt(dx ** 2 + dy ** 2)))[0] + 1)
        if ins.topography:
            enriched_dofs3 = (np.nonzero(
                np.abs(compute_interpolate_on(mfls, ls3.values(0), DOFpts_p)) < (np.sqrt(dx ** 2 + dy ** 2)))[0] + 1)
            enriched_dofsp = np.append(enriched_dofs1,enriched_dofs3)
        else:
            enriched_dofsp = enriched_dofs1
        enriched_dofsp = enriched_dofsp[enriched_dofsp > 1]
        mfp_cutoff.set_enriched_dofs(enriched_dofsp)
        mfp_cutoff_previous.set_enriched_dofs(enriched_dofsp)
        mfp_cutoff_previous2.set_enriched_dofs(enriched_dofsp)
        mfp_cutoff_init.set_enriched_dofs(enriched_dofsp)
        enriched_dofsp_previous = enriched_dofsp

        DOFpts_mat = mfmat0.basic_dof_nodes()
        if ins.topography:
            mfmat_cutoff3 = gf.MeshFem('product', mf_cutoff_ls3, mfmat_cutoff0)
            mfmat_cutoff = gf.MeshFem('product', mf_cutoff_ls1, mfmat_cutoff0)
            mfmat_cutoff_previous = gf.MeshFem('product', mf_cutoff_ls1_previous, mfmat_cutoff0)
            mfmat_cutoff_previous2 = gf.MeshFem('product', mf_cutoff_ls1_previous2, mfmat_cutoff0)
            mfmat_cutoff_init = gf.MeshFem('product', mf_cutoff_ls1_init, mfmat_cutoff0)
        else:
            mfmat_cutoff = gf.MeshFem('product', mf_cutoff_ls1, mfmat_cutoff0)
            mfmat_cutoff_previous = gf.MeshFem('product', mf_cutoff_ls1_previous, mfmat_cutoff0)
            mfmat_cutoff_previous2 = gf.MeshFem('product', mf_cutoff_ls1_previous2, mfmat_cutoff0)
            mfmat_cutoff_init = gf.MeshFem('product', mf_cutoff_ls1_init, mfmat_cutoff0)
        enriched_dofs1 = (np.nonzero(
            np.abs(compute_interpolate_on(mfls, ls1.values(0), DOFpts_mat)) < (np.sqrt(dx ** 2 + dy ** 2)))[0] + 1)
        if ins.topography:
            enriched_dofs3 = (np.nonzero(
                np.abs(compute_interpolate_on(mfls, ls3.values(0), DOFpts_mat)) < (np.sqrt(dx ** 2 + dy ** 2)))[0] + 1)
            enriched_dofsmat = np.append(enriched_dofs1, enriched_dofs3)
        else:
            enriched_dofsmat = enriched_dofs1
        enriched_dofsmat = enriched_dofsmat[enriched_dofsmat > 1]
        enriched_dofsmat_previous = enriched_dofsmat
        mfmat_cutoff.set_enriched_dofs(enriched_dofsmat)
        mfmat_cutoff_previous.set_enriched_dofs(enriched_dofsmat)
        mfmat_cutoff_previous2.set_enriched_dofs(enriched_dofsmat)
        mfmat_cutoff_init.set_enriched_dofs(enriched_dofsmat)
        #mfmat = gf.MeshFem('sum', mfmat_cutoff, mfmat0)
        mfmat = gf.MeshFem('levelset', mlsxfem, mfmat0)
        mfmat_previous = gf.MeshFem('levelset', mlsxfem_previous, mfmat0)
        mfmat_previous2 = gf.MeshFem('levelset', mlsxfem_previous2, mfmat0)
        mfmat_init = gf.MeshFem('levelset', mlsxfem_init, mfmat0)
    else:
        mfp = gf.MeshFem(mesh, 1)  # scalar field
        mfp.set_classical_fem(ins.p_k)  # continuous piecewise linear
        mfp_init = gf.MeshFem(mesh, 1)
        mfp_init.set_classical_fem(ins.p_k)
        mfp_previous = gf.MeshFem(mesh, 1)
        mfp_previous.set_classical_fem(ins.p_k)
        mfp_previous2 = gf.MeshFem(mesh, 1)
        mfp_previous2.set_classical_fem(ins.p_k)

        mfmat = gf.MeshFem(mesh, 1)  # scalar field
        mfmat.set_classical_fem(ins.mat_k)  # continuous piecewise linear
        mfmat_init = gf.MeshFem(mesh, 1)  # scalar field
        mfmat_init.set_classical_fem(ins.mat_k)  # continuous piecewise linear
        mfmat_previous = gf.MeshFem(mesh, 1)  # scalar field
        mfmat_previous.set_classical_fem(ins.mat_k)  # continuous piecewise linear
        mfmat_previous2 = gf.MeshFem(mesh, 1)  # scalar field
        mfmat_previous2.set_classical_fem(ins.mat_k)  # continuous piecewise linear

    # velocity
    mfu = gf.MeshFem(mesh, 2)  # vector field
    mfu.set_classical_fem(ins.u_k)  # continuous piecewise quadratic
    mfcurve = gf.MeshFem(mesh, 1)  # vector field
    mfcurve.set_classical_fem(ins.u_k)  # continuous piecewise quadratic

    mfmu0 = gf.MeshFem(mesh, 1)
    mfmu0.set_classical_fem(ins.t_k)

    mft0 = gf.MeshFem(mesh, 1)
    mft0.set_classical_fem(ins.t_k)
    # temperature
    if ins.t_enrich:
        if ins.free_surface:
            mft_xfem0 = gf.MeshFem(mesh, 1)
            mft_xfem0.set_classical_fem(ins.t_k)

            mft_cutoff0 = gf.MeshFem(mesh, 1)
            mft_cutoff0.set_classical_fem(1)

            DOFpts_t = mft0.basic_dof_nodes()
            if ins.topography:
                mft_cutoff3 = gf.MeshFem('product', mf_cutoff_ls3, mft_cutoff0)
                mft_cutoff = gf.MeshFem('product', mf_cutoff_ls1, mft_cutoff0)
                mft_cutoff_previous = gf.MeshFem('product', mf_cutoff_ls1_previous, mft_cutoff0)
                mft_cutoff_previous2 = gf.MeshFem('product', mf_cutoff_ls1_previous2, mft_cutoff0)
                mft_cutoff_init = gf.MeshFem('product', mf_cutoff_ls1_init, mft_cutoff0)
            else:
                mft_cutoff = gf.MeshFem('product', mf_cutoff_ls1, mft_cutoff0)
                mft_cutoff_previous = gf.MeshFem('product', mf_cutoff_ls1_previous, mft_cutoff0)
                mft_cutoff_previous2 = gf.MeshFem('product', mf_cutoff_ls1_previous2, mft_cutoff0)
                mft_cutoff_init = gf.MeshFem('product', mf_cutoff_ls1_init, mft_cutoff0)

            enriched_dofs1 = (np.nonzero(
                np.abs(compute_interpolate_on(mfls, ls1.values(0), DOFpts_t)) < (np.sqrt(dx ** 2 + dy ** 2)))[
                                  0] + 1)
            if ins.topography:
                enriched_dofs3 = (np.nonzero(
                    np.abs(compute_interpolate_on(mfls, ls3.values(0), DOFpts_t)) < (np.sqrt(dx ** 2 + dy ** 2)))[
                                  0] + 1)
                enriched_dofst = np.append(enriched_dofs1, enriched_dofs3)
            else:
                enriched_dofst = enriched_dofs1
            enriched_dofst = enriched_dofst[enriched_dofst > 1]
            enriched_dofst_previous = enriched_dofst

            mft_cutoff.set_enriched_dofs(enriched_dofst)

            #mft = gf.MeshFem('levelset', mlsxfem, mft_xfem0)
            mft = gf.MeshFem('sum', mft_cutoff, mft0)
            mft_previous = gf.MeshFem('sum', mft_cutoff_previous, mft0)
            mft_previous2 = gf.MeshFem('sum', mft_cutoff_previous2, mft0)
            mft_init = gf.MeshFem('sum', mft_cutoff_init, mft0)

            mfmu = gf.MeshFem('levelset', mlsxfem, mfmu0)
            mfmu_previous = gf.MeshFem('levelset', mlsxfem_previous, mfmu0)
            mfmu_previous2 = gf.MeshFem('levelset', mlsxfem_previous2, mfmu0)
            mfmu_init = gf.MeshFem('levelset', mlsxfem_init, mfmu0)
        else:
            mft = gf.MeshFem(mesh, 1)  # scalar field
            mft.set_classical_fem(ins.t_k)  # continuous piecewise linear
            mft_previous = gf.MeshFem(mesh, 1)  # scalar field
            mft_previous.set_classical_fem(ins.t_k)  # continuous piecewise linear
            mft_previous2 = gf.MeshFem(mesh, 1)  # scalar field
            mft_previous2.set_classical_fem(ins.t_k)  # continuous piecewise linear
            mft_init = gf.MeshFem(mesh, 1)  # scalar field
            mft_init.set_classical_fem(ins.t_k)  # continuous piecewise linear

            mfmu = gf.MeshFem(mesh, 1)  # scalar field
            mfmu.set_classical_fem(ins.t_k)  # continuous piecewise linear
            mfmu_previous = gf.MeshFem(mesh, 1)  # scalar field
            mfmu_previous.set_classical_fem(ins.t_k)  # continuous piecewise linear
            mfmu_previous2 = gf.MeshFem(mesh, 1)  # scalar field
            mfmu_previous2.set_classical_fem(ins.t_k)  # continuous piecewise linear
            mfmu_init = gf.MeshFem(mesh, 1)  # scalar field
            mfmu_init.set_classical_fem(ins.t_k)  # continuous piecewise linear

    else:
        mft = gf.MeshFem(mesh, 1)  # scalar field
        mft.set_classical_fem(ins.t_k)  # continuous piecewise quadratic
        mft_previous = gf.MeshFem(mesh, 1)  # scalar field
        mft_previous.set_classical_fem(ins.t_k)  # continuous piecewise quadratic
        mft_previous2 = gf.MeshFem(mesh, 1)  # scalar field
        mft_previous2.set_classical_fem(ins.t_k)  # continuous piecewise quadratic
        mft_init = gf.MeshFem(mesh, 1)  # scalar field
        mft_init.set_classical_fem(ins.t_k)  # continuous piecewise quadratic

        mfmu = gf.MeshFem(mesh, 1)  # scalar field
        mfmu.set_classical_fem(ins.t_k)  # continuous piecewise quadratic
        mfmu_previous = gf.MeshFem(mesh, 1)  # scalar field
        mfmu_previous.set_classical_fem(ins.t_k)  # continuous piecewise quadratic
        mfmu_previous2 = gf.MeshFem(mesh, 1)  # scalar field
        mfmu_previous2.set_classical_fem(ins.t_k)  # continuous piecewise quadratic
        mfmu_init = gf.MeshFem(mesh, 1)  # scalar field
        mfmu_init.set_classical_fem(ins.t_k)  # continuous piecewise quadratic

    # above topography
    if (ins.free_surface & (~ins.solve_air)) | (ins.topography & (~ins.solve_topography)):
        ind_p = 'mfp.basic_dof_from_cv(np.append(mim_integ.convex_index(),mim.convex_index()))'
        ind_u = 'mfu.basic_dof_from_cv(np.append(mim_integ.convex_index(),mim.convex_index()))'
        if ins.temperature:
            ind_t = 'mft.basic_dof_from_cv(np.append(mim_integ.convex_index(),mim.convex_index()))'
    else:
        ind_p = 'mfp.basic_dof_from_cv(mim_all.convex_index())'
        ind_u = 'mfu.basic_dof_from_cv(mim_all.convex_index())'
        if ins.temperature:
            ind_t = 'mft.basic_dof_from_cv(mim_all.convex_index())'

    # get mesh coordinates
    D_u = mfu.basic_dof_nodes()
    ones_u = np.ones(D_u.shape[1])
    x_u = D_u[0, :]
    y_u = D_u[1, :]

    D_curve = mfcurve.basic_dof_nodes()
    ones_curve = np.ones(D_curve.shape[1])
    x_curve = D_curve[0, :]
    y_curve = D_curve[1, :]

    D_p = mfp.basic_dof_nodes()
    ones_p = np.ones(D_p.shape[1])
    x_p = D_p[0, :]
    y_p = D_p[1, :]

    D_p0 = mfp0.basic_dof_nodes()
    ones_p0 = np.ones(D_p0.shape[1])
    x_p0 = D_p0[0, :]
    y_p0 = D_p0[1, :]

    #D_p_init = mfp.basic_dof_nodes()
    #ones_p_init = np.ones(D_p_init.shape[1])
    #x_p_init = D_p_init[0, :]
    #y_p_init = D_p_init[1, :]

    D_ls = mfls.basic_dof_nodes()
    ones_ls = np.ones(D_ls.shape[1])
    x_ls = D_ls[0, :]
    y_ls = D_ls[1, :]
    # n = int(np.sqrt(x_ls.shape[0]))
    if ins.ndim == 2:
        edges_ls = np.where((x_ls < dx) | (x_ls > (ins.L_x - dx)) | (y_ls < dy) | (y_ls > (ins.L_y - dy)))

    D_mat = mfmat.basic_dof_nodes()
    ones_mat = np.ones(D_mat.shape[1])
    x_mat = D_mat[0,:]
    y_mat = D_mat[1, :]

    D_mat0 = mfmat0.basic_dof_nodes()
    ones_mat0 = np.ones(D_mat0.shape[1])
    x_mat0 = D_mat0[0, :]
    y_mat0 = D_mat0[0, :]

    D_t = mft.basic_dof_nodes()
    ones_t = np.ones(D_t.shape[1])
    x_t = D_t[0, :]
    y_t = D_t[1, :]

    D_mu = mfmu.basic_dof_nodes()
    ones_mu = np.ones(D_mu.shape[1])
    x_mu = D_mu[0, :]
    y_mu = D_mu[1, :]

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
            pts = pts[:,((np.abs(ls1_interface)<=np.sqrt(dx ** 2 + dy ** 2)/100)&(ls3_interface>=0))|((ls1_interface<=np.sqrt(dx ** 2 + dy ** 2)/100)&(np.abs(ls3_interface)<=np.sqrt(dx ** 2 + dy ** 2)/100))]
            pts_edges = mesh.pts(PIDs=mesh.pid_in_regions(np.arange(4) + 1))
            ls1_interface_edges = compute_interpolate_on(mfls, ls1.values(0), pts_edges)
            ls3_interface_edges = compute_interpolate_on(mfls, ls3.values(0), pts_edges)
            pts_edges = pts_edges[:, (ls1_interface_edges < 0)&(ls3_interface_edges >= 0)]
            pts = np.append(pts, pts_edges, axis=1)

        else:
            pts = pts[:, ((np.abs(ls1_interface) <= np.sqrt(dx ** 2 + dy ** 2) / 100))]
            pts_edges = mesh.pts(PIDs=mesh.pid_in_regions(np.arange(4) + 1))
            ls1_interface_edges = compute_interpolate_on(mfls, ls1.values(0), pts_edges)
            pts_edges = pts_edges[:, ls1_interface_edges < 0]
            pts = np.append(pts, pts_edges, axis=1)

        if 'planar' in ins.symmetry:
            area_init = alphashape.alphashape(pts.transpose(), 2*np.sqrt(dx ** 2 + dy ** 2)).area
        else:
            area_init = 1
        area = area_init

    if (ins.free_surface & (~ins.solve_air)) | (ins.topography & (~ins.solve_topography)):
        mfp_cut = gf.MeshFem('partial', mfp, np.arange(mfp.nbdof()))
        mfu_cut = gf.MeshFem('partial', mfu, np.arange(mfu.nbdof()))
        if ins.temperature:
            mft_cut = gf.MeshFem('partial', mft, np.arange(mft.nbdof()))

        #print(eval(ind_p))
        mfp_cut.set_partial(eval(ind_p))
        mfu_cut.set_partial(eval(ind_u))
        if ins.temperature:
            mft_cut.set_partial(eval(ind_t))


    ######## Problem setup ################

    md_init = gf.Model('real')  # real vs complex system
    md = gf.Model('real')  # real vs complex system

    # Add Data

    if ins.restart:
        md.add_initialized_fem_data('Previous_u', mfu, U)
        md.add_initialized_fem_data('Previous2_u', mfu, Previous_u)
        md.add_initialized_fem_data('Previous_p', mfp_previous, P)
        md.add_initialized_fem_data('Previous2_p', mfp_previous2, Previous_p)
        md.add_initialized_fem_data('Previous_d', mfu, D)
        md.add_initialized_fem_data('Previous2_d', mfu, Previous_d)
        if ins.free_surface:
            md.add_initialized_fem_data('Previous_psi', mfls, last2_Ls1)
            md.add_initialized_fem_data('Previous2_psi', mfls, last_Ls1)

        if ins.temperature & ins.solidification:
            ls2.set_values((T-Tg)/Tg)


    else:
        u_init = ones_u * 0
        p_init = ones_p * ins.p_amb
        d_init = ones_u * 0
        md.add_initialized_fem_data('Previous_u', mfu, u_init)
        md.add_initialized_fem_data('Previous2_u', mfu, u_init)
        md.add_initialized_fem_data('p_init', mfp_init, p_init)
        md.add_initialized_fem_data('Previous_p', mfp_previous, p_init)
        md.add_initialized_fem_data('Previous2_p', mfp_previous2, p_init)
        md.add_initialized_fem_data('Previous_d', mfu, d_init)
        md.add_initialized_fem_data('Previous2_d', mfu, d_init)
        if ins.free_surface:
            psi_init = ls1.values(0)
            md.add_initialized_fem_data('psi_init', mfls, psi_init)
            md_init.add_initialized_fem_data('psi_init', mfls, psi_init)
            md.add_initialized_fem_data('Previous_psi', mfls, psi_init)
            md.add_initialized_fem_data('Previous2_psi', mfls, psi_init)

    fext = 0*ones_u
    md.add_initialized_fem_data('fext',mfu, 0*ones_u)
    md.add_initialized_fem_data('Previous_fext', mfu, 0 * ones_u)

    # Density
    if ins.free_surface:
        ls1_mat = compute_interpolate_on(mfls,ls1.values(0),mfmat.basic_dof_nodes())
    if ins.topography:
        ls3_mat = compute_interpolate_on(mfls,ls3.values(0),mfmat.basic_dof_nodes())
    rho = ones_mat * ins.rho1
    if ins.free_surface:
        rho[ls1_mat > 0] = ins.rho2
    if ins.topography:
        rho[ls3_mat < 0] = ins.rho3
    Previous_rho = rho.copy()
    rho_init = rho.copy()
    md.add_initialized_fem_data('rho', mfmat, rho)
    md.add_initialized_fem_data('Previous_rho', mfmat_previous, rho)
    md.add_initialized_fem_data('Previous2_rho', mfmat_previous2, rho)
    md.add_initialized_fem_data('rho_init', mfmat_init, rho)
    md_init.add_initialized_fem_data('rho', mfmat, rho)

    md.add_initialized_fem_data('pamb', mfp, [ins.p_amb * ones_p])
    md_init.add_initialized_fem_data('pamb', mfp, [ins.p_amb * ones_p])
    md_init.add_initialized_data('surface_tension', [ins.surface_tension])
    md.add_initialized_data('surface_tension', [ins.surface_tension])

    #md.add_initialized_fem_data('p2', mfp0, [ins.p_bound2_val * ones_p0])
    #md_init.add_initialized_fem_data('p2', mfp0, [ins.p_bound2_val * ones_p0])

    # Compressibility
    if ins.compressible:
        beta = ones_mat * ins.beta1
        if ins.free_surface:
            beta[ls1_mat > 0] = ins.beta2
        if ins.topography:
            beta[ls3_mat < 0] = ins.beta3
    else:
        beta = ones_mat * (1e-9) / rho
    #beta[beta<(1e-9)/rho] = (1e-9)/rho[beta<(1e-9)/rho]
    beta_init = beta.copy()
    md.add_initialized_fem_data('beta', mfmat, beta)
    md.add_initialized_fem_data('Previous_beta', mfmat_previous, beta)

    if ins.temperature:
        # Thermal diffusivity
        kappa = ones_mat * ins.kappa1
        if ins.free_surface:
            kappa[ls1_mat > 0] = ins.kappa2
        if ins.topography:
            kappa[ls3_mat < 0] = ins.kappa3
        kappa_init = kappa.copy()
        md.add_initialized_fem_data('kappa', mfmat, kappa)
        md.add_initialized_fem_data('Previous_kappa', mfmat_previous, kappa)

        cp = ones_mat * ins.cp1
        if ins.free_surface:
            cp[ls1_mat > 0] = ins.cp2
        if ins.topography:
            cp[ls3_mat < 0] = ins.cp3
        md.add_initialized_fem_data('cp', mfmat, cp)

    # Temperature
    if ins.temperature:

        # Initial temp
        if ins.restart:
            md.add_initialized_fem_data('Previous_t', mft_previous, T)
            md.add_initialized_fem_data('Previous2_t', mft_previous2, Previous_t)
            t_init = Previous_t

        else:
            if (type(ins.T_init) is float) or (type(ins.T_init) is int):
                t_init = mft.eval('1') * ins.T_init
            elif type(ins.T_init) is str:
                t_init = eval(ins.T_init.replace('X','x_t').replace('Y','y_t')) * ones_t
            elif type(ins.T_init) is type(None):
                t_init = mft.eval('1') * ins.T0
                if ins.free_surface:
                    ls1_t = compute_interpolate_on(mfls,ls1.values(0),mft)
                    t_init[ls1_t > np.sqrt(dx**2 + dy**2)/20] = ins.T_amb
                if ins.topography:
                    ls3_t = compute_interpolate_on(mfls, ls3.values(0), mft)
                    t_init[ls3_t < 0] = ins.T_amb

            md.add_initialized_data('tamb',[ins.T_amb])
            md.add_initialized_fem_data('Previous_t', mft_previous, t_init)
            md.add_initialized_fem_data('Previous2_t', mft_previous2, t_init)

        T = t_init
        T_mu = compute_interpolate_on(mft, T, mfmu.basic_dof_nodes())
        eta = eval(ins.eta_exp.replace('vfta', str(ins.vfta)).replace('vftb', str(ins.vftb)).replace(
            'vftc', str(ins.vftc)).replace('etar', str(ins.etar)).replace('T','T_mu').replace('exp', 'np.exp'))
        eta[T_mu <= ins.vftb/(np.log10(ins.max_eta)-ins.vfta) - 273 + ins.vftc] = ins.max_eta

        if ins.free_surface:
            ls1_mu = compute_interpolate_on(mfls, ls1.values(0), mfmu.basic_dof_nodes())
        if ins.topography:
            ls3_mu= compute_interpolate_on(mfls, ls3.values(0), mfmu.basic_dof_nodes())

        #T_ls = compute_interpolate_on(mft, t_init, mfls)
        if ins.solidification:
            ls2.set_values((t_init - Tg) / Tg)
            mls2.adapt()

        if ins.free_surface:
            eta[ls1_mu > 0] = ins.eta2
        if ins.topography:
            eta[ls3_mu < 0] = ins.eta3
    else:
        if ins.free_surface:
            ls1_mu = compute_interpolate_on(mfls, ls1.values(0), mfmu.basic_dof_nodes())
        if ins.topography:
            ls3_mu= compute_interpolate_on(mfls, ls3.values(0), mfmu.basic_dof_nodes())

        eta = eval(ins.eta_exp.replace('exp', 'np.exp').replace('T', str(ins.T0)).replace('vft','ins.vft').replace('etar','ins.etar')) * ones_mu
        eta[eta > ins.max_eta] = ins.max_eta
        eta[ins.T0 * ones_mu <= (ins.vftb / (np.log10(ins.max_eta/ins.etar) - ins.vfta) - 273 + ins.vftc)] = ins.max_eta

        if ins.free_surface:
            eta[ls1_mu > 0] = ins.eta2
        if ins.topography:
            eta[ls3_mu < 0] = ins.eta3

    # time dependence
    md.add_initialized_data('dt', [ins.dt/ins.ndt0])
    md.add_initialized_data('BDF0', [1]) #1
    md.add_initialized_data('BDF1', [-1]) #-1
    md.add_initialized_data('BDF2', [0]) #0
    md.add_initialized_data('BDFf', [0]) #1
    md.add_initialized_data('BDFf_t', [0])#1

    md_init.add_initialized_data('dt', [ins.dt/ins.ndt0])

    # Elasticity
    lam = -2/3*eta
    mu = eta
    solid = 0 * ones_mat
    topo = 1*ones_mat

    if ins.temperature & ins.solidification:
        mu_solid = ins.E / (2 * (1 + ins.nu))
        lam_solid = ins.E * ins.nu / ((1 + ins.nu) * (1 - 2 * ins.nu)) + mu_solid
        ls2_mat = compute_interpolate_on(mft0,ls2.values(0),mfmat.basic_dof_nodes())
        ls2_mu = compute_interpolate_on(mft0, ls2.values(0), mfmu.basic_dof_nodes())
        if ins.free_surface:
            lam[(ls1_mu <= 0) & (ls2_mu <= 0)] = lam_solid
            mu[(ls1_mu <= 0) & (ls2_mu <= 0)] = mu_solid
            solid[(ls1_mat <= 0) & (ls2_mat <= 0)] = 1
            beta[(ls1_mat <= 0) & (ls2_mat <= 0)] = 3 * (1 - 2 * ins.nu) / ins.E  # 1/bulk modulus

        else:
            lam[(ls2_mu <= 0)] = lam_solid
            mu[(ls2_mu <= 0)] = mu_solid
            solid[(ls2_mat <= 0)] = 1
            beta[(ls2_mat <= 0)] = 3 * (1 - 2 * ins.nu) / ins.E  # 1/bulk modulus

        if ins.free_surface:
            solid = sciinterp.griddata(np.array([x_grid.flatten(), y_grid.flatten()]).transpose(),
                                   remove_small_objects(sciinterp.griddata(D_mat.transpose(), solid,
                                                        np.array([x_grid.flatten(), y_grid.flatten()]).transpose(),
                                                        method='nearest').reshape(x_grid.shape).astype(int),min_size=9).flatten(),
                                   D_mat.transpose(),method='nearest')
        md.set_variable('beta',beta)
        md.set_variable('Previous_beta',beta)

    if ins.topography:
        ls3_mat = compute_interpolate_on(mfls, ls3.values(0), mfmat.basic_dof_nodes())
        topo[ls3_mat<0] = 0

    md.add_initialized_fem_data('lambda', mfmu, lam)
    md.add_initialized_fem_data('mu', mfmu, mu)
    md.add_initialized_fem_data('solid', mfmat, solid)
    md.add_initialized_fem_data('topo', mfmat, topo)

    md.add_initialized_fem_data('E', mfmat, ins.E*solid)
    md.add_initialized_fem_data('nu', mfmat, ins.nu*solid)

    md.add_initialized_fem_data('Previous_lambda', mfmu_previous, lam)
    md.add_initialized_fem_data('Previous_mu', mfmu_previous, mu)
    md.add_initialized_fem_data('Previous_solid', mfmat_previous, solid)

    md_init.add_initialized_fem_data('lambda', mfmu, lam)
    md_init.add_initialized_fem_data('mu', mfmu, mu)
    md_init.add_initialized_fem_data('solid', mfmat, solid)
    md_init.add_initialized_fem_data('topo', mfmat, topo)

    if ins.free_surface | ins.topography:
        mls_cut = mls.cut_mesh()
        radii = np.zeros(mls_cut.nbpts())
        D_cut = mls_cut.pts()
        for pid in mls_cut.pid():
            radii[pid] = np.min(mls_cut.convex_radius(mls_cut.cvid_from_pid(pid, share=True)))
        if ins.temperature:
            radii_t = sciinterp.griddata(D_cut.transpose(), radii, D_t.transpose())
    else:
        radii = np.mean(mesh.convex_radius()) * ones_ls
        if ins.temperature:
            radii_t = np.mean(mesh.convex_radius()) * ones_t

        if 'axial' in ins.symmetry:
            if ins.temperature:
                radii_t = radii_t*2*np.pi*(x_t + dx/2) # in the axial case the surface area is scaled by 2*pi*mean radius

    # Add governing equations
    # Add variables

    if (ins.free_surface & (~ins.solve_air)) | (ins.topography & (~ins.solve_topography)):
        md.add_fem_variable('u', mfu_cut)
        md.add_fem_variable('d', mfu_cut)
        md.add_fem_variable('p', mfp_cut)
        if ins.temperature:
            md.add_fem_variable('t', mft_cut)
    else:
        md.add_fem_variable('u', mfu)
        md.add_fem_variable('d', mfu)
        md.add_fem_variable('p', mfp)
        if ins.temperature:
            md.add_fem_variable('t', mft)

    if ins.free_surface:
        md.add_fem_variable('psi',mfls)
        md.add_fem_variable('psis',mfls)
        md.add_initialized_data('psik',ins.kappa_psi)

        Psi_grid = sciinterp.griddata(D_ls.transpose(), ls1.values(0),
                                           np.array([x_grid.flatten(), y_grid.flatten()]).transpose(), method='cubic').reshape(x_grid.shape)

        dx_Psi_grid,dy_Psi_grid,curvature,mag_grad_Psi_grid = compute_curvature(Psi_grid, dx/ins.u_k, dy/ins.u_k, symmetry=ins.symmetry)
        md.add_initialized_fem_data('curvature',mfcurve,
                                    sciinterp.griddata(np.array([x_grid.flatten(), y_grid.flatten()]).transpose(), curvature.flatten(),
                                           D_curve.transpose(),
                                           method='nearest').flatten())
        md_init.add_initialized_fem_data('curvature',mfcurve,
                                    sciinterp.griddata(np.array([x_grid.flatten(), y_grid.flatten()]).transpose(), curvature.flatten(),
                                           D_curve.transpose(),
                                           method='nearest').flatten())

    # Define variables and weak forms

    md_init.add_fem_variable('u', mfu)
    md_init.add_fem_variable('p', mfp)

    if 'planar' in ins.symmetry:
        md_init.add_linear_incompressibility_brick(mim_all, 'u', 'p')
        md_init.add_isotropic_linearized_elasticity_brick(mim_all, 'u', 'lambda*(dt*solid + (1-solid))', 'mu*(dt*solid + (1-solid))')

    elif 'axial' in ins.symmetry:
        md_init.add_linear_incompressibility_brick(mim_all, 'u', 'p')
        md_init.add_nonlinear_term(mim_all,'([1/X(1)/rho,0].u)*Test_p')
        md_init.add_isotropic_linearized_elasticity_brick(mim_all, 'u', 'lambda*(dt*solid + (1-solid))',
                                                          'mu*(dt*solid + (1-solid))')
        md_init.add_nonlinear_term(mim_all,'(1/X(1)*lambda*[u(1),0;u(1),0]):Grad_Test_u')
        md_init.add_nonlinear_term(mim_all,'[u(1)/X(1)/X(1),0]:Test_u')

    if ins.free_surface & ins.solve_air:
       md_init.add_nonlinear_term(mim_fluidbound, str(ins.max_eta*1e6) + "*((Xfem_plus(p)-Xfem_minus(p))+surface_tension*curvature).(Xfem_plus(Test_p)-Xfem_minus(Test_p))")

    md.add_macro('h', 'element_size')

    # compressible mass balance
    #if ins.compressible or (ins.temperature & ins.solidification):
    md.add_macro('rhof', 'rho*(1 + beta*(p-pamb))')
    md.add_macro('Previous_rhof', 'Previous_rho*(1 + beta*(Previous_p-pamb))')
    md.add_macro('Previous2_rhof', 'Previous2_rho*(1 + beta*(Previous2_p-pamb))')

    time_int_p = '(BDF0*p+BDF1*Previous_p+BDF2*Previous2_p)/dt'

    if ins.compressible or (ins.temperature & ins.solidification):
        if 'planar' in ins.symmetry:
            md.add_nonlinear_term(mim, time_int_p + '*Test_p')
            md.add_nonlinear_term(mim, '1/(rho*beta)*rhof*Div_u*Test_p + u.Grad(p)*Test_p')
            md.add_nonlinear_term(mim, 'BDFf*(1/(Previous_rho*Previous_beta)*Previous_rhof*Div_Previous_u*Test_p + Previous_u.Grad(Previous_p)*Test_p)')
        elif 'axial' in ins.symmetry:
            md.add_nonlinear_term(mim, time_int_p + '*X(1)*Test_p')
            md.add_nonlinear_term(mim, 'X(1)/(rho*beta)*rhof*Div_u*Test_p + u.Grad(X(1)*p)*Test_p')
            md.add_nonlinear_term(mim, 'BDFf*(X(1)/(rho*beta)*Previous_rhof*Div_Previous_u*Test_p + Previous_u.Grad(X(1)*Previous_p)*Test_p)')
        if 'SUPG' in ins.stab_p:
            if 'planar' in ins.symmetry:
                S_SUPG_p = "(" +  time_int_p + "*(u.Grad_Test_p)" + " + 1/(rho*beta)*rhof*Div_u*(u.Grad_Test_p) + u.Grad_p*(u.Grad_Test_p))"
                S_SUPG_p_init = "(" + time_int_p + "*(Previous_u.Grad_Test_p)" + " + 1/(Previous_rho*Previous_beta)*(Previous_rhof*Div_Previous_u)*(Previous_u.Grad_Test_p) + Previous_u.Grad_p*(Previous_u.Grad_Test_p))"
                tau_SUPG_p = '1/(2/dt + 2*Norm(u)/h)'
                tau_SUPG_p_init = 'BDFf*(1/(2/dt+ 2*Norm(Previous_u)/h))'
            elif 'axial' in ins.symmetry:
                S_SUPG_p = "(" + time_int_p + "*(u.Grad(X(1)*Test_p))" + " + X(1)/(rho*beta)*rhof*Div_u*(u.Grad(X(1)*Test_p)) + u.Grad(X(1)*p)*(u.Grad(X(1)*Test_p)))"
                S_SUPG_p_init = "(" + time_int_p + "*(Previous_u.Grad(X(1)*Test_p))" + " + X(1)/(Previous_rho*Previous_beta)*Previous_rhof*Div_Previous_u*(Previous_u.Grad(X(1)*Test_p)) + Previous_u.Grad(X(1)*p)*(Previous_u.Grad(X(1)*Test_p)))"
                tau_SUPG_p = '1/(2/dt + 2*Norm(X(1)*u)/h)'
                tau_SUPG_p_init = 'BDFf*(1/(2/dt + 2*Norm(X(1)*Previous_u)/h))'

            md.add_nonlinear_term(mim, tau_SUPG_p + '*' + S_SUPG_p)
            md.add_nonlinear_term(mim, tau_SUPG_p_init + '*' + S_SUPG_p_init)

    else:
        if ins.steady:
            md.add_linear_incompressibility_brick(mim, 'u', 'p')
            if 'axial' in ins.symmetry:
                md.add_nonlinear_term(mim_all, '(u.([1,0]*p))*Test_p')
        else:
            if 'planar' in ins.symmetry:
                md.add_nonlinear_term(mim, time_int_p + '*Test_p')
                md.add_nonlinear_term(mim, '1/(rho*beta)*rhof*Div_u*Test_p + u.Grad_p*Test_p')
                md.add_nonlinear_term(mim,
                                      'BDFf*(1/(Previous_rho*Previous_beta)*Previous_rhof*Div_Previous_u*Test_p + Previous_u.Grad_Previous_p*Test_p)')
            elif 'axial' in ins.symmetry:
                md.add_nonlinear_term(mim, time_int_p + '*X(1)*Test_p')
                md.add_nonlinear_term(mim, 'X(1)/(rho*beta)*rhof*Div_u*Test_p + u.Grad(X(1)*p)*Test_p')
                md.add_nonlinear_term(mim,
                                      'BDFf*(X(1)/(rho*beta)*Previous_rhof*Div_Previous_u*Test_p + Previous_u.Grad(X(1)*Previous_p)*Test_p)')

    if ins.free_surface & ins.solve_air:
        md.add_nonlinear_term(mim_fluidbound, str(ins.max_eta*1e6) + "*((Xfem_plus(p)-Xfem_minus(p))+surface_tension*curvature).(Xfem_plus(Test_p)-Xfem_minus(Test_p))")
        if ins.temperature & ins.t_enrich:
            md.add_nonlinear_term(mim_fluidbound, str(ins.max_eta*1e6) + "*((Xfem_plus(t)-Xfem_minus(t))-(1150-tamb)/2).(Xfem_plus(Test_t)-Xfem_minus(Test_t))")

    # mometum balance
    if ins.compressible:
        time_int_u = "rhof*(BDF0*u+BDF1*Previous_u+BDF2*Previous2_u)/dt.Test_u"
    else:
        time_int_u = "rho*(BDF0*u+BDF1*Previous_u+BDF2*Previous2_u)/dt.Test_u"

    if 'planar' in ins.symmetry:
        if ins.steady:
            md.add_nonlinear_term(mim, 'solid*' + time_int_u)
            md.add_isotropic_linearized_elasticity_brick(mim, 'u', 'lambda*(dt*solid + (1-solid))',
                                                              'mu*(dt*solid + (1-solid))')
            md.set_variable('BDFf', [0])
            linear_elastic = "0"
            linear_elastic_init = "0"
        else:
            md.add_nonlinear_term(mim, time_int_u)
            md.add_nonlinear_term(mim, "Grad_p.Test_u + BDFf*Grad_Previous_p.Test_u")
            linear_elastic = "(lambda*(Div_u*Div_Test_u) + mu*((Grad_u + Grad_u'):Grad_Test_u))"
            linear_elastic_init = "BDFf*(Previous_lambda*(Div_Previous_u*Div_Test_u) + Previous_mu*((Grad_Previous_u + Grad_Previous_u'):Grad_Test_u))"
    elif 'axial' in ins.symmetry:
        if ins.steady:
            md.add_nonlinear_term(mim, 'solid*X(1)*' + time_int_u)
            md.set_variable('BDFf', [0])
            md.add_nonlinear_term(mim, 'u.([1,0]*p)*Test_p')
            md.add_isotropic_linearized_elasticity_brick(mim_all, 'u', 'lambda*(dt*solid + (1-solid))',
                                                              'mu*(dt*solid + (1-solid))')
            md.add_nonlinear_term(mim_all, '(1/X(1)*lambda*[u(1),0;u(1),0]):Grad_Test_u')
            md.add_nonlinear_term(mim_all, '[u(1)/X(1)/X(1),0]:Test_u')

            md.set_variable('BDFf', [0])
            linear_elastic = "0"
            linear_elastic_init = "0"
        else:
            md.add_nonlinear_term(mim, 'X(1)*' + time_int_u)
            md.add_nonlinear_term(mim, "X(1)*Grad_p.Test_u + X(1)*BDFf*Grad_Previous_p.Test_u")

            linear_elastic = "(lambda*(Trace(Grad(X(1)*u)))*Div_Test_u + mu*((X(1)*Grad_u + X(1)*Grad_u'):Grad_Test_u + [u(1)/X(1),0]:Test_u))"
            linear_elastic_init = "BDFf*(Previous_lambda*(Trace(Grad(X(1)*Previous_u))*Div_Test_u) + Previous_mu*((X(1)*Grad_Previous_u + (X(1)*Grad_Previous_u)'):Grad_Test_u + [Previous_u(1)/X(1),0]:Test_u))"

    if ins.temperature & ins.solidification:
        md.add_nonlinear_term(mim, linear_elastic + '*(dt/BDF0*solid + (1-solid))')
        md.add_nonlinear_term(mim, linear_elastic_init + '*(dt/BDF0*Previous_solid + (1-Previous_solid))')

        if 'planar' in ins.symmetry:
            linear_elastic_adv = "-(lambda*(Trace(Grad(u.Grad_Previous_d))*Div_Test_u) + mu*((Grad(u.Grad_Previous_d) + Grad(u.Grad_Previous_d)'):Grad_Test_u))*(dt/BDF0*Previous_solid)"
            residual_stress1 = "-(lambda*(Div_Previous_d*Div_Test_u) + mu*((Grad_Previous_d + Grad_Previous_d'):Grad_Test_u))*(BDF1/BDF0)*solid"
            residual_stress2 = "-(lambda*(Div_Previous2_d*Div_Test_u) + mu*((Grad_Previous2_d + Grad_Previous2_d'):Grad_Test_u))*(BDF2/BDF0)*solid"
        elif 'axial' in ins.symmetry:
            linear_elastic_adv = "-(lambda*Trace(Grad(u.Grad(X(1)*Previous_d))*Div_Test_u) + mu*((X(1)*Grad(1/X(1)*u.Grad(X(1)*Previous_d)) + X(1)*Grad(1/X(1)*u.Grad(X(1)*Previous_d))'):Grad_Test_u + [(u.Grad(X(1)*Previous_d))(1)/X(1)/X(1),0]:Test_u))*(dt/BDF0*Previous_solid)"
            residual_stress1 = "-(lambda*(Trace(Grad(X(1)*Previous_d))*Div_Test_u) + mu*((X(1)*Grad_Previous_d + (X(1)*Grad_Previous_d)'):Grad_Test_u + [Previous_d(1)/X(1),0]:Test_u))*(BDF1/BDF0)*solid"
            residual_stress2 = "-(lambda*(Trace(Grad(X(1)*Previous2_d))*Div_Test_u) + mu*((X(1)*Grad_Previous2_d + (X(1)*Grad_Previous2_d)'):Grad_Test_u + [Previous2_d(1)/X(1),0]:Test_u))*(BDF2/BDF0)*solid"

        md.add_nonlinear_term(mim, linear_elastic_adv)
        md.add_nonlinear_term(mim, residual_stress1 + '+' + residual_stress2)

    else:
        md.add_nonlinear_term(mim, linear_elastic)
        md.add_nonlinear_term(mim, linear_elastic_init)


    if ins.free_surface:
        md.add_nonlinear_term(mim_all, "psis*Test_psis - psi*Test_psis + psik*Grad_psis.Grad_Test_psis")

        if 'planar' in ins.symmetry:
            time_int_psi = "((BDF0*psi+BDF1*Previous_psi+BDF2*Previous2_psi)/dt)"
            advection_psi = "(fext.Grad_psi)*Test_psi"
            advection_psi_init = "BDFf_t*(Previous_fext.Grad_Previous_psi)*Test_psi"
        elif 'axial' in ins.symmetry:
            time_int_psi = "(X(1)*(BDF0*psi+BDF1*Previous_psi+BDF2*Previous2_psi)/dt)"
            advection_psi = "(fext.Grad(X(1)*psi))*Test_psi"
            advection_psi_init = "BDFf_t*(Previous_fext.Grad(X(1)*Previous_psi))*Test_psi"

        md.add_nonlinear_term(mim_all, time_int_psi + '*Test_psi')
        md.add_nonlinear_term(mim_all, advection_psi)
        md.add_nonlinear_term(mim_all, advection_psi_init)

        if 'SUPG' in ins.stab_psi:
            if 'planar' in ins.symmetry:
                tau_SUPG_psi = "1/(2/dt + 2*Norm(fext)/h)"
                S_SUPG_psi = "(fext.Grad_psi).(fext.Grad_Test_psi)"
                md.add_nonlinear_term(mim, tau_SUPG_psi + '*' + time_int_psi + '.(fext.Grad_Test_psi)')
                md.add_nonlinear_term(mim, tau_SUPG_psi + '*' + S_SUPG_psi)

                tau_SUPG_psi_init = "BDFf_t/(2/dt + 2*Norm(Previous_fext)/h)"
                S_SUPG_psi_init = "(Previous_fext.Grad_Previous_psi).(Previous_fext.Grad_Test_psi)"
                md.add_nonlinear_term(mim, tau_SUPG_psi_init + '*' + time_int_psi + '.(Previous_fext.Grad_Test_psi)')
                md.add_nonlinear_term(mim, tau_SUPG_psi_init + '*' + S_SUPG_psi_init)
            elif 'axial' in ins.symmetry:
                tau_SUPG_psi = "1/(2/dt + 2*Norm(X(1)*fext)/h)"
                S_SUPG_psi = "(fext.Grad(X(1)*psi)).(fext.Grad(X(1)*Test_psi))"
                md.add_nonlinear_term(mim, tau_SUPG_psi + '*' + time_int_psi + '.(fext.Grad(X(1)*Test_psi))')
                md.add_nonlinear_term(mim, tau_SUPG_psi + '*' + S_SUPG_psi)

                tau_SUPG_psi_init = "BDFf_t/(2/dt + 2*Norm(X(1)*Previous_fext)/h)"
                S_SUPG_psi_init = "(Previous_fext.Grad(X(1)*Previous_psi)).(Previous_fext.Grad(X(1)*Test_psi))"
                md.add_nonlinear_term(mim, tau_SUPG_psi_init + '*' + time_int_psi + '.(Previous_fext.Grad(X(1)*Test_psi))')
                md.add_nonlinear_term(mim, tau_SUPG_psi_init + '*' + S_SUPG_psi_init)

        elif 'GLS' in ins.stab_psi:
            if 'planar' in ins.symmetry:
                tau_GLS_psi = "1/(2/dt + 2*Norm(fext)/h)"
                S_GLS_psi = "((fext.Grad_psi)*(Trace(Grad(fext*Test_psi))))"
                md.add_nonlinear_term(mim, tau_GLS_psi + '*' + time_int_psi + '.(Trace(Grad(fext*Test_psi)))')
                md.add_nonlinear_term(mim, tau_GLS_psi + '*' + S_GLS_psi)

                tau_GLS_psi_init = "BDFf_t/(2/dt+ 2*Norm(Previous_fext)/h)"
                S_GLS_psi_init = "((Previous_fext.Grad_Previous_psi)*(Trace(Grad(Previous_fext*Test_psi))))"
                md.add_nonlinear_term(mim, tau_GLS_psi_init + '*' + time_int_psi + '.(Trace(Grad(Previous_fext*Test_psi)))')
                md.add_nonlinear_term(mim, tau_GLS_psi_init + '*' + S_GLS_psi_init)
            elif 'axial' in ins.symmetry:
                tau_GLS_psi = "1/(2/dt + 2*Norm(X(1)*fext)/h)"
                S_GLS_psi = "((fext.Grad(X(1)*psi))*(Trace(Grad(X(1)*fext*Test_psi))))"
                md.add_nonlinear_term(mim, tau_GLS_psi + '*' + time_int_psi + '.(Trace(Grad(X(1)*fext*Test_psi)))')
                md.add_nonlinear_term(mim, tau_GLS_psi + '*' + S_GLS_psi)

                tau_GLS_psi_init = "BDFf_t/(2/dt + 2*Norm(X(1)*Previous_fext)/h)"
                S_GLS_psi_init = "((Previous_fext.Grad(X(1)*Previous_psi))*(Trace(Grad(X(1)*Previous_fext*Test_psi))))"
                md.add_nonlinear_term(mim,
                                      tau_GLS_psi_init + '*' + time_int_psi + '.(Trace(Grad(X(1)*Previous_fext*Test_psi)))')
                md.add_nonlinear_term(mim, tau_GLS_psi_init + '*' + S_GLS_psi_init)

        md.disable_variable('psis')
        md.disable_variable('psi')

    if ins.temperature:
        time_int_t = "(BDF0*t+BDF1*Previous_t+BDF2*Previous2_t)/dt"
        if 'planar' in ins.symmetry:
            advection_t = "(u.Grad_t)*Test_t"
            advection_t_init = "BDFf_t*(Previous_u.Grad_Previous_t)*Test_t"
            diffusion = "kappa*(Grad_t.Grad_Test_t)"
            diffusion_init = "BDFf_t*Previous_kappa*(Grad_Previous_t.Grad_Test_t)"

            md.add_macro('Pe', "h*Norm(u)/(2*kappa)")
            md.add_macro('Pe_init', "h*Norm(Previous_u)/(2*Previous_kappa)")

        elif 'axial' in ins.symmetry:
            time_int_t = "X(1)*" + time_int_t
            advection_t = "(u.Grad(X(1)*t))*Test_t"
            advection_t_init = "BDFf_t*(Previous_u.Grad(X(1)*Previous_t))*Test_t"
            diffusion = "kappa*((X(1)*Grad_t).Grad_Test_t)"
            diffusion_init = "BDFf_t*Previous_kappa*((X(1)*Grad_Previous_t).Grad_Test_t)"

            md.add_macro('Pe', "h*Norm(X(1)*u)/(2*kappa)")
            md.add_macro('Pe_init', "h*Norm(X(1)*Previous_u)/(2*Previous_kappa)")

        md.add_macro('xi', "min(Pe/3,1)")
        md.add_macro('xi_init', "min(Pe_init/3,1)")

        md.add_nonlinear_term(mim, time_int_t + '*Test_t')
        md.add_nonlinear_term(mim, advection_t)
        md.add_nonlinear_term(mim, diffusion)
        md.add_nonlinear_term(mim, advection_t_init)
        md.add_nonlinear_term(mim, diffusion_init)

        if 'SUPG' in ins.stab_t:
            if 'planar' in ins.symmetry:
                S_SUPG_t = "((u.Grad_t)*(u.Grad_Test_t) + kappa*(Grad_t).Grad(u.Grad_Test_t) +" + time_int_t + "*(u.Grad_Test_t))"
                S_SUPG_t_init = "(((Previous_u.Grad_Previous_t)*(Previous_u.Grad_Test_t) + Previous_kappa*(Grad_Previous_t).Grad((X(1)*Previous_u).Grad_Test_t))+" + time_int_t + "*(Previous_u.Grad_Test_t)))"
                tau_SUPG_t = '1/(2/dt*(1+BDFf_t) + 4*kappa/(h*h) + 2*Norm(u)/h)*xi'
                tau_SUPG_t_init = 'BDFf_t/(2/dt + 4*Previous_kappa/(h*h) + 2*Norm(Previous_u)/h)*xi_init'
            elif 'axial' in ins.symmetry:
                S_SUPG_t = "(u.Grad(X(1)*t)*(u.Grad(X(1)*Test_t)) + kappa*((X(1)*Grad_t).Grad(u.Grad(X(1)*Test_t))) + " + time_int_t + "*u.Grad(X(1)*Test_t))"
                S_SUPG_t_init = "((Previous_u.Grad(X(1)*Previous_t))*(Previous_u.Grad(X(1)*Test_t)) + Previous_kappa*((X(1)*Grad_Previous_t).Grad(Previous_u.Grad(X(1)*Test_t))) + " + time_int_t + "*Previous_u.Grad(X(1)*Test_t))"
                tau_SUPG_t = '1/(2/dt*(1+BDFf_t) + 4*X(1)*kappa/(h*h) + 2*Norm(X(1)*u)/h)*xi'
                tau_SUPG_t_init = 'BDFf_t/(2/dt + 4*X(1)*Previous_kappa/(h*h) + 2*Norm(X(1)*Previous_u)/h)*xi_init'
            md.add_nonlinear_term(mim, tau_SUPG_t + '*' + S_SUPG_t)
            md.add_nonlinear_term(mim, tau_SUPG_t_init + '*' + S_SUPG_t_init)

        if 'GLS' in ins.stab_t:
            if 'planar' in ins.symmetry:
                S_SUPG_t = "((u.Grad_t)*(u.Grad_Test_t) + kappa*(Grad_t).Grad(u.Grad_Test_t) +" + time_int_t + "*(u.Grad_Test_t))"
                S_SUPG_t_init = "((Previous_u.Grad_Previous_t)*(Previous_u.Grad_Test_t) + Previous_kappa*(Grad_Previous_t).Grad(Previous_u.Grad_Test_t) + " + time_int_t + "*(Previous_u.Grad_Test_t))"
                S_GLS_t = "(" + S_SUPG_t + "- (u.Grad_t)*Trace(Grad(kappa*Grad_Test_t)) + " + time_int_t + "*(Trace(Grad(kappa*Grad_Test_t))))"
                S_GLS_t_init = "(" + S_SUPG_t_init + "- (Previous_u.Grad_Previous_t)*Trace(Grad(Previous_kappa*Grad_Test_t)) + " + time_int_t + "*(Trace(Grad(Previous_kappa*Grad_Test_t))))"
                tau_GLS_t = '1/(2/dt + 4*kappa/(h*h) + 2*Norm(u)/h)*xi'
                tau_GLS_t_init = 'BDFf_t/(2/dt + 4*Previous_kappa/(h*h) + 2*Norm(Previous_u)/h)*xi_init'

            elif 'axial' in ins.symmetry:
                S_SUPG_t = "(u.Grad(X(1)*t)*(u.Grad(X(1)*Test_t)) + kappa*((X(1)*Grad_t).Grad(u.Grad(X(1)*Test_t))) + " + time_int_t + "*u.Grad(X(1)*Test_t))"
                S_SUPG_t_init = "((Previous_u.Grad(X(1)*Previous_t))*(Previous_u.Grad(X(1)*Test_t)) + Previous_kappa*((X(1)*Grad_Previous_t).Grad(Previous_u.Grad(X(1)*Test_t))) + " + time_int_t + "*Previous_u.Grad(X(1)*Test_t))"
                S_GLS_t = "(" + S_SUPG_t + "- (u.Grad(X(1)*t))*Trace(Grad(kappa*(X(1)*Grad_Test_t))) + " + time_int_t + "*(Trace(Grad(kappa*(X(1)*Grad_Test_t)))))"
                S_GLS_t_init = "(" + S_SUPG_t_init + "- (Previous_u.Grad(X(1)*Previous_t))*Trace(Grad(Previous_kappa*(X(1)*Grad_Test_t))) + " + time_int_t + "*(Trace(Grad(Previous_kappa*(X(1)*Grad_Test_t)))))"
                tau_GLS_t = '1/(2/dt + 4*X(1)*kappa/(h*h) + 2*Norm(X(1)*u)/h)*xi'
                tau_GLS_t_init = 'BDFf_t/(2/dt + 4*X(1)*Previous_kappa/(h*h) + 2*Norm(X(1)*Previous_u)/h)*xi_init'

            md.add_nonlinear_term(mim, tau_GLS_t + '*' + S_GLS_t)
            md.add_nonlinear_term(mim, tau_GLS_t_init + '*' + S_GLS_t_init)


    if ins.temperature & ins.solidification:
        time_int_d = "((BDF0*d+(BDF1*Previous_d+BDF2*Previous2_d)*Previous_solid)/dt)"
        if 'planar' in ins.symmetry:
            advection_d = "(u.Grad_Previous_d).Test_d*Previous_solid"
            source_d = "(-u.Test_d*solid - BDFf*(Previous_u.Test_d)*Previous_solid)"
        elif 'axial' in ins.symmetry:
            time_int_d = "X(1)*" + time_int_d
            advection_d = "(u.Grad(X(1)*Previous_d)).Test_d*Previous_solid"
            source_d = "(-X(1)*u.Test_d*solid - BDFf*X(1)*(Previous_u.Test_d)*Previous_solid)"

        md.add_nonlinear_term(mim, time_int_d + '.Test_d')
        md.add_nonlinear_term(mim, advection_d)
        md.add_nonlinear_term(mim, source_d)

        if 'SUPG' in ins.stab_d:
            if 'planar' in ins.symmetry:
                S_SUPG_d = "((u.Grad_Previous_d).(u.Grad_Test_d) - u.(u.Grad_Test_d))"
                S_SUPG_d_init = "((Previous_u.Grad_Previous_d).(Previous_u.Grad_Test_d) - Previous_u.(Previous_u.Grad_Test_d))"
                tau_SUPG_d = "1/(2/dt + 2*Norm(u)/h)"
                tau_SUPG_d_init = "BDFf/(2/dt+ 2*Norm(Previous_u)/h)"

                md.add_nonlinear_term(mim, tau_SUPG_d + '*' + time_int_d + '.(u.Grad_Test_d)')
                md.add_nonlinear_term(mim, tau_SUPG_d_init + '*' + time_int_d + '.(Previous_u.Grad_Test_d)')
            elif 'axial' in ins.symmetry:
                S_SUPG_d = "((u.Grad(X(1)*Previous_d)).(u.Grad(X(1)*Test_d)) - X(1)*u.(u.Grad(X(1)*Test_d)))"
                S_SUPG_d_init = "((Previous_u.Grad(X(1)*Previous_d)).(Previous_u.Grad(X(1)*Test_d)) - X(1)*Previous_u.(Previous_u.Grad(X(1)*Test_d)))"
                tau_SUPG_d = "1/(2/dt + 2*Norm(X(1)*u)/h)"
                tau_SUPG_d_init = "BDFf/(2/dt+ 2*Norm(X(1)*Previous_u)/h)"

                md.add_nonlinear_term(mim, tau_SUPG_d + '*' + time_int_d + '.(u.Grad(X(1)*Test_d))')
                md.add_nonlinear_term(mim, tau_SUPG_d_init + '*' + time_int_d + '.(Previous_u.Grad(X(1)*Test_d))')

            md.add_nonlinear_term(mim, tau_SUPG_d + '*' + S_SUPG_d + '*solid')
            md.add_nonlinear_term(mim, tau_SUPG_d_init + '*' + S_SUPG_d_init + '*Previous_solid')

    md.disable_variable('d')

    # add body force
    if (type(ins.f_x) is type(None)) and (type(ins.f_y) is type(None)):
        f_xi = None
        f_yi = None
    else:
        if (type(ins.f_x) is float) or (type(ins.f_x) is int):
            f_xi = ones_mat * ins.f_x
        elif type(ins.f_x) is str:
            P_mat = ins.p_amb*ones_mat
            f_xi = eval(ins.f_x.replace('Y', 'y_mat').replace('X', 'x_mat')) * ones_mat
        elif type(ins.f_x) is type(None):
            f_xi = ones_mat * 0

        if (type(ins.f_y) is float) or (type(ins.f_y) is int):
            f_yi = ones_mat * ins.f_y
        elif type(ins.f_y) is str:
            P_mat = ins.p_amb * ones_mat
            f_yi = eval(ins.f_y.replace('Y', 'y_mat').replace('X', 'x_mat')) * ones_mat
        elif type(ins.f_y) is type(None):
            f_yi = ones_mat * 0

        md_init.add_initialized_fem_data('body', mfmat, [f_xi, f_yi])
        md_init.add_nonlinear_term(mim_all, 'body.Test_u')

        if ('axial' in ins.symmetry) & (~ins.steady):
            f_xi = x_mat * f_xi
            f_yi = x_mat * f_yi
        md.add_initialized_fem_data('body', mfmat, [f_xi, f_yi])
        md.add_nonlinear_term(mim, '-(1+BDFf)*body.Test_u')


    # add boundary conditions
    vars = ['u']

    for var in vars:
        for i, bound in enumerate(bounds):
            mesh.set_region(i + 1, eval('fb_' + bound))
            # Dirichlet boundaries
            dirichlet = False
            H = np.array([[0,0],[0,0]])
            if (type(eval('ins.' + bound + '_' + var + 'x')) is float) or (type(eval('ins.' + bound + '_' + var + 'x')) is int):
                data_x = eval('ins.' + bound + '_' + var + 'x') * ones_p
                u_init[mfu.basic_dof_on_region(i+1)][::2] = eval('ins.' + bound + '_' + var + 'x') * ones_u[mfu.basic_dof_on_region(i+1)][::2]
                H[0,0] = 1
                dirichlet = True
            if type(eval('ins.' + bound + '_' + var + 'x')) is str:
                data_x = eval(eval('ins.' + bound + '_' + var + 'x').replace('X','x_p').replace('Y','y_p')) * ones_p
                u_init[mfu.basic_dof_on_region(i + 1)][::2] = (eval(eval('ins.' + bound + '_' + var + 'x').replace('X','x_u').replace('Y','y_u')) * ones_u)[
                                                                                                       mfu.basic_dof_on_region(
                                                                                                           i + 1)][::2]
                dirichlet = True
                H[0,0] = 1
            if type(eval('ins.' + bound + '_' + var + 'y')) is type(None):
                data_y = 0 * ones_p

            if (type(eval('ins.' + bound + '_' + var + 'y')) is float) or (type(eval('ins.' + bound + '_' + var + 'y')) is int):
                data_y = eval('ins.' + bound + '_' + var + 'y') * ones_p
                u_init[mfu.basic_dof_on_region(i + 1)][1::2] = eval('ins.' + bound + '_' + var + 'y') * ones_u[
                                                                                                       mfu.basic_dof_on_region(
                                                                                                           i + 1)][1::2]
                dirichlet = True
                H[1,1] = 1
            if type(eval('ins.' + bound + '_' + var + 'y')) is str:
                data_y = eval(eval('ins.' + bound + '_' + var + 'y').replace('X','x_p').replace('Y','y_p')) * ones_p
                u_init[mfu.basic_dof_on_region(i + 1)][1::2] = (eval(eval('ins.' + bound + '_' + var + 'y').replace('X','x_u').replace('Y','y_u')) * ones_u)[
                                                                                                       mfu.basic_dof_on_region(
                                                                                                           i + 1)][1::2]
                dirichlet = True
                H[1,1] = 1
            if type(eval('ins.' + bound + '_' + var + 'x')) is type(None):
                data_x = 0 * ones_p

            if dirichlet:
                md.add_initialized_fem_data(bound + 'data_' + var, mfp, [data_x, data_y])
                md_init.add_initialized_fem_data(bound + 'data_' + var, mfp, [data_x, data_y])
                md.add_initialized_data(bound + 'H_' + var,H)
                md_init.add_initialized_data(bound + 'H_' + var, H)

                md.add_generalized_Dirichlet_condition_with_multipliers(mim, var, 2, i + 1,
                                                            dataname=bound + 'data_' + var,Hname=bound + 'H_' + var)
                md_init.add_generalized_Dirichlet_condition_with_multipliers(mim_all, var, 2, i + 1,
                                                                        dataname=bound + 'data_' + var, Hname=bound + 'H_' + var)
                md.set_variable('Previous2_u',u_init)
                md.set_variable('Previous_u', u_init)

            neumann = False
            if (type(eval('ins.' + bound + '_d' + var + 'x')) is float) or (type(eval('ins.' + bound + '_d' + var + 'x')) is int):
                data_dx = eval('ins.' + bound + '_d' + var + 'x') * ones_u
                neumann = True
            if type(eval('ins.' + bound + '_d' + var + 'x')) is str:
                data_dx = eval(eval('ins.' + bound + '_d' + var + 'x')) * ones_u
                neumann = True
            if type(eval('ins.' + bound + '_d' + var + 'y')) is type(None):
                data_dy = 0 * ones_u

            if (type(eval('ins.' + bound + '_d' + var + 'y')) is float) or (type(eval('ins.' + bound + '_d' + var + 'y')) is int):
                data_dy = eval('ins.' + bound + '_d' + var + 'y') * ones_u
                neumann = True
            if type(eval('ins.' + bound + '_d' + var + 'y')) is str:
                data_dy = eval(eval('ins.' + bound + '_d' + var + 'y')) * ones_u
                neumann = True
            if type(eval('ins.' + bound + '_d' + var + 'x')) is type(None):
                data_dx = 0 * ones_u

            if neumann:
                md.add_initialized_fem_data(bound + 'data_d' + var, mfu, [(1+md.variable('BDFf'))*data_dx, (1+md.variable('BDFf'))*data_dy])
                md_init.add_initialized_fem_data(bound + 'data_d' + var, mfu, [data_dx, data_dy])

                md.add_normal_source_term_brick(mim, var, bound + 'data_d' + var, i + 1)
                md_init.add_normal_source_term_brick(mim_all, var, bound + 'data_d' + var, i + 1)

    # add pressure on boundary or free surface

    if 'bottom_left' in ins.p_bound:
        md_init.add_initialized_data('pbound', [0, 0])
        md.add_initialized_data('pbound', [0, 0])
    if 'bottom_right' in ins.p_bound:
        md_init.add_initialized_data('pbound', [ins.L_x, 0])
        md.add_initialized_data('pbound', [ins.L_x, 0])
    if 'top_left' in ins.p_bound:
        md_init.add_initialized_data('pbound', [0, ins.L_y])
        md.add_initialized_data('pbound', [0, ins.L_y])
    if 'top_right' in ins.p_bound:
        md_init.add_initialized_data('pbound', [ins.L_x, ins.L_y])
        md.add_initialized_data('pbound', [ins.L_x, ins.L_y])

    md_init.add_initialized_data('pamb0',[ins.p_amb])
    md.add_initialized_data('pamb0', [ins.p_amb])

    if (not ins.free_surface) | ins.solve_air:
        md_init.add_pointwise_constraints_with_penalization('p',ins.max_eta*1e6,'pbound','pamb0')
        md.add_pointwise_constraints_with_penalization('p',ins.max_eta*1e6,'pbound','pamb0')
    else:
        md.add_nonlinear_term(mim_surf, str(ins.max_eta*1e6) + '*(p-pamb)*Test_p')

    if ins.temperature:
        for i, bound in enumerate(bounds):
            dirichlet = False
            # Dirichlet boundaries
            if (type(eval('ins.' + bound + '_t')) is float) or (type(eval('ins.' + bound + '_t')) is int):
                data_t = eval('ins.' + bound + '_t') * ones_t
                t_init[mft.basic_dof_on_region(i+1)] = eval('ins.' + bound + '_t') * ones_t[mft.basic_dof_on_region(i+1)]
                dirichlet = True
            if type(eval('ins.' + bound + '_t')) is str:
                data_t = eval(eval('ins.' + bound + '_t').replace('X','x_t').replace('Y','y_t')) * ones_t
                t_init[mft.basic_dof_on_region(i + 1)] = (eval(eval('ins.' + bound + '_t').replace('X','x_t').replace('Y','y_t')) * ones_t)[mft.basic_dof_on_region(i + 1)]

                dirichlet = True

            if dirichlet:
                md.add_initialized_fem_data(bound + 'data_t', mft, [data_t])
                md.add_Dirichlet_condition_with_multipliers(mim, 't', 1, i + 1,
                                                            dataname=bound + 'data_t')

                md.set_variable('Previous2_t',t_init)
                md.set_variable('Previous_t', t_init)

            # Neumann boundaries
            else:
                if (type(eval('ins.' + bound + '_dt')) is float) or (type(eval('ins.' + bound + '_dt')) is int):
                    data_t = -eval('ins.' + bound + '_dt') * ones_mat * kappa
                if type(eval('ins.' + bound + '_dt')) is str:
                    data_t = -eval(eval('ins.' + bound + '_dt')) * kappa

                if 'axial' in ins.symmetry:
                    data_t = x_mat*data_t

                md.add_initialized_fem_data(bound + 'data_dt', mfmat, [(1+md.variable('BDFf_t'))*data_t, (1+md.variable('BDFf_t'))*data_t])
                md.add_normal_source_term_brick(mim, 't', bound + 'data_dt', i + 1)

        # add temperature or flux to free surface (default to temperature if provided)
        # fixed temperature
        if ins.free_surface:
            if (type(ins.surface_temp) is float) or (type(ins.surface_temp) is int):
                md.add_initialized_fem_data('surface_temp', mft, [ins.surface_temp * ones_t])
                md.add_nonlinear_term(mim_surf,'1e6*(t-surface_temp)*Test_t')
                #md.add_Dirichlet_condition_with_multipliers(mim_surf, 't', 2, -1, dataname='surface_temp')
            # fixed temperature flux
            elif (type(ins.surface_flux) is float) or (type(ins.surface_flux) is int):
                if 'planar' in ins.symmetry:
                    md.add_initialized_fem_data('surface_flux', mft, [ins.surface_flux * ones_t/ins.rho1/ins.cp1])
                else:
                    md.add_initialized_fem_data('surface_flux', mft, [x_t*ins.surface_flux * ones_t / ins.rho1 / ins.cp1])
                md.add_source_term_brick(mim_surf, 't', 'surface_flux', -1)
            # dynamic surface flux
            elif type(ins.surface_flux) is str:
                surface_flux_budget = ones_t * 0
                if 'radiation' in ins.surface_flux:
                    surface_flux_budget += radii_t / ins.rho1 / ins.cp1 * ins.emissivity * ins.stefan_boltzmann * (1-ins.crust_cover) * (
                        (ins.T0+273) ** 4 - (ins.T_amb+273) ** 4)
                if 'forced convection' in ins.surface_flux:
                    surface_flux_budget += ins.heat_transfer_coeff * (t_init - ins.T_amb)
                if 'axial' in ins.symmetry:
                    surface_flux_budget = x_t * surface_flux_budget
                md.add_initialized_fem_data('surface_flux', mft, [surface_flux_budget * ones_t])
                md.add_source_term_brick(mim_surf, 't', 'surface_flux', -1)

        if ins.topography & (not ins.solve_topography):
            # add temperature or flux to flow base (default to temperature if provided)
            if (type(ins.basal_temp) is float) or (type(ins.basal_temp) is int):
                md.add_initialized_fem_data('basal_temp', mft, [ins.basal_temp * ones_t])
                md.add_Dirichlet_condition_with_multipliers(mim_base, 't', 2, -1, dataname='basal_temp')
            elif (type(ins.basal_flux) is float) or (type(ins.basal_flux) is int):
                if 'planar' in ins.symmetry:
                    md.add_initialized_fem_data('basal_flux', mft, [ins.basal_flux * ones_t])
                    md.add_source_term_brick(mim_base, 't', 'basal_flux/Previous_rho/cp', -1)
                elif 'axial' in ins.symmetry:
                    md.add_initialized_fem_data('basal_flux', mft, [x_t * ins.basal_flux * ones_t])
                    md.add_source_term_brick(mim_base, 't', 'X(1)*basal_flux/Previous_rho/cp', -1)
            elif type(ins.basal_flux) is str:
                if 'conduction' in ins.basal_flux:
                    if 'planar' in ins.symmetry:
                        md.add_initialized_fem_data('basal_flux', mft_init, [ins.kappa3 * (t_init - ins.basal_temp_i)])
                        md.add_source_term_brick(mim_base, 't', 'basal_flux', -1)
                    if 'axial' in ins.symmetry:
                        md.add_initialized_fem_data('basal_flux', mft_init, [x_t*ins.kappa3 * (t_init - ins.basal_temp_i)])
                        md.add_source_term_brick(mim_base, 't', 'basal_flux', -1)

    #### Solve ####
    print('Starting solve')
    if ins.restart:
        err_u = np.zeros(int(np.ceil(ins.tf / ins.dt) + 1))
        err_p = np.zeros(int(np.ceil(ins.tf / ins.dt) + 1))
        err_ls1 = np.zeros(int(np.ceil(ins.tf / ins.dt) + 1))
        err_d = np.zeros(int(np.ceil(ins.tf / ins.dt) + 1))
        err_t = np.zeros(int(np.ceil(ins.tf / ins.dt) + 1))

        hf = h5py.File(ins.outfile + '/' + ins.outfile.split('/')[-1] + '.h5', 'r+')
        del hf['err_u']
        hf.create_dataset('err_u', data=err_u, maxshape=(None,))
        del hf['err_p']
        hf.create_dataset('err_p', data=err_p, maxshape=(None,))
        del hf['err_d']
        hf.create_dataset('err_d', data=err_d, maxshape=(None,))
        if ins.temperature:
            del hf['err_t']
            hf.create_dataset('err_t', data=err_t, maxshape=(None,))
        if ins.free_surface:
            del hf['err_ls1']
            hf.create_dataset('err_ls1', data=err_ls1, maxshape=(None,))

        u_init = 0*Previous_u
        d_init = 0*Previous_u

    else:
        # initialize pressure and velocity with incompressible, steady stokes
        if ins.steady | ins.steady_init:
            md_init.solve('max_res', ins.max_residual, 'max_iter', ins.max_iter, 'noisy')

            p_init = compute_interpolate_on(mfp,md_init.variable('p'),mfp_init)
            u_init = md_init.variable('u')

        md.set_variable('p', p_init)
        md.set_variable('Previous_p', p_init)
        md.set_variable('Previous2_p', p_init)
        md.set_variable('u', u_init)
        md.set_variable('Previous_u', u_init)
        md.set_variable('Previous2_u', u_init)
        if ins.temperature:
            md.set_variable('t',t_init)
            md.set_variable('Previous_t', t_init)
            md.set_variable('Previous2_t', t_init)

        err_u = np.zeros(int(np.ceil(ins.tf / ins.dt) + 1))
        err_p = np.zeros(int(np.ceil(ins.tf / ins.dt) + 1))
        err_ls1 = np.zeros(int(np.ceil(ins.tf / ins.dt) + 1))
        err_d = np.zeros(int(np.ceil(ins.tf / ins.dt) + 1))
        err_t = np.zeros(int(np.ceil(ins.tf / ins.dt) + 1))

        numstr = '0' * ndigits

        if ins.vtk:
            mfu.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_u_' + numstr + '.vtk', u_init)
            if ins.true_ux:
                true_u = eval(ins.true_ux.replace('X', 'x_u').replace('Y', 'y_u'))
                true_u[1::2] = eval(ins.true_uy.replace('X', 'x_u').replace('Y', 'y_u'))[1::2]
                mfu.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_utrue_' + numstr + '.vtk',
                                  true_u)
                if np.min(np.abs(true_u))>0:
                    err_u[0] = compute_L2_dist(mfu, u_init, mim, mfu, true_u) / compute_L2_norm(mfu,true_u, mim)
                else:
                    err_u[0] = compute_L2_dist(mfu, u_init, mim, mfu, true_u)
            else:
                err_u[0] = compute_L2_norm(mfu, u_init, mim)
            mfp_init.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_P_' + numstr + '.vtk', p_init)
            if ins.true_p:
                mfp.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_Ptrue_' + numstr + '.vtk',
                                  eval(ins.true_p.replace('X', 'x_p').replace('Y', 'y_p')))
                if np.min(np.abs(eval(ins.true_p.replace('X', 'x_p').replace('Y','y_p')))) > 0:
                    err_p[0] = compute_L2_dist(mfp, p_init, mim, mfp,
                                                          eval(ins.true_p.replace('X', 'x_p').replace('Y',
                                                                                                      'y_p'))) / compute_L2_norm(
                    mfp,
                    eval(ins.true_p.replace('X', 'x_p').replace('Y', 'y_p')), mim)
                else:
                    err_p[0] = compute_L2_dist(mfp, p_init, mim, mfp,
                                                          eval(ins.true_p.replace('X', 'x_p').replace('Y',
                                                                                                      'y_p')))
            else:
                err_p[0] = compute_L2_norm(mfp_init, p_init, mim)
            if ins.free_surface:
                mfls.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_Ls1_' + numstr + '.vtk',
                                   ls1.values(0))
                #mfls.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_Ls1post_' + numstr + '.vtk',
                #                   ls1.values(0))
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
                else:
                    err_ls1[0] = compute_L2_norm(mfls,ls1.values(0), mim)
                mfmat.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_rho_' + numstr + '.vtk', rho)
            mfmu.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_mu_' + numstr + '.vtk', mu)
            if ins.temperature:
                if (ins.free_surface & (~ins.solve_air)) | (ins.topography & (~ins.solve_topography)):
                    T[eval(ind_t)] = compute_interpolate_on(mft_cut, md.variable('t'), mft)
                else:
                    T = md.variable('t')
                mft_init.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_T_' + numstr + '.vtk', t_init)
                if ins.true_t:
                    mft.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_Ttrue_' + numstr + '.vtk',
                                      eval(ins.true_t.replace('ti','0')))
                    err_t[0] = compute_L2_dist(mft_init, t_init, mim, mft_init,
                                                              eval(ins.true_t.replace('ti','0'))) / compute_L2_norm(mft_init, eval(
                        ins.true_t.replace('ti','0')), mim)
                else:
                    err_t[0] = compute_L2_norm(mft_init, t_init, mim)
                if ins.solidification:
                    mfu.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_d_' + numstr + '.vtk', d_init)
                    if ins.true_dx:
                        true_d = eval(ins.true_dx.replace('X', 'x_u').replace('Y', 'y_u'))
                        true_d[1::2] = eval(ins.true_dy.replace('X', 'x_u').replace('Y', 'y_u'))[1::2]
                        mfu.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_dtrue_' + numstr + '.vtk',
                                          true_d)
                        err_d[0] = compute_L2_dist(mfu, d_init, mim, mfu, true_d)
                    else:
                        err_d[0] = compute_L2_norm(mfu, d_init, mim)
                    #mfp.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_VM_' + numstr + '.vtk',
                    #                      md.compute_isotropic_linearized_Von_Mises_pstrain('d', 'E', 'nu', mfp))
                    mfmat.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_solid_' + numstr + '.vtk', solid)
                    mft0.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_Ls2_' + numstr + '.vtk',
                                      (t_init - Tg) / Tg)
            if ins.topography:
                mfls.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_Ls3_' + numstr + '.vtk',ls3.values(0))


        if ins.free_surface:
            md.set_variable('psi',ls1.values(0))

        hf = h5py.File(ins.outfile + '/' + ins.outfile.split('/')[-1] + '.h5', 'w')
        hf.create_dataset('last_u', data=u_init)
        hf.create_dataset('last2_u', data=u_init)
        hf.create_dataset('err_u', data=err_u, maxshape=(None,))
        hf.create_dataset('last_p', data=compute_interpolate_on(mfp_init,p_init,mfp0))
        hf.create_dataset('last2_p', data=compute_interpolate_on(mfp_init,p_init,mfp0))
        hf.create_dataset('err_p', data=err_p, maxshape=(None,))
        hf.create_dataset('last_d', data=d_init)
        hf.create_dataset('last2_d', data=d_init)
        hf.create_dataset('err_d', data=err_d, maxshape=(None,))
        if ins.temperature:
            hf.create_dataset('last_t', data=t_init)
            hf.create_dataset('last2_t', data=t_init)
            hf.create_dataset('err_t', data=err_t, maxshape=(None,))
        if ins.free_surface:
            hf.create_dataset('last_ls1', data=ls1.values(0))
            hf.create_dataset('last2_ls1', data=ls1.values(0))
            hf.create_dataset('err_ls1', data=err_ls1, maxshape=(None,))
            hf.create_dataset('expected_area', data=[0])
        hf.create_dataset('last_ti', data=[0])
        hf.close()

    if ins.restart:
        tstart = last_ti
        t = np.arange(tstart, ins.tf+ins.dt, ins.dt)
    else:
        tstart = ins.dt
        t = np.append((np.arange(ins.ndt0-1)+1)*ins.dt/ins.ndt0, np.arange(tstart, ins.tf + ins.dt, ins.dt))

        Previous_u = u_init
        U = u_init
        Previous_p = p_init
        P = p_init
        Previous_d = d_init
        D = d_init
        if ins.temperature:
            Previous_t = t_init
            T = t_init

        if ins.free_surface:
            if 'planar' in ins.symmetry:
                pts, idx = mls.cut_mesh().pts_from_cvid(CVIDs=mls.cut_mesh().region(0)[0])
                ls1_interface = compute_interpolate_on(mfls, ls1.values(0), pts)
                if ins.topography:
                    ls3_interface = compute_interpolate_on(mfls, ls3.values(0), pts)
                    pts = pts[:, ((np.abs(ls1_interface) <= np.sqrt(dx ** 2 + dy ** 2) / 100) & (
                            ls3_interface >= -np.sqrt(dx ** 2 + dy ** 2) / 100)) | (
                                         (ls1_interface <= np.sqrt(dx ** 2 + dy ** 2) / 100) & (
                                         np.abs(ls3_interface) <= np.sqrt(dx ** 2 + dy ** 2) / 100))]

                    pts_edges = mesh.pts(PIDs=mesh.pid_in_regions(np.arange(4) + 1))
                    ls1_interface_edges = compute_interpolate_on(mfls, ls1.values(0), pts_edges)
                    ls3_interface_edges = compute_interpolate_on(mfls, ls3.values(0), pts_edges)
                    pts_edges = pts_edges[:, (ls1_interface_edges < 0) & (ls3_interface_edges >= 0)]
                    pts = np.append(pts, pts_edges, axis=1)
                    expected_area = alphashape.alphashape(pts.transpose(), 2 * np.sqrt(dx ** 2 + dy ** 2)).area

                else:
                    pts = pts[:, ((np.abs(ls1_interface) <= np.sqrt(dx ** 2 + dy ** 2) / 100))]
                    pts_edges = mesh.pts(PIDs=mesh.pid_in_regions(np.arange(4)+1))
                    ls1_interface_edges = compute_interpolate_on(mfls, ls1.values(0), pts_edges)
                    pts_edges = pts_edges[:, ls1_interface_edges < 0]
                    pts = np.append(pts,pts_edges,axis=1)
                    expected_area = alphashape.alphashape(pts.transpose(), 2 * np.sqrt(dx ** 2 + dy ** 2)).area
            else:
                expected_area = 1
        else:
            expected_area = 1

    # Main loop
    for ti in t:
        print('Time = %g' % ti)

        if np.abs(ti - 2*ins.dt)<(ins.dt/100):
            if ins.free_surface:
                ls1_previous2.set_values(ls1_init.values(0))
                ls1_previous.set_values(ls1.values(0))
                mlsxfem_previous.adapt()
                mlsxfem_previous2.adapt()
                mfp_previous.adapt()
                mfp_previous2.adapt()
                mfmat_previous.adapt()
                mfmat_previous2.adapt()
                if ins.t_enrich:
                    mfmu_previous.adapt()
                    mfmu_previous2.adapt()

            md.set_variable('Previous2_u', u_init)
            md.set_variable('Previous_u', md.variable('u'))

            md.set_variable('Previous2_p', p_init)
            md.set_variable('Previous_p', md.variable('p'))

            md.set_variable('Previous2_d', d_init)
            md.set_variable('Previous_d', md.variable('d'))

            if ins.temperature:
                md.set_variable('Previous2_t', t_init)
                md.set_variable('Previous_t', md.variable('t'))

            if ins.free_surface:
                md.set_variable('Previous2_psi', psi_init)
                md.set_variable('Previous_psi', ls1.values(0))
                if ins.t_enrich:
                    mfmu_previous2.adapt()
                mfmat_previous2.adapt()

            md.set_variable('Previous2_rho', rho_init)
            if ins.ndt0>1:
                md.set_variable('dt', [ins.dt])

        else:
            if ins.free_surface:
                ls1_previous2.set_values(md.variable('Previous_psi'))
                ls1_previous.set_values(md.variable('psi'))
                mlsxfem_previous.adapt()
                mlsxfem_previous2.adapt()

            md.set_variable('Previous2_u', md.variable('Previous_u'))
            md.set_variable('Previous_u', md.variable('u'))

            if ins.free_surface:
                mfp_previous2.adapt()
            md.set_variable('Previous2_p', md.variable('Previous_p'))
            if ins.free_surface:
                mfp_previous.adapt()
            md.set_variable('Previous_p', md.variable('p'))

            md.set_variable('Previous2_d', md.variable('Previous_d'))
            md.set_variable('Previous_d', md.variable('d'))

            if ins.temperature:
                md.set_variable('Previous2_t', md.variable('Previous_t'))
                md.set_variable('Previous_t', md.variable('t'))

            if ins.free_surface:
                md.set_variable('Previous2_psi', md.variable('Previous_psi'))
                md.set_variable('Previous_psi', ls1.values(0))
                md.set_variable('Previous_fext',fext)

                mfmat_previous2.adapt()
                if ins.t_enrich:
                    mfmu_previous2.adapt()
            md.set_variable('Previous2_rho', md.variable('Previous_rho'))

        if ins.free_surface:
            mfmat_previous.adapt()
            if ins.t_enrich:
                mfmu_previous.adapt()
        md.set_variable('Previous_rho',md.variable('rho'))
        md.set_variable('Previous_beta', md.variable('beta'))
        if ins.temperature:
            md.set_variable('Previous_kappa', md.variable('kappa'))

        md.set_variable('Previous_mu', md.variable('mu'))
        md.set_variable('Previous_lambda', md.variable('lambda'))
        md.set_variable('Previous_solid', md.variable('solid'))

        md.enable_variable('u')
        md.enable_variable('p')
        if ins.temperature:
            md.enable_variable('t')

        for j in np.arange(ins.n_outer_iter):
            # density
            Previous_rho = 1*rho
            rho = ones_mat * ins.rho1
            if ins.free_surface:
                ls1_mat = compute_interpolate_on(mfls,ls1.values(0),D_mat)
                rho[ls1_mat > 0] = ins.rho2
            if ins.topography:
                ls3_mat = compute_interpolate_on(mfls, ls3.values(0), D_mat)
                rho[ls3_mat < 0] = ins.rho3
            md.set_variable('rho', rho)

            if ins.compressible:
                if ins.free_surface:
                    beta = 0*ones_mat
                    beta[(ls1_mat <= 0)] = ins.beta1
                    beta[(ls1_mat > 0)] = ins.beta2
                if ins.topography:
                    beta[ls3_mat < 0] = ins.beta3
            else:
                beta = ones_mat * (1e-9) / rho
            md.set_variable('beta', beta)

            if ins.temperature:
                # thermal diffusivity
                if ins.free_surface:
                    kappa = 0*ones_mat
                    kappa[(ls1_mat <= 0)] = ins.kappa1
                    kappa[(ls1_mat > 0)] = ins.kappa2
                if ins.topography:
                    kappa[ls3_mat < 0] = ins.kappa3
                md.set_variable('kappa', kappa)

                # update viscosity field
                T_mu = compute_interpolate_on(mft,T,D_mu)
                eta = eval(ins.eta_exp.replace('exp', 'np.exp').replace('vft','ins.vft').replace('etar','ins.etar').replace('T','T_mu'))
                eta[eta > ins.max_eta] = ins.max_eta
                eta[T_mu <= ins.vftb/(np.log10(ins.max_eta)-ins.vfta) - 273 + ins.vftc] = ins.max_eta
                if ins.temperature & ins.solidification:
                    T_ls2 = compute_interpolate_on(mft0,md.variable('t'),mft)
                    ls2.set_values((T_ls2 - Tg) / Tg)
                    ls2_mu = compute_interpolate_on(mft0, ls2.values(0), D_mu)
                    ls2_mat = compute_interpolate_on(mft0, ls2.values(0), D_mat)

            else:
                eta = eval(ins.eta_exp.replace('exp', 'np.exp').replace('T', str(ins.T0)).replace('vft','ins.vft').replace('etar','ins.etar')) * ones_mu
                eta[eta > ins.max_eta] = ins.max_eta
                eta[ins.T0*ones_mu <= ins.vftb / (np.log10(ins.max_eta) - ins.vfta) - 273 + ins.vftc] = ins.max_eta

            if ins.free_surface:
                ls1_mu = compute_interpolate_on(mfls, ls1.values(0), D_mu)
                eta[(ls1_mu > 0)] = ins.eta2
            if ins.topography:
                ls3_mu = compute_interpolate_on(mfls, ls3.values(0), mfmu.basic_dof_nodes())
                eta[ls3_mu < 0] = ins.eta3
            mu = 1*eta
            lam = -2/3*eta
            solid = 0 * ones_mat
            if ins.temperature & ins.solidification:
                # elasticity
                if ins.free_surface:
                    lam[(ls1_mu <= 0) & (ls2_mu <= 0)] = lam_solid
                    mu[(ls1_mu <= 0) & (ls2_mu <= 0)] = mu_solid

                    ls1_mat = compute_interpolate_on(mfls, ls1.values(0), D_mat)

                    solid[(ls1_mat <= 0) & (ls2_mat <= 0)] = 1
                    beta[(ls1_mat <= 0) & (ls2_mat <= 0)] = 3*(1-2*ins.nu)/ins.E
                else:
                    lam[ls2_mu <= 0] = lam_solid
                    mu[ls2_mu <= 0] = mu_solid
                    solid[ls2_mat <= 0] = 1
                    beta[ls2_mat <= 0] = 3 * (1 - 2 * ins.nu) / ins.E

                if ins.free_surface:
                    solid = sciinterp.griddata(np.array([x_grid.flatten(), y_grid.flatten()]).transpose(),
                                           remove_small_objects(sciinterp.griddata(D_mat.transpose(), solid,
                                                                                   np.array([x_grid.flatten(),
                                                                                             y_grid.flatten()]).transpose(),
                                                                                   method='nearest').reshape(x_grid.shape).astype(int),
                                                                min_size=9).flatten(),
                                           D_mat.transpose(), method='nearest')

                md.set_variable('solid', solid)
                md.set_variable('beta',beta)

            md.set_variable('lambda', lam)
            md.set_variable('mu', mu)
            md.set_variable('E', ins.E*solid)
            md.set_variable('nu',ins.nu*solid)

            # update body force
            if (type(ins.f_x) is type(None)) and (type(ins.f_y) is type(None)):
                f_xi = None
                f_yi = None
            else:
                if (type(ins.f_x) is float) or (type(ins.f_x) is int):
                    f_xi = ones_mat * ins.f_x
                elif type(ins.f_x) is str:
                    P_mat = compute_interpolate_on(mfp,md.variable('p'),mfmat.basic_dof_nodes())
                    f_xi = eval(ins.f_x.replace('Y', 'y_mat').replace('X', 'x_mat')) * ones_mat
                elif type(ins.f_x) is type(None):
                    f_xi = ones_mat * 0

                if (type(ins.f_y) is float) or (type(ins.f_y) is int):
                    f_yi = ones_mat * ins.f_y
                elif type(ins.f_y) is str:
                    P_mat = compute_interpolate_on(mfp,md.variable('p'),mfmat.basic_dof_nodes())
                    f_yi = eval(ins.f_y.replace('Y', 'y_mat').replace('X', 'x_mat')) * ones_mat
                elif type(ins.f_y) is type(None):
                    f_yi = ones_mat * 0

                if ('axial' in ins.symmetry) & (~ins.steady):
                    f_xi = x_mat * f_xi
                    f_yi = x_mat * f_yi
                md.set_variable('body', [f_xi, f_yi])

            # update BCs
            vars = ['u']
            for var in vars:
                for i, bound in enumerate(bounds):
                    # Dirichlet boundaries
                    dirichlet = False
                    if (type(eval('ins.' + bound + '_' + var + 'x')) is float) or (
                            type(eval('ins.' + bound + '_' + var + 'x')) is int):
                        data_x = eval('ins.' + bound + '_' + var + 'x') * ones_p
                        dirichlet = True
                    if type(eval('ins.' + bound + '_' + var + 'x')) is str:
                        data_x = eval(eval('ins.' + bound + '_' + var + 'x').replace('X', 'x_p').replace('Y', 'y_p')) * ones_p
                        dirichlet = True
                    if type(eval('ins.' + bound + '_' + var + 'y')) is type(None):
                        data_y = 0 * ones_p

                    if (type(eval('ins.' + bound + '_' + var + 'y')) is float) or (
                            type(eval('ins.' + bound + '_' + var + 'y')) is int):
                        data_y = eval('ins.' + bound + '_' + var + 'y') * ones_p
                        dirichlet = True
                    if type(eval('ins.' + bound + '_' + var + 'y')) is str:
                        data_y = eval(eval('ins.' + bound + '_' + var + 'y').replace('X', 'x_p').replace('Y', 'y_p')) * ones_p
                        dirichlet = True
                    if type(eval('ins.' + bound + '_' + var + 'x')) is type(None):
                        data_x = 0 * ones_p

                    if dirichlet:
                        md.set_variable(bound + 'data_' + var, [data_x, data_y])

                    neumann = False
                    if (type(eval('ins.' + bound + '_d' + var + 'x')) is float) or (
                            type(eval('ins.' + bound + '_d' + var + 'x')) is int):
                        data_dx = eval('ins.' + bound + '_d' + var + 'x') * ones_u
                        neumann = True
                    if type(eval('ins.' + bound + '_d' + var + 'x')) is str:
                        data_dx = eval(eval('ins.' + bound + '_d' + var + 'x')) * ones_u
                        neumann = True
                    if type(eval('ins.' + bound + '_d' + var + 'y')) is type(None):
                        data_dy = 0 * ones_u

                    if (type(eval('ins.' + bound + '_d' + var + 'y')) is float) or (
                            type(eval('ins.' + bound + '_d' + var + 'y')) is int):
                        data_dy = eval('ins.' + bound + '_d' + var + 'y') * ones_u
                        neumann = True
                    if type(eval('ins.' + bound + '_d' + var + 'y')) is str:
                        data_dy = eval(eval('ins.' + bound + '_d' + var + 'y')) * ones_u
                        neumann = True
                    if type(eval('ins.' + bound + '_d' + var + 'x')) is type(None):
                        data_dx = 0 * ones_u

                    if 'axial' in ins.symmetry:
                        data_dx = x_u * data_dx
                        data_dy = x_u * data_dy

                    if neumann:
                        md.set_variable(bound + 'data_d' + var, [data_dx, data_dy])

            if ins.temperature:
                for i, bound in enumerate(bounds):
                    dirichlet = False
                    # Dirichlet boundaries
                    if (type(eval('ins.' + bound + '_t')) is float) or (type(eval('ins.' + bound + '_t')) is int):
                        data_t = eval('ins.' + bound + '_t') * ones_t
                        dirichlet = True
                    if type(eval('ins.' + bound + '_t')) is str:
                        data_t = eval(eval('ins.' + bound + '_t').replace('X', 'x_t').replace('Y', 'y_t')) * ones_t
                        dirichlet = True

                    if dirichlet:
                        md.set_variable(bound + 'data_t', [data_t])

                    # Neumann boundaries
                    else:
                        if (type(eval('ins.' + bound + '_dt')) is float) or (type(eval('ins.' + bound + '_dt')) is int):
                            data_t = -eval('ins.' + bound + '_dt') * ones_mat * kappa
                        if type(eval('ins.' + bound + '_dt')) is str:
                            data_t = -eval(eval('ins.' + bound + '_dt')) * kappa
                        if 'axial' in ins.symmetry:
                            data_t = x_mat * data_t
                        md.set_variable(bound + 'data_dt', [data_t, data_t])

            # update surface flux
            if ins.temperature:
                if ins.free_surface | ins.topography:
                    mls_cut = mls.cut_mesh()
                    radii = np.zeros(mls_cut.nbpts())
                    D_cut = mls_cut.pts()
                    for pid in mls_cut.pid():
                        radii[pid] = np.min(mls_cut.convex_radius(mls_cut.cvid_from_pid(pid, share=True)))
                    if ins.temperature:
                        radii_t = sciinterp.griddata(D_cut.transpose(), radii, D_t.transpose())

                else:
                    radii = np.mean(mesh.convex_radius()) * ones_ls
                    if ins.temperature:
                        radii_t = np.mean(mesh.convex_radius()) * ones_t

                if 'axial' in ins.symmetry:
                    radii_t = radii_t * 2 * np.pi * (x_t + dx / 2)

                if ins.free_surface:
                    if type(ins.surface_flux) is str:
                        surface_flux_budget = ones_t * 0
                        if 'radiation' in ins.surface_flux:
                            surface_flux_budget += radii_t / ins.rho1 / ins.cp1 * ins.emissivity * ins.stefan_boltzmann * (1-ins.crust_cover) * (
                                (ins.T0+273) ** 4 - (ins.T_amb+273) ** 4)
                        if 'forced convection' in ins.surface_flux:
                            surface_flux_budget += ins.heat_transfer_coeff * (T - ins.T_amb)
                        if 'axial' in ins.symmetry:
                            surface_flux_budget = x_t*surface_flux_budget
                        md.set_variable('surface_flux', [surface_flux_budget * ones_t])

                if ins.topography & (not ins.solve_topography):
                    if type(ins.basal_flux) is str:
                        if 'conduction' in ins.basal_flux:
                            if 'planar' in ins.symmetry:
                                md.set_variable('basal_flux', [ins.kappa1 * (T - ins.basal_temp_i) / radii_t * ones_t])
                            if 'axial' in ins.symmetry:
                                md.set_variable('basal_flux', [x_t*ins.kappa1 * (T - ins.basal_temp_i) / radii_t * ones_t])

            md.enable_variable('u')
            md.enable_variable('p')
            if ins.temperature:
                md.enable_variable('t')
            md.solve('max_res', ins.max_residual, 'max_iter', ins.max_iter, 'noisy')

            if j == 0:
                Previous_u = U
            U = ones_u * 0
            U[eval(ind_u)] = md.variable('u')

            if j == 0:
                Previous_p = P
            P = ones_p * 0
            P[eval(ind_p)] = md.variable('p')

            if ins.temperature:
                if j == 0:
                    Previous_t = T
                T[eval(ind_t)] = md.variable('t')

            md.disable_variable('u')
            md.disable_variable('p')
            if ins.temperature:
                md.disable_variable('t')

            if ins.temperature:
                if ins.solidification:

                    md.enable_variable('d')
                    md.solve('max_res', ins.max_residual*10**-5, 'max_iter', ins.max_iter, 'noisy')

                    if j == 0:
                        Previous_d = D
                    D = ones_u * 0
                    D[eval(ind_u)] = md.variable('d')
                    D[sciinterp.griddata(D_mat.transpose(), solid, D_u.transpose())<(1-np.sqrt(dx**2 + dy**2)/4)] = 0


                    if (np.min(ls2.values(0)) < 0) & (np.max(ls2.values(0)) > 0):
                        Ls2_ext = sciinterp.griddata(D_t.transpose(), ls2.values(0),
                                                     np.array([x_grid.flatten(), y_grid.flatten()]).transpose(),
                                                     method='linear')
                        d_ls2 = skfmm.distance(Ls2_ext.reshape(x_grid.shape), dx=[dy/ins.u_k, dx/ins.u_k])
                    else:
                        d_ls2 = np.ones(x_grid.shape) * (-2 * np.max([ins.L_x, ins.L_y]))

                    if ins.free_surface:
                        Ls1_ext = sciinterp.griddata(D_ls.transpose(), ls1.values(0),
                                                     np.array([x_grid.flatten(), y_grid.flatten()]).transpose(),
                                                     method='linear')
                        d_ls1 = skfmm.distance(Ls1_ext.reshape(x_grid.shape), dx=[dy/ins.u_k, dx/ins.u_k])
                        d = np.maximum(np.array(d_ls1), np.array(d_ls2))
                    else:
                        d = d_ls2

                    if (np.min(d) < 0) & (np.max(d) > 0):
                        ux_grid = sciinterp.griddata(D_u[:, ::2].transpose(), U[::2],
                                                     np.array([x_grid.flatten(), y_grid.flatten()]).transpose(),
                                                     method='nearest')
                        d, D_ext_x = skfmm.extension_velocities(d,
                                                                sciinterp.griddata(D_u[:, ::2].transpose(), U[::2],
                                                     np.array([x_grid.flatten(), y_grid.flatten()]).transpose(),
                                                     method='nearest').flatten().reshape(x_grid.shape),
                                                                  dx=[dy/ins.u_k, dx/ins.u_k])
                        d, D_ext_y = skfmm.extension_velocities(d,
                                                                sciinterp.griddata(D_u[:, ::2].transpose(), D[::2],
                                                     np.array([x_grid.flatten(), y_grid.flatten()]).transpose(),
                                                     method='nearest').flatten().reshape(
                                                                    x_grid.shape),
                                                                dx=[dy / ins.u_k, dx / ins.u_k])
                        D_ext_x = sciinterp.griddata(np.array([x_grid.flatten(), y_grid.flatten()]).transpose(), D_ext_x.flatten(),
                                                   D_u.transpose(),
                                                   method='nearest').flatten()
                        D_ext_y = sciinterp.griddata(np.array([x_grid.flatten(), y_grid.flatten()]).transpose(), D_ext_y.flatten(),
                                                     D_u.transpose(),
                                                     method='nearest').flatten()
                        D_ext = D_ext_x
                        D_ext[1::2] = D_ext_y[1::2]

                    else:
                        D_ext = 0*ones_u

                    md.disable_variable('d')

                else:
                    Previous_d = 0*u_init
                    D = 0*u_init

                if ins.topography:
                    Ls3_u = compute_interpolate_on(mfls, ls3.values(0), mfu.basic_dof_nodes())
                    D[(Ls3_u < 0)] = 0

            if (((int(ti/ins.dt) % ins.noutput == 0) & (ti>=ins.dt)) or (np.abs(ti - ins.tf) < ins.dt)) & (j == (ins.n_outer_iter-1)):
                numstr = str(round(ti * 10 ** (ndecimal))).split('.')[0].zfill(ndigits)
                # print('Average temperature %g' % np.mean(T))
                if ins.vtk:
                    mfu.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_u_' + numstr + '.vtk', U)
                    if ins.true_ux:
                        true_u = eval(ins.true_ux.replace('X', 'x_u').replace('Y', 'y_u'))
                        true_u[1::2] = eval(ins.true_uy.replace('X', 'x_u').replace('Y', 'y_u'))[1::2]
                        mfu.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_utrue_' + numstr + '.vtk',
                                          true_u)

                        if np.min(np.abs(true_u)) > 0:
                            err_u[int(ti / ins.dt)] = compute_L2_dist(mfu, U, mim, mfu, true_u) / compute_L2_norm(mfu,
                                true_u, mim)
                        else:
                            err_u[int(ti / ins.dt)] = compute_L2_dist(mfu, U, mim, mfu, true_u)
                    else:
                        err_u[int(ti / ins.dt)] = compute_L2_norm(mfu,U, mim)
                    mfp.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_P_' + numstr + '.vtk', md.variable('p'))
                    if ins.true_p:
                        mfp.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_Ptrue_' + numstr + '.vtk',
                                           eval(ins.true_p.replace('X','x_p').replace('Y','y_p')))
                        if np.min(np.abs(eval(ins.true_p.replace('X','x_p').replace('Y','y_p')))) > 0:
                            err_p[int(ti / ins.dt)] = compute_L2_dist(mfp, P, mim, mfp,
                                                                  eval(ins.true_p.replace('X','x_p').replace('Y','y_p'))) / compute_L2_norm(mfp,
                                                                  eval(ins.true_p.replace('X','x_p').replace('Y','y_p')),mim)
                        else:
                            err_p[int(ti / ins.dt)] = compute_L2_dist(mfp, P, mim, mfp,
                                                                      eval(ins.true_p.replace('X', 'x_p').replace('Y',
                                                                                                                  'y_p')))
                    else:
                        err_p[int(ti / ins.dt)] = compute_L2_norm(mfp, P, mim)
                    if ins.free_surface:
                        mfls.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_Ls1_' + numstr + '.vtk',
                                           md.variable('psi'))
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
                        else:
                            err_ls1[int(ti / ins.dt)] = compute_L2_norm(mfls, ls1.values(0), mim)
                            #print(pts[1,:])
                            #print(eval('-'+ins.true_ls1.replace('X', 'pts[0,:]').replace('Y', '0')))
                            # err_ls1[int(ti/ins.dt)] = compute_L2_dist(mfls, md.variable('Previous_psi'), mim_all, mfls,
                            #                                          eval(ins.true_ls1))/compute_L2_norm(mfls,eval(ins.true_ls1),mim_all)
                        mfmat.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_rho_' + numstr + '.vtk', rho)
                    mfmu.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_mu_' + numstr + '.vtk', mu)
                    if ins.temperature:
                        mft.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_T_' + numstr + '.vtk',
                                          md.variable('t'))
                        if ins.true_t:
                            mft.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_Ttrue_' + numstr + '.vtk',
                                              eval(ins.true_t))
                            err_t[int(ti / ins.dt)] = compute_L2_dist(mft, T, mim, mft,
                                                                      eval(ins.true_t)) / compute_L2_norm(mft, eval(
                                ins.true_t), mim)
                        else:
                            err_t[int(ti / ins.dt)] = compute_L2_norm(mft, T, mim)

                        if ins.solidification:
                            mfu.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_d_' + numstr + '.vtk', D)
                            if ins.true_dx:
                                true_d = eval(ins.true_dx.replace('X', 'x_u').replace('Y', 'y_u'))
                                true_d[1::2] = eval(ins.true_dy.replace('X', 'x_u').replace('Y', 'y_u'))[1::2]
                                mfu.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_dtrue_' + numstr + '.vtk',
                                                  true_d)
                                err_d[int(ti / ins.dt)] = compute_L2_dist(mfu, D, mim, mfu, true_d)
                            else:
                                err_d[int(ti / ins.dt)] = compute_L2_norm(mfu, D, mim)
                            #mfp.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_VM_' + numstr + '.vtk',
                            #                      md.compute_isotropic_linearized_Von_Mises_pstrain('d', 'E', 'nu', mfp))
                            mfmat.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_solid_' + numstr + '.vtk',solid)
                            T_ls2 = compute_interpolate_on(mft0, md.variable('t'), mft)
                            mft0.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_Ls2_' + numstr + '.vtk',
                                              (T_ls2 - Tg) / Tg)
                    if ins.topography:
                        mfls.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_Ls3_' + numstr + '.vtk',
                                           ls3.values(0))

                hf = h5py.File(ins.outfile + '/' + ins.outfile.split('/')[-1] + '.h5', 'a')
                hf['last_u'][:] = U
                hf['last2_u'][:] = Previous_u
                hf['err_u'][:] = err_u
                hf['last_p'][:] = compute_interpolate_on(mfp,P,mfp0)
                hf['last2_p'][:] = compute_interpolate_on(mfp_previous,md.variable('Previous_p'),mfp0)
                hf['err_p'][:] = err_p
                hf['last_d'][:] = D
                hf['last2_d'][:] = Previous_d
                hf['err_d'][:] = err_d
                if ins.temperature:
                    hf['last_t'][:] = T
                    hf['last2_t'][:] = Previous_t
                    hf['err_t'][:] = err_t
                if ins.free_surface:
                    hf['last_ls1'][:] = ls1.values(0)
                    hf['last2_ls1'][:] = ls1_previous.values(0)
                    hf['err_ls1'][:] = err_ls1
                    hf['expected_area'][:] = [expected_area]
                hf['last_ti'][:] = [ti]
                hf.close()

            if ins.free_surface:
                md.enable_variable('psis')
                md.solve('max_res', ins.max_residual, 'max_iter', ins.max_iter, 'noisy')
                md.disable_variable('psis')

                # construct extension velocities
                # speed on the interface is U.Grad(Psi)

                Psi_grid = sciinterp.griddata(D_ls.transpose(), md.variable('psis'),
                                              np.array([x_grid.flatten(), y_grid.flatten()]).transpose(),
                                              method='cubic').reshape(x_grid.shape)

                dx_Psi_grid, dy_Psi_grid, curvature, mag_grad_Psi_grid = compute_curvature(Psi_grid, dx/ins.u_k, dy/ins.u_k)
                md.set_variable('curvature', sciinterp.griddata(np.array([x_grid.flatten(), y_grid.flatten()]).transpose(),
                                                               curvature.flatten(),
                                                               D_curve.transpose(),
                                                               method='nearest').flatten())
                if ins.topography:
                    Ls3_grid = sciinterp.griddata(D_ls.transpose(), ls3.values(0),
                                           np.array([x_grid.flatten(), y_grid.flatten()]).transpose(), method='linear').reshape(x_grid.shape)
                else:
                        Ls3_grid = np.ones_like(x_grid) + np.sqrt(dx ** 2 + dy ** 2)

                eta_grid = sciinterp.griddata(D_mu.transpose(), eta,
                                              np.array([x_grid.flatten(), y_grid.flatten()]).transpose(),
                                              method='nearest').reshape(x_grid.shape)
                mean_eta_surf = np.mean(eta_grid[(np.abs(Psi_grid)<np.sqrt(dx**2 + dy**2))])
                mean_curvature = np.mean(curvature[(np.abs(Psi_grid)<np.sqrt(dx**2 + dy**2))])

                ux_grid = sciinterp.griddata(D_u[:,::2].transpose(), U[::2],
                                             np.array([x_grid.flatten(), y_grid.flatten()]).transpose(),
                                             method='nearest')
                uy_grid = sciinterp.griddata(D_u[:,1::2].transpose(), U[1::2],
                                             np.array([x_grid.flatten(), y_grid.flatten()]).transpose(),
                                             method='nearest')

                #F_grid = ux_grid * (dx_Psi_grid/mag_grad_Psi_grid).flatten() + uy_grid * (dy_Psi_grid/mag_grad_Psi_grid).flatten()
                relax_speed = (ins.epsilon_psi*np.mean(np.sqrt(f_xi**2 + f_yi**2)+1)/mean_eta_surf * (curvature - mean_curvature)).flatten()

                relax_max = np.sqrt((dx/ins.u_k)**2 + (dy/ins.u_k)**2)/(2*md.variable('dt'))
                relax_speed[relax_speed > relax_max] = relax_max - (np.sqrt(ux_grid**2 + uy_grid**2)/(2*md.variable('dt')))[relax_speed > relax_max]
                relax_speed[relax_speed < -relax_max] = -relax_max + (np.sqrt(ux_grid**2 + uy_grid**2)/(2*md.variable('dt')))[relax_speed < -relax_max]
                relax_speed = relax_speed.reshape(x_grid.shape)

                if ins.topography:
                    relax_speed[Ls3_grid<np.sqrt(dx**2+dy**2)/2] = 0

                mask_grid = (Psi_grid > 0) | (sciinterp.griddata(D_ls.transpose(), ls1.values(0),
                                              np.array([x_grid.flatten(), y_grid.flatten()]).transpose(),
                                              method='linear').reshape(x_grid.shape)>0)
                d_ls1, Fx_ext_grid = skfmm.extension_velocities(sciinterp.griddata(D_ls.transpose(), md.variable('psi'),
                                                                                  np.array([x_grid.flatten(),
                                                                                            y_grid.flatten()]).transpose(),
                                                                                  method='linear').reshape(x_grid.shape),
                                                               ux_grid.reshape(x_grid.shape) - relax_speed*dx_Psi_grid/mag_grad_Psi_grid,
                                                               dx=[dy/ins.u_k, dx/ins.u_k], ext_mask=mask_grid)
                d_ls1, Fy_ext_grid = skfmm.extension_velocities(sciinterp.griddata(D_ls.transpose(), md.variable('psi'),
                                                                                   np.array([x_grid.flatten(),
                                                                                             y_grid.flatten()]).transpose(),
                                                                                   method='linear').reshape(x_grid.shape),
                                                                uy_grid.reshape(x_grid.shape) - relax_speed * dy_Psi_grid / mag_grad_Psi_grid,
                                                                dx=[dy/ins.u_k, dx/ins.u_k], ext_mask=mask_grid)

                Fx_ext = sciinterp.griddata(np.array([x_grid.flatten(), y_grid.flatten()]).transpose(), Fx_ext_grid.flatten(),
                                           D_u[:,::2].transpose(),
                                           method='nearest').flatten()
                Fy_ext = sciinterp.griddata(np.array([x_grid.flatten(), y_grid.flatten()]).transpose(),
                                            Fy_ext_grid.flatten(),
                                            D_u[:,1::2].transpose(),
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

                fext = 0 * ones_u
                fext[::2] = Fx_ext
                fext[1::2] = Fy_ext

                md.set_variable('fext', fext)
                if (ti<=2*ins.dt/ins.ndt0) & (j == (ins.n_outer_iter-1)):
                    fext_init = fext.copy()
                    md.set_variable('Previous_fext',fext)

                #if (((int(ti/ins.dt) % ins.noutput == 0) & (ti>=ins.dt)) or (np.abs(ti - ins.tf) < ins.dt)) & (j == (ins.n_outer_iter-1)):
                    #numstr = str(ti * 10 ** ndecimal).split('.')[0].zfill(ndigits)
                    # print('Average temperature %g' % np.mean(T))
                    #if ins.vtk:
                        #if ins.free_surface:
                            #mfu.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_Fext_' + numstr + '.vtk', fext)
                            #mfcurve.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_psigrid_' + numstr + '.vtk',
                            #                   sciinterp.griddata(
                            #                       np.array([x_grid.flatten(), y_grid.flatten()]).transpose(),
                            #                       Psi_grid.flatten(),
                            #                       D_curve.transpose(),
                            #                       method='nearest'))
                            #mfcurve.export_to_vtk(
                            #    outfile + '/' + ins.outfile.split('/')[-1] + '_dxpsigrid_' + numstr + '.vtk',
                            #    sciinterp.griddata(
                            #        np.array([x_grid.flatten(), y_grid.flatten()]).transpose(),
                            #        dx_Psi_grid.flatten(),
                            #        D_curve.transpose(),
                            #        method='nearest'))
                            #mfcurve.export_to_vtk(
                            #    outfile + '/' + ins.outfile.split('/')[-1] + '_dypsigrid_' + numstr + '.vtk',
                            #    sciinterp.griddata(
                            #        np.array([x_grid.flatten(), y_grid.flatten()]).transpose(),
                            #        dy_Psi_grid.flatten(),
                            #        D_curve.transpose(),
                            #        method='nearest'))
                            #mfls.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_psis_' + numstr + '.vtk', md.variable('psis'))
                            #mfcurve.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_curvature_' + numstr + '.vtk',
                            #                   sciinterp.griddata(
                            #                       np.array([x_grid.flatten(), y_grid.flatten()]).transpose(),
                            #                       curvature.flatten(),
                            #                       D_curve.transpose(),
                            #                       method='nearest'))
                            #mfcurve.export_to_vtk(
                            #    outfile + '/' + ins.outfile.split('/')[-1] + '_relax_' + numstr + '.vtk',
                            #    sciinterp.griddata(
                            #        np.array([x_grid.flatten(), y_grid.flatten()]).transpose(),
                            #        relax_speed.flatten(),
                            #        D_curve.transpose(),
                            #        method='nearest'))

                md.enable_variable('psi')
                md.solve('max_res', ins.max_residual, 'max_iter', ins.max_iter, 'noisy')
                md.disable_variable('psi')

                if ins.influx & ins.fix_ls:
                    psi = md.variable('psi')
                    psi[mfls.basic_dof_on_region(5)] = ls1_init.values(0)[mfls.basic_dof_on_region(5)]
                    psi[mfls.basic_dof_on_region(bounds.index(ins.fix_ls_bound))] = ls1_init.values(0)[mfls.basic_dof_on_region(bounds.index(ins.fix_ls_bound))]
                    md.set_variable('psi',psi)

                ls1.set_values(md.variable('psi'))
                #if ins.topography:
                #    ls1xfem.set_values(md.variable('psi'),'-(' + ins.ls3p + ')')
                itr = 0
                #if ins.influx:
                #    d_area = ins.dt * ((ins.influx_right - ins.influx_left) * eval(ins.influx_uy) + (
                #                ins.influx_top - ins.influx_bottom) * eval(ins.influx_ux))
                #    expected_area += d_area
                while (np.abs(area - expected_area) / expected_area > 1e-4) & (itr < 10) & (np.min(np.abs(ls1.values(0)[edges_ls]))>=np.sqrt(dx**2 + dy**2)/2):
                    if itr > 0:
                        if ins.topography:
                            ls1.set_values(ls1.values(0) + (area - expected_area) / pts.shape[1] / np.sqrt(dx ** 2 + dy ** 2)*(ls3.values(0)>np.sqrt(dx**2 + dy**2)/2)/2)
                            ls1xfem.set_values(ls1.values(0),'-(' + ins.ls3p + ')')
                        else:
                            ls1.set_values(ls1.values(0) + (area - expected_area) / pts.shape[1] / np.sqrt(dx ** 2 + dy ** 2) / 2)
                    itr += 1
                    mls.adapt()
                    mlsxfem.adapt()
                    pts, idx = mls.cut_mesh().pts_from_cvid(CVIDs=mls.cut_mesh().region(0)[0])
                    ls1_interface = compute_interpolate_on(mfls, ls1.values(0), pts)
                    if ins.topography:
                        ls3_interface = compute_interpolate_on(mfls, ls3.values(0), pts)
                        pts = pts[:, ((np.abs(ls1_interface) <= np.sqrt(dx ** 2 + dy ** 2) / 100) & (
                            ls3_interface >= -np.sqrt(dx ** 2 + dy ** 2) / 100)) | (
                                         (ls1_interface <= np.sqrt(dx ** 2 + dy ** 2) / 100) & (
                                         np.abs(ls3_interface) <= np.sqrt(dx ** 2 + dy ** 2) / 100))]
                        pts_edges = mesh.pts(PIDs=mesh.pid_in_regions(np.arange(4) + 1))
                        ls1_interface_edges = compute_interpolate_on(mfls, ls1.values(0), pts_edges)
                        ls3_interface_edges = compute_interpolate_on(mfls, ls3.values(0), pts_edges)
                        pts_edges = pts_edges[:, (ls1_interface_edges<0)&(ls3_interface_edges>=0)]
                        pts = np.append(pts, pts_edges, axis=1)
                        expected_area = alphashape.alphashape(pts.transpose(), 2 * np.sqrt(dx ** 2 + dy ** 2)).area
                    else:
                        pts = pts[:, ((np.abs(ls1_interface) <= np.sqrt(dx ** 2 + dy ** 2) / 100))]
                        pts_edges = mesh.pts(PIDs=mesh.pid_in_regions(np.arange(4) + 1))
                        ls1_interface_edges = compute_interpolate_on(mfls, ls1.values(0), pts_edges)
                        pts_edges = pts_edges[:, ls1_interface_edges < 0]
                        pts = np.append(pts, pts_edges, axis=1)
                        expected_area = alphashape.alphashape(pts.transpose(), 2 * np.sqrt(dx ** 2 + dy ** 2)).area
                    area = alphashape.alphashape(pts.transpose(), 2 * np.sqrt(dx ** 2 + dy ** 2)).area
                    print((area - expected_area) / expected_area)

                #if (((round(ti / ins.dt) % ins.noutput == 0) & (ti >= ins.dt)) or (np.abs(ti - ins.tf) < ins.dt)) & (j == (ins.n_outer_iter-1)):
                    #numstr = str(round(ti * 10 ** ndecimal)).split('.')[0].zfill(ndigits)
                    # print('Average temperature %g' % np.mean(T))
                    #if ins.vtk:
                        #mfls.export_to_vtk(outfile + '/' + ins.outfile.split('/')[-1] + '_Ls1post_' + numstr + '.vtk',
                        #                   md.variable('psi'))
                mls.adapt()
                mlsxfem.adapt()
                if ins.temperature & ins.solidification:
                    mls2.adapt()
                mfp.adapt()
                D_p = mfp.basic_dof_nodes()
                ones_p = np.ones(D_p.shape[1])
                x_p = D_p[0, :]
                y_p = D_p[1, :]
                #if ins.temperature:
                #    mft.adapt()
                mfmat.adapt()
                D_mat = mfmat.basic_dof_nodes()
                ones_mat = np.ones(D_mat.shape[1])
                x_mat = D_mat[0, :]
                y_mat = D_mat[1, :]

                if ins.t_enrich:
                    mfmu.adapt()
                    D_mu = mfmu.basic_dof_nodes()
                    ones_mu = np.ones(D_mu.shape[1])
                    x_mu = D_mu[0, :]
                    y_mu = D_mu[1, :]

                mim.adapt()
                mim_surf.adapt()
                mim_fluidbound.adapt()
                if ins.topography:
                    mim_base.adapt()
                    mim_topo.adapt()
                mim_all.adapt()
                mim_integ.adapt()

                #mfp_cut.set_partial(eval(ind_p))
                #mfu_cut.set_partial(eval(ind_u))
                #    mft_cut.set_partial(eval(ind_t))
                if ins.t_enrich & ins.free_surface:
                    mft_cutoff_previous2.set_enriched_dofs(enriched_dofst_previous)
                    mft_cutoff_previous.set_enriched_dofs(enriched_dofst)
                    enriched_dofst_previous = enriched_dofst
                    enriched_dofs1 = (np.nonzero(
                        np.abs(compute_interpolate_on(mfls, ls1.values(0), DOFpts_t)) < (np.sqrt(dx ** 2 + dy ** 2)))[
                                          0] + 1)
                    if ins.topography:
                        enriched_dofs3 = (np.nonzero(
                        np.abs(compute_interpolate_on(mfls, ls3.values(0), DOFpts_t)) < (np.sqrt(dx ** 2 + dy ** 2)))[
                                          0] + 1)
                        enriched_dofst = np.append(enriched_dofs1, enriched_dofs3)
                    else:
                        enriched_dofst= enriched_dofs1
                    enriched_dofst = enriched_dofst[enriched_dofst > 1]
                    mft_cutoff.set_enriched_dofs(enriched_dofst)

                mfp_cutoff_previous2.set_enriched_dofs(enriched_dofsp_previous)
                mfp_cutoff_previous.set_enriched_dofs(enriched_dofsp)
                enriched_dofsp_previous = enriched_dofsp
                enriched_dofs1 = (np.nonzero(
                    np.abs(compute_interpolate_on(mfls, ls1.values(0), DOFpts_p)) < (np.sqrt(dx ** 2 + dy ** 2)))[0] + 1)
                if ins.topography:
                    enriched_dofs3 = (np.nonzero(
                        np.abs(compute_interpolate_on(mfls, ls3.values(0), DOFpts_p)) < (np.sqrt(dx ** 2 + dy ** 2)))[0] + 1)
                    enriched_dofsp = np.append(enriched_dofs1, enriched_dofs3)
                else:
                    enriched_dofsp = enriched_dofs1
                enriched_dofsp = enriched_dofsp[enriched_dofsp > 1]
                mfp_cutoff.set_enriched_dofs(enriched_dofsp)

                mfmat_cutoff_previous2.set_enriched_dofs(enriched_dofsmat_previous)
                mfmat_cutoff_previous.set_enriched_dofs(enriched_dofsmat)
                enriched_dofsmat_previous = enriched_dofsmat
                enriched_dofs1 = np.nonzero(
                    np.abs(compute_interpolate_on(mfls, ls1.values(0), DOFpts_mat)) < (np.sqrt(dx ** 2 + dy ** 2)))[0] + 1
                if ins.topography:
                    enriched_dofs3 = np.nonzero(
                        np.abs(compute_interpolate_on(mfls, ls3.values(0), DOFpts_mat)) < (np.sqrt(dx ** 2 + dy ** 2)))[0] + 1
                    enriched_dofsmat = np.append(enriched_dofs1, enriched_dofs3)
                else:
                    enriched_dofsmat = enriched_dofs1
                enriched_dofsmat = enriched_dofsmat[enriched_dofsmat > 1]
                mfmat_cutoff.set_enriched_dofs(enriched_dofsmat)

                if ins.temperature & ins.solidification:
                    Previous_Ls_u = compute_interpolate_on(mfls, md.variable('Previous_psi'), mfu.basic_dof_nodes())
                    Ls1_u = compute_interpolate_on(mfls, ls1.values(0), mfu.basic_dof_nodes())
                    D[(Previous_Ls_u >= 0) & (Ls1_u <= 0)] = D_ext[(Previous_Ls_u >= 0) & (Ls1_u <= 0)]

            if ins.temperature & ins.solidification:
                T_ls2 = compute_interpolate_on(mft0, md.variable('t'), mft)
                ls2.set_values((T_ls2 - Tg) / Tg)
                ls2_mu = compute_interpolate_on(mft0, ls2.values(0), mfmu)
                ls2_mat = compute_interpolate_on(mft0,ls2.values(0),mfmat)

        # update BDF coefficients
        md.set_variable('BDF0', [3 / 2])
        md.set_variable('BDF1', [-2])
        md.set_variable('BDF2', [1 / 2])
        md.set_variable('BDFf', [0])
        md.set_variable('BDFf_t', [0])

    #if ins.free_surface | ins.topography:
    #    mfp_cut.set_partial(eval(ind_p))
    #    mfu_cut.set_partial(eval(ind_u))
    #    if ins.temperature:
    #        mft_cut.set_partial(eval(ind_t))

    # Visualize results
    # %%
    if ins.plots:

        U = md.variable('u')
        P = md.variable('p')
        if ins.temperature:
            T = md.variable('t')
        else:
            T = None

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
        if ins.temperature:
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

            if ins.temperature & ins.solidification:
                axi[0].tricontour(x_t, y_t, ls2.values(0), levels=[0], colors='k',
                                  linestyles='--')
        ax[0, 0].set_title('U', fontsize=18)
        ax[0, 1].set_title('V', fontsize=18)
        ax[1, 0].set_title('P', fontsize=18)

        fig2.savefig(outfile + '/' + ins.outfile.split('/')[-1] + '_solution.pdf')
        plt.close(fig2)