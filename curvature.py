import numpy as np
import skfmm
import alphashape

def compute_curvature(Psi_grid, dx, dy):
    dx_Psi_grid = np.zeros_like(Psi_grid)
    dx_Psi_grid[:, 1:-1] = (Psi_grid[:, 2:] - Psi_grid[:, :-2]) / (2 * dx)
    dx_Psi_grid[:, 0] = (Psi_grid[:, 1] - Psi_grid[:, 0]) / dx
    dx_Psi_grid[:, -1] = (Psi_grid[:, -1] - Psi_grid[:, -2]) / dx

    dy_Psi_grid = np.zeros_like(Psi_grid)
    dy_Psi_grid[1:-1, :] = (Psi_grid[2:, :] - Psi_grid[:-2, :]) / (2 * dy)
    dy_Psi_grid[0, :] = (Psi_grid[1, :] - Psi_grid[0, :]) / dy
    dy_Psi_grid[-1, :] = (Psi_grid[-1, :] - Psi_grid[-2, :]) / dy

    mag_grad_Psi_grid = np.sqrt(dx_Psi_grid ** 2 + dy_Psi_grid ** 2)
    mag_grad_Psi_grid[mag_grad_Psi_grid < 1e-14] = 1e-14

    dxx_Psi_grid = np.zeros_like(Psi_grid)
    dxx_Psi_grid[:, 1:-1] = (Psi_grid[:, 2:] - 2 * Psi_grid[:, 1:-1] + Psi_grid[:, :-2]) / (dx ** 2)
    dxx_Psi_grid[:, 0] = dxx_Psi_grid[:, 1]
    dxx_Psi_grid[:, -1] = dxx_Psi_grid[:, -2]

    dyy_Psi_grid = np.zeros_like(Psi_grid)
    dyy_Psi_grid[1:-1, :] = (Psi_grid[2:, :] - 2 * Psi_grid[1:-1, :] + Psi_grid[:-2, :]) / (dy ** 2)
    dyy_Psi_grid[0, :] = dyy_Psi_grid[1, :]
    dyy_Psi_grid[-1, :] = dyy_Psi_grid[-2, :]

    dxy_Psi_grid = np.zeros_like(Psi_grid)
    dxy_Psi_grid[1:-1, 1:-1] = (Psi_grid[2:, 2:] - Psi_grid[2:, :-2] - Psi_grid[:-2, 2:] + Psi_grid[:-2, :-2]) / (
                4 * dx * dy)
    dxy_Psi_grid[0, 1:-1] = (Psi_grid[1, 2:] - Psi_grid[1, :-2] - Psi_grid[0, 2:] + Psi_grid[0, :-2]) / (2 * dx * dy)
    dxy_Psi_grid[-1, 1:-1] = (Psi_grid[-1, 2:] - Psi_grid[-1, :-2] - Psi_grid[-2, 2:] + Psi_grid[-2, :-2]) / (
                2 * dx * dy)
    dxy_Psi_grid[1:-1, 0] = (Psi_grid[2:, 1] - Psi_grid[2:, 1] - Psi_grid[:-2, 0] + Psi_grid[:-2, 0]) / (2 * dx * dy)
    dxy_Psi_grid[1:-1, -1] = (Psi_grid[2:, -1] - Psi_grid[2:, -1] - Psi_grid[:-2, -2] + Psi_grid[:-2, -2]) / (
                2 * dx * dy)
    dxy_Psi_grid[0, 0] = (Psi_grid[1, 1] - Psi_grid[1, 0] - Psi_grid[0, 1] + Psi_grid[0, 0]) / (dx * dy)
    dxy_Psi_grid[0, -1] = (Psi_grid[1, -1] - Psi_grid[1, -2] - Psi_grid[0, -1] + Psi_grid[0, -2]) / (dx * dy)
    dxy_Psi_grid[-1, 0] = (Psi_grid[-1, 1] - Psi_grid[-1, 0] - Psi_grid[-2, 1] + Psi_grid[-2, 0]) / (dx * dy)
    dxy_Psi_grid[-1, -1] = (Psi_grid[-1, -1] - Psi_grid[-1, -2] - Psi_grid[-2, -1] + Psi_grid[-2, -2]) / (dx * dy)

    curvature = (dxx_Psi_grid * (dy_Psi_grid ** 2) - 2 * (dx_Psi_grid * dy_Psi_grid * dxy_Psi_grid) + dyy_Psi_grid * (
        dx_Psi_grid) ** 2) / (mag_grad_Psi_grid ** 3)
    return dx_Psi_grid, dy_Psi_grid, curvature, mag_grad_Psi_grid

def compute_mean_curvature(Psi_grid,Ls3_grid,curvature,eta_grid,dx,dy):
    curvature_ext = skfmm.extension_velocities(Psi_grid,
                                            curvature,
                                            dx=[dy, dx])
    weights = np.zeros_like(Psi_grid)
    weights[(np.abs(Psi_grid)<np.sqrt(dx**2 + dy**2))] = 1/eta_grid[(np.abs(Psi_grid)<np.sqrt(dx**2 + dy**2))]
    weights[(Ls3_grid)<np.sqrt(dx**2 + dy**2)] = 0
    return np.sum(np.sum(curvature_ext*weights))/np.sum(np.sum(weights))

