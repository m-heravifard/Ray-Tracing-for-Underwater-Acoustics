# Author: Mohammad E. Heravifard
#Ray tracing via RK4 for the eikonal solver in (r, z)
# Equations (range r, depth z; arclength s):
#   dr/ds = c * xi
#   dxi/ds = -(1/c^2) * (∂c/∂r)
#   dz/ds = c * zeta
#   dzeta/ds = -(1/c^2) * (∂c/∂z)
#
# Initial conditions at s=0:
#   r(0)=r0, z(0)=z0,
#   xi(0)=cos(theta0)/c(z0), zeta(0)=sin(theta0)/c(z0)
#
# Travel time accumulation:
#   dτ = ds / c
#
# Sound speed: Munk profile c(z) = c0 + Δc * (1 + ((z - za)/B)^2)
#   ∂c/∂z = Δc * 2 * (z - za) / B^2
#   ∂c/∂r = 0  (range-independent SSP)

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

# --------------------------
# Configuration
# --------------------------
@dataclass
class Config:
    # Geometry / domain
    r0: float = 0.0           # source range [m]
    z0: float = 1000.0        # source depth [m]
    z_surface: float = 0.0    # sea surface (z=0)
    z_bottom: float = 5000.0  # bottom depth [m]
    r_max: float = 120e3      # max horizontal range to trace [m]

    # Munk profile parameters (example deep-water values)
    c0: float = 1500.0        # reference speed [m/s]
    delta_c: float = 100.0    # Δc [m/s]
    za: float = 1300.0        # axis depth [m]
    B: float = 1300.0         # scale depth [m]

    # RK4 / stepping
    ds_shallow: float = 40.0  # step when z < z_switch
    ds_deep: float = 20.0     # step when z >= z_switch
    z_switch: float = 500.0   # per manuscript note on step size usage
    n_steps_max: int = 200000 # safety cap

    # Launch fan
    theta_min_deg: int = -30
    theta_max_deg: int = 30
    theta_step_deg: int = 2

    # Plotting
    ray_color: str | None = None   # let matplotlib choose defaults if None
    figsize: tuple = (9, 5)
    show_travel_time: bool = True  # annotate a few rays with τ at end

cfg = Config()

# --------------------------
# Sound speed profile (Munk) and gradients
# --------------------------
def c_munk(z, c0=cfg.c0, delta_c=cfg.delta_c, za=cfg.za, B=cfg.B):
    zscaled = (z - za) / B
    return c0 + delta_c * (1.0 + zscaled**2)

def dc_dz_munk(z, delta_c=cfg.delta_c, za=cfg.za, B=cfg.B):
    return delta_c * 2.0 * (z - za) / (B**2)

def c_and_grads(r, z):
    # Range-independent Munk: ∂c/∂r = 0
    c = c_munk(z)
    dcdz = dc_dz_munk(z)
    dcdr = 0.0
    return c, dcdr, dcdz

# --------------------------
# RHS of the first-order ray system
# state y = [r, z, xi, zeta]
# --------------------------
def f_rhs(y):
    r, z, xi, zeta = y
    c, dcdr, dcdz = c_and_grads(r, z)
    drds = c * xi
    dzds = c * zeta
    dxids = -(1.0 / c**2) * dcdr
    dzetads = -(1.0 / c**2) * dcdz
    return np.array([drds, dzds, dxids, dzetads], dtype=float), c

# --------------------------
# One RK4 step in arclength s
# --------------------------
def rk4_step(y, ds):
    k1, c1 = f_rhs(y)
    k2, c2 = f_rhs(y + 0.5 * ds * k1)
    k3, c3 = f_rhs(y + 0.5 * ds * k2)
    k4, c4 = f_rhs(y + ds * k3)
    y_next = y + (ds / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    # Use c from the midpoint (k2/k3) to accumulate τ more stably; here average them
    c_mid = 0.5 * (c2 + c3)
    return y_next, c_mid

# --------------------------
# Simple specular reflections at surface and bottom
# Reverse vertical component: zeta -> -zeta, clamp z to boundary
# --------------------------
def reflect_if_needed(y):
    r, z, xi, zeta = y
    bounced = False
    if z < cfg.z_surface:
        z = cfg.z_surface
        zeta = -zeta
        bounced = True
    elif z > cfg.z_bottom:
        z = cfg.z_bottom
        zeta = -zeta
        bounced = True
    return np.array([r, z, xi, zeta], dtype=float), bounced

# --------------------------
# Trace one ray until it exits the domain or hits caps
# Returns arrays of r, z, τ (travel time), and number of bounces
# --------------------------
def trace_one_ray(theta_deg):
    theta = np.deg2rad(theta_deg)
    c0_here = c_munk(cfg.z0)

    # Initial state
    y = np.array([
        cfg.r0,                    # r
        cfg.z0,                    # z
        np.cos(theta)/c0_here,     # xi
        np.sin(theta)/c0_here      # zeta
    ], dtype=float)

    rs, zs, taus = [y[0]], [y[1]], [0.0]
    bounces = 0

    for _ in range(cfg.n_steps_max):
        # Choose ds based on depth (per manuscript note on step size)
        ds = cfg.ds_shallow if y[1] < cfg.z_switch else cfg.ds_deep

        y_new, c_mid = rk4_step(y, ds)
        # accumulate travel time τ: dτ = ds / c
        tau_new = taus[-1] + (ds / max(c_mid, 1e-9))  # guard against degenerate c

        # Handle reflections
        y_new, bounced = reflect_if_needed(y_new)
        if bounced:
            bounces += 1

        rs.append(y_new[0]); zs.append(y_new[1]); taus.append(tau_new)
        y = y_new

        # Termination: range reached or out of bounds horizontally
        if y[0] > cfg.r_max:
            break

    return np.array(rs), np.array(zs), np.array(taus), bounces

# --------------------------
# Run a fan of rays and plot
# --------------------------
angles = np.arange(cfg.theta_min_deg, cfg.theta_max_deg + 1e-9, cfg.theta_step_deg)
plt.figure(figsize=cfg.figsize)

annotated = 0
for th in angles:
    r_arr, z_arr, tau_arr, nb = trace_one_ray(th)
    plt.plot(r_arr/1000.0, z_arr, lw=1.0)  # range in km

    # Annotate a few rays with final travel time (optional)
    if cfg.show_travel_time and annotated < 6 and len(r_arr) > 5:
        plt.text(r_arr[-1]/1000.0, z_arr[-1],
                 f"θ0={th}°, τ={tau_arr[-1]:.1f}s",
                 fontsize=8)
        annotated += 1

plt.gca().invert_yaxis()
plt.xlabel("Range r (km)")
plt.ylabel("Depth z (m)")
plt.title("Ray paths (RK4 eikonal solver, Munk SSP)")
plt.grid(True, alpha=0.3)

# Draw surface and bottom
plt.axhline(cfg.z_surface, ls="--", lw=1, alpha=0.6)
plt.axhline(cfg.z_bottom, ls="--", lw=1, alpha=0.6)
plt.tight_layout()
plt.show()

