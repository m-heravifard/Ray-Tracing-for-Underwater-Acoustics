#@author: Mohammad E. Heravifard
#Supervisor: Prof. Parviz Ghadimi

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Sound-speed profile c(z) and its derivative c'(z)
# ----------------------------

def c(z):
    """
    Depth-dependent sound speed profile (Munk-like).
    z is depth in meters (negative downward in this setup).
    """
    zc = 1300.0
    zbar = 2.0 * (z - zc) / zc
    return 1500.0 * (1.0 + 0.00737 * (zbar - 1.0 + np.exp(-zbar)))


def cprime(z):
    """
    Derivative of c(z) with respect to z.
    """
    zc = 1300.0
    zbar = 2.0 * (z - zc) / zc
    return 1500.0 * 0.00737 * ((2.0 / zc) + (-2.0 / zc) * np.exp(-zbar))


# --- Alternative profiles (if you want to switch later) ---

# Parabolic profile:
# def c(z):
#     return (z**2.0) / 100000.0 + 1000.0
#
# def cprime(z):
#     return z / 50000.0

# Constant sound speed:
# def c(z):
#     return 1500.0
#
# def cprime(z):
#     return 0.0


# ----------------------------
# Ray equations (dr/dt, dz/dt, dtheta/dt)
# ----------------------------

def f1(t, r, z, theta):
    # dr/dt
    return c(z) * np.cos(theta)

def f2(t, r, z, theta):
    # dz/dt
    return c(z) * np.sin(theta)

def f3(t, r, z, theta):
    # dtheta/dt
    return -cprime(z) * np.cos(theta)


# ----------------------------
# Parameters and array allocation
# ----------------------------

n = 500   # number of integration steps
s = 121   # number of rays (angles from -60 to +60 degrees)

# Allocate (n+1) x s arrays
t     = np.zeros((n + 1, s))
r     = np.zeros((n + 1, s))
z     = np.zeros((n + 1, s))
theta = np.zeros((n + 1, s))

# Integration step size
h = dt = 0.01

# Launch angles from -60 to +60 degrees
angle_deg = np.linspace(-60.0, 60.0, s)
angle_rad = np.deg2rad(angle_deg)


# ----------------------------
# Initial conditions and RK4 integration
# ----------------------------

for j in range(s):
    # initial values for each ray
    t[0, j]     = 0.0
    r[0, j]     = 0.0
    z[0, j]     = -1000.0   # source depth at -1000 m
    theta[0, j] = angle_rad[j]

    for i in range(n):
        # Simple reflections at surface (z = 0) and bottom (z = -5000)
        if z[i, j] >= 0.0:
            z[i, j] = 0.0
            theta[i, j] = -theta[i, j]

        if z[i, j] <= -5000.0:
            z[i, j] = -5000.0
            theta[i, j] = -theta[i, j]

        # RK4 for (r, z, theta)
        k1_r = f1(t[i, j], r[i, j], z[i, j], theta[i, j])
        k1_z = f2(t[i, j], r[i, j], z[i, j], theta[i, j])
        k1_th = f3(t[i, j], r[i, j], z[i, j], theta[i, j])

        k2_r = f1(t[i, j] + h/2.0,
                  r[i, j] + h/2.0 * k1_r,
                  z[i, j] + h/2.0 * k1_z,
                  theta[i, j] + h/2.0 * k1_th)
        k2_z = f2(t[i, j] + h/2.0,
                  r[i, j] + h/2.0 * k1_r,
                  z[i, j] + h/2.0 * k1_z,
                  theta[i, j] + h/2.0 * k1_th)
        k2_th = f3(t[i, j] + h/2.0,
                   r[i, j] + h/2.0 * k1_r,
                   z[i, j] + h/2.0 * k1_z,
                   theta[i, j] + h/2.0 * k1_th)

        k3_r = f1(t[i, j] + h/2.0,
                  r[i, j] + h/2.0 * k2_r,
                  z[i, j] + h/2.0 * k2_z,
                  theta[i, j] + h/2.0 * k2_th)
        k3_z = f2(t[i, j] + h/2.0,
                  r[i, j] + h/2.0 * k2_r,
                  z[i, j] + h/2.0 * k2_z,
                  theta[i, j] + h/2.0 * k2_th)
        k3_th = f3(t[i, j] + h/2.0,
                   r[i, j] + h/2.0 * k2_r,
                   z[i, j] + h/2.0 * k2_z,
                   theta[i, j] + h/2.0 * k2_th)

        k4_r = f1(t[i, j] + h,
                  r[i, j] + h * k3_r,
                  z[i, j] + h * k3_z,
                  theta[i, j] + h * k3_th)
        k4_z = f2(t[i, j] + h,
                  r[i, j] + h * k3_r,
                  z[i, j] + h * k3_z,
                  theta[i, j] + h * k3_th)
        k4_th = f3(t[i, j] + h,
                   r[i, j] + h * k3_r,
                   z[i, j] + h * k3_z,
                   theta[i, j] + h * k3_th)

        # Update r, z, theta, t
        r[i + 1, j]     = r[i, j] + (h/6.0) * (k1_r + 2.0*k2_r + 2.0*k3_r + k4_r)
        z[i + 1, j]     = z[i, j] + (h/6.0) * (k1_z + 2.0*k2_z + 2.0*k3_z + k4_z)
        theta[i + 1, j] = theta[i, j] + (h/6.0) * (k1_th + 2.0*k2_th + 2.0*k3_th + k4_th)
        t[i + 1, j]     = t[i, j] + h


# ----------------------------
# Plot results
# ----------------------------

plt.figure(figsize=(8, 5))
for j in range(s):
    plt.plot(r[:, j], z[:, j], 'g', linewidth=0.8)

plt.xlabel('Range r (m)')
plt.ylabel('Depth z (m)')
plt.title('Ray Trajectories')
plt.grid(True)
# Depth is negative downward; if you want oceanographic style (0 at top):
# plt.gca().invert_yaxis()
plt.show()

