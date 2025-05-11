import casadi as ca
import numpy as np
from traj_opt import Optimizer
import matplotlib.pyplot as plt

def euler2quat(roll, pitch, yaw):
    """
    Convert Euler angles to quaternion.
    """
    cy = ca.cos(yaw * 0.5)
    sy = ca.sin(yaw * 0.5)
    cr = ca.cos(roll * 0.5)
    sr = ca.sin(roll * 0.5)
    cp = ca.cos(pitch * 0.5)
    sp = ca.sin(pitch * 0.5)

    q0 = cy * cr * cp + sy * sr * sp
    q1 = cy * sr * cp - sy * cr * sp
    q2 = cy * cr * sp + sy * sr * cp
    q3 = sy * cr * cp - cy * sr * sp

    q = ca.vertcat(q0, q1, q2, q3) # Quaternion [q0, q1, q2, q3]-----------------, -
    q = q / ca.norm_2(q) # Normalize quaternion----------------------------, -
    return q

### CONSTANTS AND PARAMETERS ###
# Physical constants
deg     = ca.pi / 180 # Degrees to radians-------------, rad/deg
percent = 0.01        # Percentage---------------------, -

# Initial conditions
phi_0   = 0           # Initial roll angle-------------, rad
theta_0 = 0           # Initial pitch angle------------, rad
psi_0   = 0           # Initial yaw angle--------------, rad
wx_0    = 0           # Initial roll velocity----------, rad/s
wy_0    = 0           # Initial pitch velocity---------, rad/s
wz_0    = 0           # Initial yaw velocity-----------, rad/s

# Final conditions
phi_f   = 90 * deg    # Final roll angle---------------, rad
theta_f = 0           # Final pitch angle--------------, rad
psi_f   = 0           # Final yaw angle----------------, rad
wx_f    = 0           # Final roll velocity------------, rad/s
wy_f    = 0           # Final pitch velocity-----------, rad/s
wz_f    = 0           # Final yaw velocity-------------, rad/s

# Inertia parameters
i_x     = 1           # Moment of inertia around x-axis, kg*m^2
i_y     = 1           # Moment of inertia around y-axis, kg*m^2
i_z     = 1           # Moment of inertia around z-axis, kg*m^2

# Time & solver parameters
T_0     = 29.5        # Total time of flight-----------, s 
N_steps = 100         # Number of time steps-----------, -
lb_dt   = 0.001       # Minimum time step--------------, s
ub_dt   = 0.1         # Maximum time step--------------, s

# Actuator limits
tau_max = 1           # Maximum torque-----------------, Nm

### SYMBOLIC VARIABLES ###
# States
X   = ca.vertcat(
    ca.SX.sym('q', 4),  # Quaternion [q0, q1, q2, q3], -
    ca.SX.sym('w', 3),  # Angular rate [wx, wy, wz]--, rad/s
)
N_X = X.size1()         # Number of states-----------, -

# Controls
U   = ca.vertcat(
    ca.SX.sym('tau', 3) # Torque---------------------, Nm
)
N_U = U.size1()         # Number of controls---------, -

# Time step
dt  = ca.SX.sym('dt')   # Time step------------------, s

### DYNAMICS ###

# Dynamics equations
q         = X[0:4]      # Quaternion-------------------------, -
w         = X[4:7]      # Angular rate-----------------------, rad/s
tau       = U[0:3]      # Torque-----------------------------, Nm

S = ca.SX.zeros(3, 3) # Skew-symmetric matrix
S[0, 1] = -w[2]
S[0, 2] = w[1]
S[1, 0] = w[2]
S[1, 2] = -w[0]
S[2, 0] = -w[1]
S[2, 1] = w[0]

I = ca.diag([i_x, i_y, i_z]) # Inertia matrix------------------, kg*m^2

q_dot     = 0.5 * ca.mtimes(S, q) # Quaternion derivative----------------, -
w_dot     = ca.inv(I) * (tau - ca.cross(w, ca.mtimes(I, w))) # Angular rate derivative----------------, rad/s

# State derivatives
X_dot     = ca.vertcat(q_dot, w_dot)

# RK4 integration
f         = ca.Function('f', [X, U], [X_dot], ["X", "U"], ["X_dot"]) # Continuous dynamics function  (X, U)------------> X_dot
F         = ca.Function('F', [X, U, dt], [rk4(f, X, U, dt)])         # Discretized dynamics function (X[k], U[k], dt)--> X[k + dt]

### OPTIMIZATION SETUP ###
# Final conditions [x, z, theta, vx, vz, wy, m_prop_used]
q_f   = euler2quat(phi_f, theta_f, psi_f) # Final quaternion------------------, -
X_f   = ca.vertcat(q_f, wx_f, wy_f, wz_f)

# Initial conditions [x, z, theta, vx, vz, wy, m_prop_used]
q_0   = euler2quat(phi_0, theta_0, psi_0) # Initial quaternion------------------, -
X_0   = ca.vertcat(q_0, wx_0, wy_0, wz_0) # Initial state----------------------------, -

# Control bounds [T_m_r, T_m_c, T_m_l, T_thr_r, T_thr_l, gimbal]
lb_U  = ca.vertcat(-tau_max, -tau_max, -tau_max) # Control lower bounds
ub_U  = ca.vertcat( tau_max,  tau_max,  tau_max) # Control upper bounds

opt   = Optimizer(N_X, N_U, T_0, N_steps, f, F, X_0, X_f, lb_U, ub_U, lb_dt, ub_dt)
opt_X, opt_U, opt_X_dot, opt_T = opt.solve()
