import casadi as ca
import numpy as np
from traj_opt import Optimizer
import matplotlib.pyplot as plt

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
    ca.MX.sym('phi'),   # Roll angle--------------, rad
    ca.MX.sym('theta'), # Pitch angle-------------, rad
    ca.MX.sym('psi'),   # Yaw angle---------------, rad
    ca.MX.sym('wx'),    # Roll velocity-----------, rad/s
    ca.MX.sym('wz'),    # Pitch velocity----------, rad/s
    ca.MX.sym('wy'),    # Yaw velocity------------, rad/s
)
N_X = X.size1()         # Number of states--------, -

# Controls
U   = ca.vertcat(
    ca.MX.sym('tau_x'), # Torque around the x axis, Nm
    ca.MX.sym('tau_y'), # Torque around the y axis, Nm
    ca.MX.sym('taw_z'), # Torque around the z axis, Nm
)
N_U = U.size1()         # Number of controls------, -

# Time step
dt  = ca.MX.sym('dt')   # Time step---------------, s

### DYNAMICS ###
tau_x     = U[0]        # Torque around the x axis-----------, Nm              
tau_y     = U[1]        # Torque around the y axis-----------, Nm             
tau_z     = U[2]        # Torque around the z axis-----------, Nm

# Dynamics equations
phi       = X[0]        # Vertical position------------------, m   
theta     = X[1]        # Horizontal velocity----------------, m/s 
psi       = X[2]        # Vertical velocity------------------, m/s   
phi_dot   = X[3]        # Angular velocity-------------------, rad/s  
theta_dot = X[4]        # Horizontal acceleration------------, m/s^2
psi_dot   = X[5]        # Lunar gravity----------------------, m/s^2    
wx_dot    = (tau_x - (i_z -i_y)* / i_x # Vertical acceleration--------------, m/s^2
wy_dot    = tau_y / i_y # Vertical acceleration--------------, m/s^2
wz_dot    = tau_z / i_z # Vertical acceleration--------------, m/s^2

# State derivatives
X_dot     = ca.vertcat(phi_dot, theta_dot, psi_dot, wx_dot, wy_dot, wz_dot)

# RK4 integration
f         = ca.Function('f', [X, U], [X_dot], ["X", "U"], ["X_dot"]) # Continuous dynamics function  (X, U)------------> X_dot
F         = ca.Function('F', [X, U, dt], [rk4(f, X, U, dt)])         # Discretized dynamics function (X[k], U[k], dt)--> X[k + dt]

### OPTIMIZATION SETUP ###
# Final conditions [x, z, theta, vx, vz, wy, m_prop_used]
X_f   = [phi_f, theta_f, psi_f, wx_f, wy_f, wz_f]

# Initial conditions [x, z, theta, vx, vz, wy, m_prop_used]
X_i   = [phi_0, theta_0, psi_0, wx_0, wy_0, wz_0]

# Control bounds [T_m_r, T_m_c, T_m_l, T_thr_r, T_thr_l, gimbal]
lb_U  = [-tau_max, -tau_max, -tau_max] # Control lower bounds
ub_U  = [ tau_max,  tau_max,  tau_max] # Control upper bounds

opt   = Optimizer(N_X, N_U, T_0, N_steps, f, F, X_i, X_f, lb_U, ub_U, lb_dt, ub_dt)
opt_X, opt_U, opt_X_dot, opt_T = opt.solve()
