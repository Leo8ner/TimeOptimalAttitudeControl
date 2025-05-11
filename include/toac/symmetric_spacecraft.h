/* This header file contains the definition of the parameters for the optimization problem */
#ifndef SYMMETRICSPACERAFT_H
#define SYMMETRICSPACERAFT_H

// Constants
const float PI {3.1415926f};     // Value of pi---------------------, -
const float DEG {PI/180.0f};     // Degrees to radians conversion---, rad/deg

// Initial values
const float phi_0 {0.0f};        // Initial roll angle--------------, rad
const float theta_0 {0.0f};      // Initial pitch angle-------------, rad
const float psi_0 {0.0f};        // Initial yaw angle---------------, rad
const float wx_0 {0.0f};         // Initial roll rate---------------, rad/s
const float wy_0 {0.0f};         // Initial pitch rate--------------, rad/s
const float wz_0 {0.0f};         // Initial yaw rate----------------, rad/s

// Final values
const float phi_f {90.0f * DEG}; // Final roll angle----------------, rad
const float theta_f {0.0f};      // Final pitch angle---------------, rad
const float psi_f {0.0f};        // Final yaw angle-----------------, rad
const float wx_f {0.0f};         // Final roll rate-----------------, rad/s
const float wy_f {0.0f};         // Final pitch rate----------------, rad/s
const float wz_f {0.0f};         // Final yaw rate------------------, rad/s

// Time parameters
const float T_0 {3.0f};          // Time horizon initial guess------, s
const int n_stp {100};           // Number of time steps------------, -
const float lb_dt {0.0001f};     // Minimum time step---------------, s
const float ub_dt {0.1f};        // Maximum time step---------------, s

// Inertia parameters
const float i_x {1.0f};           // Moment of inertia around x-axis, kg*m^2
const float i_y {1.0f};           // Moment of inertia around y-axis, kg*m^2
const float i_z {1.0f};           // Moment of inertia around z-axis, kg*m^2

// Actuator limits
const float tau_max {1.0f};       // Maximum torque-----------------, Nm

#endif // SYMMETRICSPACERAFT_H