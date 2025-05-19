/* This header file contains the definition of the parameters for the optimization problem */
#ifndef SYMMETRICSPACERAFT_H
#define SYMMETRICSPACERAFT_H

// Constants
const double PI{3.141592653589793}; // Pi------------------------------, -
const double DEG{PI/180.0};         // Degrees to radians conversion---, rad/deg

// Initial values
const double phi_0{0.0};            // Initial roll angle--------------, rad
const double theta_0{0.0};          // Initial pitch angle-------------, rad
const double psi_0{0.0};            // Initial yaw angle---------------, rad
const double wx_0{0.0};             // Initial roll rate---------------, rad/s
const double wy_0{0.0};             // Initial pitch rate--------------, rad/s
const double wz_0{0.0};             // Initial yaw rate----------------, rad/s

// Final values
const double phi_f{90.0 * DEG};     // Final roll angle----------------, rad
const double theta_f{0.0};          // Final pitch angle---------------, rad
const double psi_f{0.0};            // Final yaw angle-----------------, rad
const double wx_f{0.0};             // Final roll rate-----------------, rad/s
const double wy_f{0.0};             // Final pitch rate----------------, rad/s
const double wz_f{0.0};             // Final yaw rate------------------, rad/s

// Time parameters
const double T_0{2.4011};           // Time horizon initial guess------, s
const int n_stp{100};               // Number of time steps------------, -
const double dt_0{T_0/n_stp};       // Time step initial guess----------, s
const double dt_min{0.0001};        // Minimum time step---------------, s
const double dt_max{0.1};           // Maximum time step---------------, s

// Inertia parameters
const double i_x{1.0};              // Moment of inertia around x-axis, kg*m^2
const double i_y{1.0};              // Moment of inertia around y-axis, kg*m^2
const double i_z{1.0};              // Moment of inertia around z-axis, kg*m^2

// Actuator limits
const double tau_max{1.0};          // Maximum torque-----------------, Nm
const double tau_dot_max{1.0};      // Maximum torque rate of change--, Nm/s

#endif // SYMMETRICSPACERAFT_H