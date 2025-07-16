/* This header file contains the definition of the parameters for the optimization problem */
#ifndef SYMMETRICSPACERAFT_H
#define SYMMETRICSPACERAFT_H

// Constants
inline constexpr double PI{3.141592653589793}; // Pi------------------------------, -
inline constexpr double DEG{PI/180.0};         // Degrees to radians conversion---, rad/deg

// Initial values
inline constexpr double phi_0{0.0};            // Initial roll angle--------------, rad
inline constexpr double theta_0{0.0};          // Initial pitch angle-------------, rad
inline constexpr double psi_0{0.0};            // Initial yaw angle---------------, rad
inline constexpr double wx_0{0.0};             // Initial roll rate---------------, rad/s
inline constexpr double wy_0{0.0};             // Initial pitch rate--------------, rad/s
inline constexpr double wz_0{0.0};             // Initial yaw rate----------------, rad/s

// Final values
inline constexpr double phi_f{180.0 * DEG};     // Final roll angle----------------, rad
inline constexpr double theta_f{0.0};          // Final pitch angle---------------, rad
inline constexpr double psi_f{0.0};            // Final yaw angle-----------------, rad
inline constexpr double wx_f{0.0};             // Final roll rate-----------------, rad/s
inline constexpr double wy_f{0.0};             // Final pitch rate----------------, rad/s
inline constexpr double wz_f{0.0};             // Final yaw rate------------------, rad/s

// SIM parameters
constexpr double n_step_temp{phi_f/(90.0*DEG)*25}; // Number of steps calculator------, -
inline constexpr double T_0{3.2};              // Time horizon initial guess------, s
inline constexpr int n_stp{(int)n_step_temp}; // Number of time steps------------, -
inline constexpr double dt_0{T_0/n_stp};        // Time step initial guess---------, s
inline constexpr double dt_min{0.0};       // Minimum time step---------------, s
inline constexpr double dt_max{1.0};           // Maximum time step---------------, s
inline constexpr int n_states{7};              // Number of states----------------, -
inline constexpr int n_controls{3};            // Number of controls--------------, -

// Inertia parameters
inline constexpr double i_x{1.0};              // Moment of inertia around x-axis-, kg*m^2
inline constexpr double i_y{1.0};              // Moment of inertia around y-axis-, kg*m^2
inline constexpr double i_z{1.0};              // Moment of inertia around z-axis-, kg*m^2

// Actuator limits
inline constexpr double tau_max{1.0};          // Maximum torque------------------, Nm
inline constexpr double tau_dot_max{1.0};      // Maximum torque rate of change---, Nm/s


#endif // SYMMETRICSPACERAFT_H