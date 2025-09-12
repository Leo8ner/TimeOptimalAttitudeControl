/* This header file contains the definition of the parameters for the optimization problem */
#ifndef SYMMETRICSPACERAFT_H
#define SYMMETRICSPACERAFT_H

// Constants
inline constexpr double PI{3.141592653589793}; // Pi------------------------------, -
inline constexpr double DEG{PI/180.0};         // Degrees to radians conversion---, rad/deg

// SIM parameters
inline constexpr double T_0{2.4};              // Time horizon initial guess------, s
inline constexpr int n_stp{50};                 // Number of time steps------------, -
inline constexpr double dt_0{T_0/n_stp};        // Time step initial guess---------, s
inline constexpr double dt_min{0.0};       // Minimum time step---------------, s
inline constexpr double dt_max{1.0};           // Maximum time step---------------, s
inline constexpr int n_states{7};              // Number of states----------------, -
inline constexpr int n_quat{4};             // Number of quaternions------------------, -
inline constexpr int n_vel{3};              // Number of angular velocities-----, -
inline constexpr int n_controls{3};        // Number of controls--------------, -
inline constexpr int nnz{37};              // Number of nonzeros in Jacobian--, -
inline constexpr int n_states_total{n_states * n_stp}; // Total number of states across all steps, -
inline constexpr int n_controls_total{n_controls * n_stp}; // Total number of controls across all steps, -

// Inertia parameters
inline constexpr double i_x{1.0};              // Moment of inertia around x-axis-, kg*m^2
inline constexpr double i_y{1.0};              // Moment of inertia around y-axis-, kg*m^2
inline constexpr double i_z{1.0};              // Moment of inertia around z-axis-, kg*m^2

// Actuator limits
inline constexpr double tau_max{1.0};          // Maximum torque------------------, Nm
inline constexpr double tau_dot_max{1.0};      // Maximum torque rate of change---, Nm/s


#endif // SYMMETRICSPACERAFT_H