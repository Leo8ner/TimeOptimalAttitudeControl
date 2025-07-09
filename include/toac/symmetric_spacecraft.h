/* This header file contains the definition of the parameters for the optimization problem */
#ifndef SYMMETRICSPACERAFT_H
#define SYMMETRICSPACERAFT_H

// Constants
inline constexpr float PI{3.1415927f};         // Pi------------------------------, -
inline constexpr float DEG{PI/180.0f};         // Degrees to radians conversion---, rad/deg

// Initial values
inline constexpr float phi_0{0.0f};            // Initial roll angle--------------, rad
inline constexpr float theta_0{0.0f};          // Initial pitch angle-------------, rad
inline constexpr float psi_0{0.0f};            // Initial yaw angle---------------, rad
inline constexpr float wx_0{0.0f};             // Initial roll rate---------------, rad/s
inline constexpr float wy_0{0.0f};             // Initial pitch rate--------------, rad/s
inline constexpr float wz_0{0.0f};             // Initial yaw rate----------------, rad/s

// Final values
inline constexpr float phi_f{180.0f * DEG};     // Final roll angle----------------, rad
inline constexpr float theta_f{0.0f};          // Final pitch angle---------------, rad
inline constexpr float psi_f{0.0f};            // Final yaw angle-----------------, rad
inline constexpr float wx_f{0.0f};             // Final roll rate-----------------, rad/s
inline constexpr float wy_f{0.0f};             // Final pitch rate----------------, rad/s
inline constexpr float wz_f{0.0f};             // Final yaw rate------------------, rad/s

// SIM parameters
constexpr float n_step_temp{phi_f/(90.0f*DEG)*25};       // Number of steps calculator------, -
inline constexpr float T_0{3.2f};              // Time horizon initial guess------, s
inline constexpr int n_stp{(int)n_step_temp}; // Number of time steps------------, -
inline constexpr float dt_0{T_0/n_stp};        // Time step initial guess---------, s
inline constexpr float dt_min{0.0f};       // Minimum time step---------------, s
inline constexpr float dt_max{1.0f};           // Maximum time step---------------, s
inline constexpr int n_states{7};              // Number of states----------------, -
inline constexpr int n_controls{3};            // Number of controls--------------, -

// Inertia parameters
inline constexpr float i_x{1.0f};              // Moment of inertia around x-axis-, kg*m^2
inline constexpr float i_y{1.0f};              // Moment of inertia around y-axis-, kg*m^2
inline constexpr float i_z{1.0f};              // Moment of inertia around z-axis-, kg*m^2

// Actuator limits
inline constexpr float tau_max{1.0f};          // Maximum torque------------------, Nm
inline constexpr float tau_dot_max{1.0f};      // Maximum torque rate of change---, Nm/s


#endif // SYMMETRICSPACERAFT_H