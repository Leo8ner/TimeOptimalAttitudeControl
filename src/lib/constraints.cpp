#include <toac/constraints.h>

using namespace casadi;

Constraints::Constraints() : lb_dt{dt_min}, ub_dt{dt_max} {
    // Constructor implementation
    DM q_0{euler2quat(phi_0, theta_0, psi_0)};                      // Initial quaternion
    X_0 = DM::vertcat({q_0, wx_0, wy_0, wz_0});                     // Initial state

    DM q_f{euler2quat(phi_f, theta_f, psi_f)};                      // Final quaternion
    X_f = DM::vertcat({q_f, wx_f, wy_f, wz_f});                     // Final state

    lb_U = DM::vertcat({-tau_max, -tau_max, -tau_max});             // Lower bound for torque
    ub_U = DM::vertcat({ tau_max,  tau_max,  tau_max});             // Upper bound for torque
}

void Constraints::setUdot() {
    // Set the constraints for the control input
    Udot = true;
}

// Converts Euler angles to a quaternion
DM euler2quat(const double& phi, const double& theta, const double& psi) {
    double q0{cos(phi/2) * cos(theta/2) * cos(psi/2) + sin(phi/2) * sin(theta/2) * sin(psi/2)};
    double q1{sin(phi/2) * cos(theta/2) * cos(psi/2) - cos(phi/2) * sin(theta/2) * sin(psi/2)};
    double q2{cos(phi/2) * sin(theta/2) * cos(psi/2) + sin(phi/2) * cos(theta/2) * sin(psi/2)};
    double q3{cos(phi/2) * cos(theta/2) * sin(psi/2) - sin(phi/2) * sin(theta/2) * cos(psi/2)};

    // Normalize the quaternion to eliminate numerical errors
    DM q{DM::vertcat({q0, q1, q2, q3})}; 
    q = q / norm_2(q); 

    return q;
}