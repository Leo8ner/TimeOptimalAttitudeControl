#include <casadi/casadi.hpp>
#include <toac/symmetric_spacecraft.h>
//#include <toac/optimizer.h>
#include <toac/dynamics.h>

using namespace casadi;

int main() {

    // Dynamics
    Dynamics dyn; // Create an instance of the Dynamics class

    // Constraints
    DM q_0{euler2quat(phi_0, theta_0, psi_0)};                        // Initial quaternion
    DM X_0{DM::vertcat({q_0, wx_0, wy_0, wz_0})};                     // Initial state

    DM q_f{euler2quat(phi_f, theta_f, psi_f)};                        // Final quaternion
    DM X_f{DM::vertcat({q_f, wx_f, wy_f, wz_f})};                     // Final state

    DM lb_U{DM::vertcat({-tau_max, -tau_max, -tau_max})};             // Lower bound for torque
    DM ub_U{DM::vertcat({ tau_max,  tau_max,  tau_max})};             // Upper bound for torque

    //Optimizer opti{dyn, X_0, X_f, lb_U, ub_U, lb_dt, ub_dt};          // Create an instance of the Optimizer class

    return 0;
}