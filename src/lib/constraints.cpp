#include <toac/constraints.h>

using namespace casadi;

Constraints::Constraints() : lb_dt{dt_min}, ub_dt{dt_max} {

    lb_U = DM::vertcat({-tau_max, -tau_max, -tau_max});             // Lower bound for torque
    ub_U = DM::vertcat({ tau_max,  tau_max,  tau_max});             // Upper bound for torque
}

void Constraints::setUdot() {
    // Set the constraints for the control input
    Udot = true;
}