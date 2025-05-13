#ifndef CONSTRAINTS_H
#define CONSTRAINTS_H

#include <casadi/casadi.hpp>
#include <toac/symmetric_spacecraft.h>

using namespace casadi;

struct Constraints {
    DM X_0, X_f;
    DM lb_U, ub_U;
    DM lb_dt, ub_dt;

    Constraints();
};

// Converts Euler angles to a quaternion
DM euler2quat(const double& phi, const double& theta, const double& psi);


#endif // CONSTRAINTS_H