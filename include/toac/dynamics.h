#ifndef DYNAMICS_H
#define DYNAMICS_H

#include <casadi/casadi.hpp>
#include <toac/symmetric_spacecraft.h>

using namespace casadi;

struct Dynamics {
    MX X, U, dt;
    MX I;
    Function f, F;

    Dynamics();

private:

    // Takes a 3D vector w and returns a 4x4 skew-symmetric matrix
    MX skew4(const MX& w);

    // RK4 integrator
    MX rk4(const Function& f, const MX& x, const MX& u, const MX& dt);
};

// Converts Euler angles to a quaternion
DM euler2quat(const double& phi, const double& theta, const double& psi);

#endif // DYNAMICS_H