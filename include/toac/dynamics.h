#ifndef DYNAMICS_H
#define DYNAMICS_H

#include <casadi/casadi.hpp>
#include <toac/symmetric_spacecraft.h>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <filesystem>

using namespace casadi;

struct Dynamics {

    SX X, U, dt;
    SX I;
    Function F, jac_F, jac_jac_F;
    int n_X, n_U;

    Dynamics();

private:

    // Takes a 3D vector w and returns a 4x4 skew-symmetric matrix
    SX skew4(const SX& w);

    // RK4 integrator
    SX rk4(const SX& x_dot, const SX& x, const SX& dt);
};

Function getDynamics();

#endif // DYNAMICS_H