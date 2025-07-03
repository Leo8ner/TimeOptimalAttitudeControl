#ifndef DYNAMICS_H
#define DYNAMICS_H

#include <toac/symmetric_spacecraft.h>
#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>
#include <filesystem>
#include <casadi/casadi.hpp>

using namespace casadi;

class ExplicitDynamics {

    SX X, U, dt;

public:
    Function F; 

    ExplicitDynamics();
};

struct ImplicitDynamics {
    
    Function F; 

    ImplicitDynamics(const std::string& plugin = "ipopt");
};

// Takes a 3D vector w and returns a 4x4 skew-symmetric matrix
SX skew4(const SX& w);

// RK4 integrator
SX rk4(const SX& x_dot, const SX& x, const SX& dt);

#endif // DYNAMICS_H