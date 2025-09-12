#ifndef DYNAMICS_H
#define DYNAMICS_H

#include <toac/symmetric_spacecraft.h>
#include <toac/helper_functions.h>
#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>
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

#endif // DYNAMICS_H