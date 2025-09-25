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

struct Dynamics {
    
    Function F; 

    Dynamics(const std::string& plugin = "ipopt");
};

#endif // DYNAMICS_H