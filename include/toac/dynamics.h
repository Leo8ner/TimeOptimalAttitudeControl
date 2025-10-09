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
    std::string method, plugin; 

    Dynamics(const std::string& plugin_ = "ipopt", const std::string& method_ = "shooting");
};

#endif // DYNAMICS_H