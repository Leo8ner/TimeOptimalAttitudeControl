#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <casadi/casadi.hpp>
#include <toac/symmetric_spacecraft.h>
#include <toac/helper_functions.h>
#include <toac/dynamics.h>

using namespace casadi;

class Optimizer {

    Function F; // Dynamics functions
    DM lb_U, ub_U, lb_dt, ub_dt;
    DM X_0, X_f;
    Opti opti;                   // Optimization problem
    Slice all;
    MX X, U, T, dt;
    MX p_X0, p_Xf;                      // Parameters
    std::string plugin, method;       // Solver plugin and method

    void SetDynamicConstraints();

public:

    Function solver; // Solver function
    Optimizer(const Dynamics& dyn, bool fixed_step = true);
};

#endif // OPTIMIZER_H