#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <casadi/casadi.hpp>
#include <toac/symmetric_spacecraft.h>
#include <toac/helper_functions.h>

using namespace casadi;

class Optimizer {

    Function F; // Dynamics functions
    DM lb_U, ub_U, lb_dt, ub_dt;
    DM X_0, X_f;
    Opti opti;                   // Optimization problem
    Slice all;
    MX X, U, T, dt;
    MX p_X0, p_Xf;                      // Parameters
    //DM X_guess, U_guess, dt_guess;     // Initial guesses for states, controls, and time steps

public:

    Function solver; // Solver function
    Optimizer(const Function& dyn, const std::string& plugin = "ipopt", 
               bool fixed_step = true);
};

#endif // OPTIMIZER_H