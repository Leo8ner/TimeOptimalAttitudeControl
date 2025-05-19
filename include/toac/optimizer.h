#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <casadi/casadi.hpp>
#include <toac/symmetric_spacecraft.h>
#include <toac/dynamics.h>
#include <toac/constraints.h>
#include <iostream>

using namespace casadi;

class Optimizer {

    Function F; // Dynamics functions
    DM X_0, X_f;
    DM lb_U, ub_U, lb_dt, ub_dt;
    Opti opti {Opti()};                   // Optimization problem
    Slice all;
    MX X, U, dt;
    MX p_X0, p_Xf;                      // Parameters
    MX delta_U_max, delta_U;
    int n_X{7};                     // Number of states
    int n_U{3};                     // Number of controls

public:

    Function solver; // Solver function

    Optimizer(const Function& dyn, const Constraints& cons);

    //std::tuple<DM, DM, DM, DM> solve();

};

Function get_solver();


#endif // OPTIMIZER_H