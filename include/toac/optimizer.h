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
    MX X, U, T, dt;
    MX delta_U_max, delta_U;

public:

    Optimizer(const Function& dyn, const Constraints& cons);

    std::tuple<DM, DM, DM, DM> solve();

};

#endif // OPTIMIZER_H