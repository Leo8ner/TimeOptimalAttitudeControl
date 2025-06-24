#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <casadi/casadi.hpp>
#include <toac/symmetric_spacecraft.h>
#include <toac/cuda_dynamics.h>
#include <toac/constraints.h>
#include <filesystem>

using namespace casadi;

class CUDAOptimizer {

    Function F; // Dynamics functions
    DM lb_U, ub_U, lb_dt, ub_dt;
    DM X_0, X_f;
    Opti opti;                   // Optimization problem
    Slice all;
    MX X, U, T, dt;
    MX p_X0, p_Xf;                      // Parameters

    void setupOptimizationProblem();

public:

    Function solver; // Solver function
    CUDAOptimizer(const Function& dyn, const Constraints& cons);

};

Function get_solver();

DM stateInterpolator(const DM& x0, const DM& xf, int n_stp);
DM inputInterpolator(const auto& x0, const auto& xf, int n_stp);
DM ratesInterpolator(const auto& x0, const auto& xf, int n_stp);

#endif // OPTIMIZER_H