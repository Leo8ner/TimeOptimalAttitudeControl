#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <casadi/casadi.hpp>
#include <toac/symmetric_spacecraft.h>
#include <toac/cuda_dynamics.h>
#include <helper_functions.h>
#include <filesystem>
#include <fstream>
#include <sstream>

using namespace casadi;

class Optimizer {

    Function F; // Dynamics functions
    DM lb_U, ub_U, lb_dt, ub_dt;
    Opti opti;                   // Optimization problem
    Slice all;
    MX X, U, T, dt;
    MX p_X0, p_Xf;                      // Parameters

    void setupOptimizationProblem();
    void extractInitialGuess();


public:

    Function solver; // Solver function
    Optimizer(const Function& dyn);

};

class BatchDynamics {

    Slice all;

public:
    BatchDynamics();
    Function F;
};

#endif // OPTIMIZER_H