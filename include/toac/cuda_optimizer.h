#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <casadi/casadi.hpp>
#include <toac/symmetric_spacecraft.h>
#include <toac/cuda_dynamics.h>
#include <toac/constraints.h>
#include <toac/helper_functions.h>
#include <filesystem>
#include <fstream>
#include <sstream>

using namespace casadi;

class Optimizer {

    Function F; // Dynamics functions
    DM lb_U, ub_U, lb_dt, ub_dt;
    DM X_0, X_f;
    Opti opti;                   // Optimization problem
    Slice all;
    MX X, U, T, dt;
    MX p_X0, p_Xf;                      // Parameters
    DM X_guess, U_guess, dt_guess;     // Initial guesses
    std::string csv_file;              // CSV file for initial guess

    void setupOptimizationProblem();
    void extractInitialGuess();


public:

    Function solver; // Solver function
    Optimizer(const Function& dyn, const Constraints& cons, const std::string& csv_data = "");

};

class BatchDynamics {

    Slice all;

public:
    BatchDynamics();
    Function F;
};

#endif // OPTIMIZER_H